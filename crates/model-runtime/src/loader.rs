use super::*;
fn expect(w: &HashMap<String, Weight>, name: &str, dims: &[usize]) -> Result<Weight> {
    let v = w
        .get(name)
        .ok_or_else(|| bad(format!("missing tensor {name}")))?;
    if v.dims() != dims {
        return Err(bad(format!(
            "{name}: expected {dims:?}, got {:?}",
            v.dims()
        )));
    }
    Ok(v.clone())
}
fn dense(w: &HashMap<String, Weight>, prefix: &str, h: usize, suffix: &str) -> Result<Dense> {
    let name = format!("{prefix}ffn_gate{suffix}.weight");
    let g = w.get(&name).ok_or_else(|| bad(&name))?;
    if g.dims().len() != 2 || g.dims()[0] != h {
        return Err(bad(name));
    }
    let n = g.dims()[1];
    Ok(Dense {
        gate: g.clone(),
        up: expect(w, &format!("{prefix}ffn_up{suffix}.weight"), &[h, n])?,
        down: expect(w, &format!("{prefix}ffn_down{suffix}.weight"), &[n, h])?,
    })
}
impl Model {
    /// Includes retained mapping pages and small transformed RoPE/SSM buffers.
    pub fn weight_allocation_bound(gpu: &Gpu, file: &Checkpoint) -> Result<usize> {
        let c = &file.config;
        let auxiliaries = c
            .rotary
            .checked_mul(2)
            .and_then(|n| {
                c.layers
                    .checked_mul(c.linear_value_heads)?
                    .checked_mul(8)?
                    .checked_add(n)
            })
            .ok_or_else(|| bad("weight allocation estimate overflow"))?;
        gpu.weight_allocation_bound(file)?
            .checked_add(auxiliaries)
            .ok_or_else(|| bad("weight allocation estimate overflow"))
    }
    pub fn load(gpu: &mut Gpu, file: Rc<Checkpoint>, budget: usize) -> Result<Self> {
        let config = file.config.clone();
        let weight_bytes = Self::weight_allocation_bound(gpu, &file)?;
        if weight_bytes > budget {
            return Err(Error::Memory {
                required: weight_bytes,
                available: budget,
            });
        }
        let weights = gpu.weights(&file)?;
        let c = &config;
        let h = c.hidden;
        let q = c.heads * c.head_dim;
        let kv = c.kv_heads * c.head_dim;
        let dk = c.linear_key_dim;
        let kh = c.linear_key_heads;
        let vh = c.linear_value_heads;
        let vd = c.linear_value_dim * vh;
        let conv = 2 * dk * kh + vd;
        let embedding = expect(&weights, "token_embd.weight", &[h, c.vocab])?;
        let output = match c.tied_embeddings {
            Some(true) => embedding.clone(),
            Some(false) => expect(&weights, "output.weight", &[h, c.vocab])?,
            None => {
                if weights.contains_key("output.weight") {
                    expect(&weights, "output.weight", &[h, c.vocab])?
                } else {
                    embedding.clone()
                }
            }
        };
        let output_norm = expect(&weights, "output_norm.weight", &[h])?;
        let mut layers = Vec::new();
        let mut max_width = h.max(q * 2).max(conv).max(vd).max(vh * 2);
        for i in 0..c.layers {
            let p = format!("blk.{i}.");
            let get = |name: &str, dims: &[usize]| expect(&weights, &format!("{p}{name}"), dims);
            let norm = get("attn_norm.weight", &[h])?;
            let post = get(
                if c.hybrid() {
                    "post_attention_norm.weight"
                } else {
                    "ffn_norm.weight"
                },
                &[h],
            )
            .or_else(|_| get("attn_post_norm.weight", &[h]))?;
            let linear = c.hybrid()
                && (weights.contains_key(&format!("{p}attn_qkv.weight"))
                    || weights.contains_key(&format!("{p}ssm_in.weight")));
            let mixer = if linear {
                if c.conv_kernel < 2 {
                    return Err(bad("conv kernel must be >=2"));
                }
                let modern = weights.contains_key(&format!("{p}attn_qkv.weight"));
                let qkv = if modern {
                    Some(get("attn_qkv.weight", &[h, conv])?)
                } else {
                    None
                };
                let z = if modern {
                    Some(get("attn_gate.weight", &[h, vd])?)
                } else {
                    None
                };
                let legacy = if modern {
                    None
                } else {
                    max_width = max_width.max(conv + vd);
                    Some(get("ssm_in.weight", &[h, conv + vd])?)
                };
                let next = c.architecture == "qwen3next";
                let ba = if next {
                    Some(get("ssm_ba.weight", &[h, vh * 2])?)
                } else {
                    None
                };
                let alpha = if next {
                    None
                } else {
                    Some(get("ssm_alpha.weight", &[h, vh])?)
                };
                let beta = if next {
                    None
                } else {
                    Some(get("ssm_beta.weight", &[h, vh])?)
                };
                let aw_values = file.decode(&format!("{p}ssm_a"))?;
                let dt_values = file.decode(&format!("{p}ssm_dt.bias"))?;
                if aw_values.len() != vh
                    || dt_values.len() != vh
                    || aw_values.iter().any(|x| !x.is_finite() || *x > 0.)
                    || dt_values.iter().any(|x| !x.is_finite())
                {
                    return Err(bad("SSM A/dt"));
                }
                Mixer::Linear {
                    qkv,
                    z,
                    legacy,
                    alpha,
                    beta,
                    ba,
                    conv: get("ssm_conv1d.weight", &[c.conv_kernel, conv])?,
                    norm: get("ssm_norm.weight", &[c.linear_value_dim])?,
                    out: get("ssm_out.weight", &[vd, h])?,
                    aw: gpu.upload(&aw_values)?,
                    dt: gpu.upload(&dt_values)?,
                }
            } else {
                Mixer::Attention {
                    q: get("attn_q.weight", &[h, q * if c.hybrid() { 2 } else { 1 }])?,
                    k: get("attn_k.weight", &[h, kv])?,
                    v: get("attn_v.weight", &[h, kv])?,
                    out: get("attn_output.weight", &[q, h])?,
                    qn: if c.hybrid() {
                        Some(get("attn_q_norm.weight", &[c.head_dim])?)
                    } else {
                        None
                    },
                    kn: if c.hybrid() {
                        Some(get("attn_k_norm.weight", &[c.head_dim])?)
                    } else {
                        None
                    },
                }
            };
            let ffn = if let Some(router) = weights.get(&format!("{p}ffn_gate_inp.weight")) {
                if router.dims().len() != 2
                    || router.dims()[0] != h
                    || c.experts_used == 0
                    || c.experts_used > router.dims()[1]
                {
                    return Err(bad("MoE routing dimensions"));
                }
                let experts = router.dims()[1];
                let gate = weights
                    .get(&format!("{p}ffn_gate_exps.weight"))
                    .ok_or_else(|| bad("expert gates"))?;
                if gate.dims().len() != 3 || gate.dims()[0] != h || gate.dims()[2] != experts {
                    return Err(bad("expert gates dimensions"));
                }
                let ff = gate.dims()[1];
                max_width = max_width
                    .max(ff * c.experts_used)
                    .max(h * c.experts_used)
                    .max(experts);
                let shared = dense(&weights, &p, h, "_shexp")?;
                max_width = max_width.max(shared.gate.dims()[1]);
                let shared_gate = get("ffn_gate_inp_shexp.weight", &[h])?.reshape(&[h, 1])?;
                FeedForward::Moe {
                    router: router.clone(),
                    gate: gate.clone(),
                    up: get("ffn_up_exps.weight", &[h, ff, experts])?,
                    down: get("ffn_down_exps.weight", &[ff, h, experts])?,
                    shared,
                    shared_gate,
                }
            } else {
                let d = dense(&weights, &p, h, "")?;
                max_width = max_width.max(d.gate.dims()[1]);
                FeedForward::Dense(d)
            };
            layers.push(Layer {
                norm,
                post,
                mixer,
                ffn,
            });
        }
        let factors = file.rope_factors.clone();
        if factors.len() != c.rotary / 2 || factors.iter().any(|f| !f.is_finite() || *f <= 0.) {
            return Err(bad("RoPE factors"));
        }
        let factors = gpu.upload(&factors)?;
        gpu.warmup()?;
        Ok(Self {
            config,
            file,
            layers,
            embedding,
            output,
            output_norm,
            factors,
            weight_bytes,
            scratch_elements: max_width,
        })
    }
}
