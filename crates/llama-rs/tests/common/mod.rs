use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static ID: AtomicU64 = AtomicU64::new(0);
pub struct Fixture {
    pub path: PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
fn string(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}
pub fn fixture(arch: &str) -> Fixture {
    let hybrid = arch != "llama";
    let moe = arch == "qwen35moe" || arch == "qwen3next";
    let mut meta: BTreeMap<String, (u32, Vec<u8>)> = BTreeMap::new();
    let mut put_s = |k: &str, s: &str| {
        let mut b = Vec::new();
        string(&mut b, s);
        meta.insert(k.into(), (8, b));
    };
    put_s("general.architecture", arch);
    put_s("tokenizer.ggml.model", "llama");
    let mut vocab = Vec::new();
    vocab.extend(8u32.to_le_bytes());
    vocab.extend(32u64.to_le_bytes());
    for i in 0..32 {
        string(&mut vocab, &format!("tok{i}"))
    }
    meta.insert("tokenizer.ggml.tokens".into(), (9, vocab));
    let mut types = Vec::new();
    types.extend(4u32.to_le_bytes());
    types.extend(32u64.to_le_bytes());
    for _ in 0..32 {
        types.extend(1u32.to_le_bytes())
    }
    meta.insert("tokenizer.ggml.token_type".into(), (9, types));
    for (k, v) in [
        ("embedding_length", 32),
        ("block_count", 2),
        ("attention.head_count", 2),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 16),
        ("rope.dimension_count", if hybrid { 8 } else { 16 }),
        ("context_length", 64),
        ("feed_forward_length", 48),
        ("full_attention_interval", 2),
        ("expert_count", 4),
        ("expert_feed_forward_length", 16),
        ("expert_shared_feed_forward_length", 8),
        ("ssm.state_size", 8),
        ("ssm.group_count", 2),
        ("ssm.time_step_rank", 4),
        ("ssm.inner_size", 32),
        ("ssm.conv_kernel", 4),
        ("expert_used_count", 2),
    ] {
        if k.starts_with("expert") && !moe {
            continue;
        }
        meta.insert(
            format!("{arch}.{k}"),
            (4, (v as u32).to_le_bytes().to_vec()),
        );
    }
    meta.insert(
        format!("{arch}.attention.layer_norm_rms_epsilon"),
        (6, 1e-5f32.to_le_bytes().to_vec()),
    );
    if hybrid {
        let mut sections = Vec::new();
        sections.extend(4u32.to_le_bytes());
        sections.extend(4u64.to_le_bytes());
        for n in [2u32, 1, 1, 0] {
            sections.extend(n.to_le_bytes())
        }
        meta.insert(format!("{arch}.rope.dimension_sections"), (9, sections));
    }
    let mut weights: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let mut seed = 51u64;
    let mut add = |name: String, dims: &[usize], norm: bool| {
        let data = (0..dims.iter().product())
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let x = ((seed >> 32) as u32) as f32 / (u32::MAX as f32);
                if norm {
                    0.7 + x * 0.6
                } else {
                    (x - 0.5) * 0.16
                }
            })
            .collect();
        weights.insert(name, (dims.to_vec(), data));
    };
    add("token_embd.weight".into(), &[32, 32], false);
    add("output_norm.weight".into(), &[32], true);
    add("output.weight".into(), &[32, 32], false);
    for i in 0..2 {
        let p = format!("blk.{i}.");
        add(format!("{p}attn_norm.weight"), &[32], true);
        add(
            format!(
                "{p}{}",
                if hybrid {
                    "post_attention_norm.weight"
                } else {
                    "ffn_norm.weight"
                }
            ),
            &[32],
            true,
        );
        if hybrid && i == 0 {
            if arch == "qwen3next" {
                add(format!("{p}ssm_in.weight"), &[32, 96], false);
                add(format!("{p}ssm_ba.weight"), &[32, 8], false);
            } else {
                add(format!("{p}attn_qkv.weight"), &[32, 64], false);
                add(format!("{p}attn_gate.weight"), &[32, 32], false);
                add(format!("{p}ssm_alpha.weight"), &[32, 4], false);
                add(format!("{p}ssm_beta.weight"), &[32, 4], false);
            }
            add(format!("{p}ssm_conv1d.weight"), &[4, 64], false);
            add(format!("{p}ssm_dt.bias"), &[4], false);
            add(format!("{p}ssm_a"), &[4], false);
            add(format!("{p}ssm_norm.weight"), &[8], true);
            add(format!("{p}ssm_out.weight"), &[32, 32], false);
        } else {
            add(
                format!("{p}attn_q.weight"),
                &[32, if hybrid { 64 } else { 32 }],
                false,
            );
            add(format!("{p}attn_k.weight"), &[32, 16], false);
            add(format!("{p}attn_v.weight"), &[32, 16], false);
            add(format!("{p}attn_output.weight"), &[32, 32], false);
            if hybrid {
                add(format!("{p}attn_q_norm.weight"), &[16], true);
                add(format!("{p}attn_k_norm.weight"), &[16], true);
            }
        }
        if moe {
            add(format!("{p}ffn_gate_inp.weight"), &[32, 4], false);
            add(format!("{p}ffn_gate_exps.weight"), &[32, 16, 4], false);
            add(format!("{p}ffn_up_exps.weight"), &[32, 16, 4], false);
            add(format!("{p}ffn_down_exps.weight"), &[16, 32, 4], false);
            add(format!("{p}ffn_gate_inp_shexp.weight"), &[32], false);
            add(format!("{p}ffn_gate_shexp.weight"), &[32, 8], false);
            add(format!("{p}ffn_up_shexp.weight"), &[32, 8], false);
            add(format!("{p}ffn_down_shexp.weight"), &[8, 32], false);
        } else {
            add(format!("{p}ffn_gate.weight"), &[32, 48], false);
            add(format!("{p}ffn_up.weight"), &[32, 48], false);
            add(format!("{p}ffn_down.weight"), &[48, 32], false);
        }
    }
    for (name, (_, v)) in &mut weights {
        if name.ends_with(".ssm_a") {
            v.fill(-0.5)
        }
    }
    let mut data = Vec::new();
    let mut infos = Vec::new();
    for (name, (dims, values)) in &weights {
        while data.len() % 32 != 0 {
            data.push(0)
        }
        string(&mut infos, name);
        infos.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            infos.extend((*d as u64).to_le_bytes());
        }
        infos.extend(0u32.to_le_bytes());
        infos.extend((data.len() as u64).to_le_bytes());
        for x in values {
            data.extend(x.to_le_bytes());
        }
    }
    let mut file = b"GGUF".to_vec();
    file.extend(3u32.to_le_bytes());
    file.extend((weights.len() as u64).to_le_bytes());
    file.extend((meta.len() as u64).to_le_bytes());
    for (k, (ty, v)) in meta {
        string(&mut file, &k);
        file.extend(ty.to_le_bytes());
        file.extend(v);
    }
    file.extend(infos);
    while file.len() % 32 != 0 {
        file.push(0)
    }
    file.extend(data);
    let path = std::env::temp_dir().join(format!(
        "llama-rust-{}-{}-{arch}.gguf",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, file).unwrap();
    Fixture { path }
}
