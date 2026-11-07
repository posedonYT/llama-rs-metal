use super::*;
impl Session {
    pub(super) fn encode(
        &self,
        b: &mut Batch<'_>,
        tokens: &[u32],
        pos: usize,
        logits: bool,
    ) -> Result<()> {
        let c = &self.model.config;
        let count = tokens.len();
        let h = c.hidden;
        let q = c.heads * c.head_dim;
        let kv = c.kv_heads * c.head_dim;
        let s = |slot: Slot, n: usize| {
            self.scratch
                .view(slot, if slot == Slot::Logits { n } else { n * count })
        };
        let hidden = s(Slot::Hidden, h)?;
        let norm = s(Slot::Normalized, h)?;
        let projected = s(Slot::Branch, h)?;
        b.embeddings(&self.model.embedding, tokens, &hidden)?;
        for (layer, state) in self.model.layers.iter().zip(&self.states) {
            b.norm(&hidden, &layer.norm, &norm, count, h, c.eps)?;
            match (&layer.mixer, state) {
                (
                    Mixer::Attention {
                        q: wq,
                        k: wk,
                        v: wv,
                        out,
                        qn,
                        kn,
                    },
                    State::Attention { key, value },
                ) => {
                    let qp = s(Slot::QUERY_PROJECTION, q * if c.hybrid() { 2 } else { 1 })?;
                    let qn_buf = s(Slot::QUERY_NORMALIZED, q)?;
                    let qrot = s(Slot::QUERY_ROTATED, q)?;
                    let kp = s(Slot::KEY_PROJECTION, kv)?;
                    let kn_buf = s(Slot::KEY_NORMALIZED, kv)?;
                    let krot = s(Slot::KEY_ROTATED, kv)?;
                    let vp = s(Slot::VALUE_PROJECTION, kv)?;
                    let attn = s(Slot::ATTENTION_OUTPUT, q)?;
                    b.matmul(wq, &norm, &qp, count)?;
                    b.matmul(wk, &norm, &kp, count)?;
                    b.matmul(wv, &norm, &vp, count)?;
                    if let Some(w) = qn {
                        b.norm(&qp, w, &qn_buf, c.heads * count, c.head_dim * 2, c.eps)?
                    } else {
                        b.copy(&qp, &qn_buf, 0)?
                    }
                    if let Some(w) = kn {
                        b.norm(&kp, w, &kn_buf, c.kv_heads * count, c.head_dim, c.eps)?
                    } else {
                        b.copy(&kp, &kn_buf, 0)?
                    }
                    b.rope_many(
                        &qn_buf,
                        &qrot,
                        c.head_dim,
                        c.rotary,
                        c.rotary_layout == RotaryLayout::SplitHalf,
                        pos,
                        c.rope_base,
                        c.rope_scale,
                        &self.model.factors,
                        c.heads,
                    )?;
                    b.rope_many(
                        &kn_buf,
                        &krot,
                        c.head_dim,
                        c.rotary,
                        c.rotary_layout == RotaryLayout::SplitHalf,
                        pos,
                        c.rope_base,
                        c.rope_scale,
                        &self.model.factors,
                        c.kv_heads,
                    )?;
                    b.copy(&krot, key, pos * kv)?;
                    b.copy(&vp, value, pos * kv)?;
                    if count == 1 && pos >= 128 {
                        // K projection is dead after its cache write; reuse that
                        // slot for partial attention sums, avoiding allocations.
                        let chunks = (pos + 1).div_ceil(128).min(32);
                        let partial = self
                            .scratch
                            .view(Slot::KEY_PROJECTION, c.heads * chunks * (c.head_dim + 2))?;
                        b.attention_decode(
                            &qrot,
                            key,
                            value,
                            &attn,
                            &partial,
                            c.heads,
                            c.kv_heads,
                            c.head_dim,
                            pos + 1,
                        )?;
                    } else {
                        b.attention_many(
                            &qrot, key, value, &attn, c.heads, c.kv_heads, c.head_dim, pos, count,
                        )?;
                    }
                    if c.hybrid() {
                        let gate = s(Slot::ATTENTION_GATE, q)?;
                        b.gather(&qp, &gate, c.head_dim, c.head_dim * 2, c.head_dim)?;
                        b.gate(&attn, &gate, &attn)?
                    }
                    b.matmul(out, &attn, &projected, count)?;
                }
                (
                    Mixer::Linear {
                        qkv,
                        z,
                        legacy,
                        alpha,
                        beta,
                        ba,
                        conv: wc,
                        norm: wn,
                        out,
                        aw,
                        dt,
                    },
                    State::Linear { conv, state },
                ) => {
                    let dk = c.linear_key_dim;
                    let kh = c.linear_key_heads;
                    let vh = c.linear_value_heads;
                    let dv = c.linear_value_dim;
                    let kd = dk * kh;
                    let vd = vh * dv;
                    let width = kd * 2 + vd;
                    let qkv_buf = s(Slot::LINEAR_QKV, width)?;
                    let zbuf = s(Slot::LINEAR_GATE, vd)?;
                    let a = s(Slot::LINEAR_ALPHA, vh)?;
                    let be = s(Slot::LINEAR_BETA, vh)?;
                    let cv = s(Slot::LINEAR_CONV, width)?;
                    let delta = s(Slot::LINEAR_DELTA, vd)?;
                    let normalized = s(Slot::LINEAR_NORMALIZED, vd)?;
                    if let (Some(qkv), Some(z)) = (qkv, z) {
                        b.matmul(qkv, &norm, &qkv_buf, count)?;
                        b.matmul(z, &norm, &zbuf, count)?
                    } else if let Some(w) = legacy {
                        let mixed = s(Slot::LINEAR_MIXED, width + vd)?;
                        b.matmul(w, &norm, &mixed, count)?;
                        let stride = (width + vd) / kh;
                        let part = dv * (vh / kh);
                        for t in 0..count {
                            let src = mixed.view(t * (width + vd), width + vd)?;
                            let dst = qkv_buf.view(t * width, width)?;
                            b.gather(&src, &dst.view(0, kd)?, dk, stride, 0)?;
                            b.gather(&src, &dst.view(kd, kd)?, dk, stride, dk)?;
                            b.gather(&src, &dst.view(kd * 2, vd)?, part, stride, dk * 2)?;
                            b.gather(&src, &zbuf.view(t * vd, vd)?, part, stride, dk * 2 + part)?;
                        }
                    } else {
                        return Err(bad("linear projections"));
                    }
                    if let (Some(wa), Some(wb)) = (alpha, beta) {
                        b.matmul(wa, &norm, &a, count)?;
                        b.matmul(wb, &norm, &be, count)?
                    } else if let Some(w) = ba {
                        let mixed = s(Slot::LINEAR_MIXED, vh * 2)?;
                        b.matmul(w, &norm, &mixed, count)?;
                        let part = vh / kh;
                        b.gather(&mixed, &be, part, part * 2, 0)?;
                        b.gather(&mixed, &a, part, part * 2, part)?
                    }
                    b.conv(&qkv_buf, wc, conv, &cv)?;
                    b.delta(
                        &cv,
                        &a,
                        &be,
                        aw,
                        dt,
                        state,
                        &delta,
                        dk,
                        kh,
                        vh,
                        dv,
                        c.linear_norm_eps,
                        c.recurrent_head_layout == RecurrentHeadLayout::Interleaved,
                        c.l2_normalization == L2Normalization::AddEpsilon,
                    )?;
                    b.norm(&delta, wn, &normalized, vh * count, dv, c.eps)?;
                    b.swiglu(&zbuf, &normalized, &normalized)?;
                    b.matmul(out, &normalized, &projected, count)?;
                }
                _ => return Err(bad("layer state mismatch")),
            }
            b.add(&hidden, &projected, &hidden)?;
            b.norm(&hidden, &layer.post, &norm, count, h, c.eps)?;
            match &layer.ffn {
                FeedForward::Dense(d) => encode_dense(b, d, &norm, &projected, &self.scratch)?,
                FeedForward::Moe {
                    router,
                    gate,
                    up,
                    down,
                    shared,
                    shared_gate,
                } => {
                    for token in 0..count {
                        let norm = norm.view(token * h, h)?;
                        let projected = projected.view(token * h, h)?;
                        let s = |slot: Slot, n: usize| self.scratch.view(slot, n);
                        let count = c.experts_used;
                        let ff = gate.dims()[1];
                        let routes = s(Slot::ROUTER, router.dims()[1])?;
                        let ids = s(Slot::EXPERT_IDS, count)?;
                        let weights = s(Slot::EXPERT_PROBABILITIES, count)?;
                        let gates = s(Slot::EXPERT_GATE, count * ff)?;
                        let ups = s(Slot::EXPERT_UP, count * ff)?;
                        let expert_out = s(Slot::EXPERT_OUTPUT, count * h)?;
                        b.matmul(router, &norm, &routes, 1)?;
                        b.route(&routes, &ids, &weights, c.normalize_expert_weights)?;
                        b.expert_matmul(gate, &norm, &ids, &gates, false)?;
                        b.expert_matmul(up, &norm, &ids, &ups, false)?;
                        b.swiglu(&gates, &ups, &gates)?;
                        b.expert_matmul(down, &gates, &ids, &expert_out, true)?;
                        b.mix(&expert_out, &weights, &projected)?;
                        let shared_out = s(Slot::SHARED_OUTPUT, h)?;
                        encode_dense(b, shared, &norm, &shared_out, &self.scratch)?;
                        let sg = s(Slot::SHARED_GATE, 1)?;
                        b.matmul(shared_gate, &norm, &sg, 1)?;
                        b.gate_scalar(&shared_out, &sg, &shared_out)?;
                        b.add(&projected, &shared_out, &projected)?;
                    }
                }
            }
            b.add(&hidden, &projected, &hidden)?;
        }
        if logits {
            let last = hidden.view((count - 1) * h, h)?;
            let norm = norm.view(0, h)?;
            b.norm(&last, &self.model.output_norm, &norm, 1, h, c.eps)?;
            b.matmul(&self.model.output, &norm, &s(Slot::Logits, c.vocab)?, 1)?;
            b.argmax(
                &s(Slot::Logits, c.vocab)?,
                &self.scratch.view(Slot::Greedy, 1)?,
            )?;
        }
        Ok(())
    }
}
fn encode_dense(
    b: &mut Batch<'_>,
    d: &Dense,
    input: &Buffer,
    out: &Buffer,
    scratch: &Workspace,
) -> Result<()> {
    let width = d.gate.dims()[1];
    let count = input.len() / d.gate.dims()[0];
    let gate = scratch.view(Slot::FfnGate, width * count)?;
    let up = scratch.view(Slot::FfnUp, width * count)?;
    b.matmul(&d.gate, input, &gate, count)?;
    b.matmul(&d.up, input, &up, count)?;
    b.swiglu(&gate, &up, &gate)?;
    b.matmul(&d.down, &gate, out, count)?;
    Ok(())
}
