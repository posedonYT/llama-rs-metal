//! Keep source-framework names at the file boundary. The executor sees semantic roles.
pub(super) fn canonical(name: &str, hybrid: bool) -> Option<String> {
    let name = name.strip_prefix("language_model.").unwrap_or(name);
    let name = if let Some(rest) = name.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else {
        name.to_string()
    };
    match name.as_str() {
        "model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.norm.weight" => return Some("output_norm.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        _ => {}
    }
    let rest = name.strip_prefix("model.layers.")?;
    let (layer, role) = rest.split_once('.')?;
    let _: usize = layer.parse().ok()?;
    let role = match role {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => {
            if hybrid {
                "post_attention_norm.weight"
            } else {
                "ffn_norm.weight"
            }
        }
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "mlp.gate.weight" => "ffn_gate_inp.weight",
        "mlp.switch_mlp.gate_proj.weight" => "ffn_gate_exps.weight",
        "mlp.switch_mlp.up_proj.weight" => "ffn_up_exps.weight",
        "mlp.switch_mlp.down_proj.weight" => "ffn_down_exps.weight",
        "mlp.shared_expert.gate_proj.weight" => "ffn_gate_shexp.weight",
        "mlp.shared_expert.up_proj.weight" => "ffn_up_shexp.weight",
        "mlp.shared_expert.down_proj.weight" => "ffn_down_shexp.weight",
        "mlp.shared_expert_gate.weight" => "ffn_gate_inp_shexp.weight",
        "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
        "linear_attn.in_proj_z.weight" => "attn_gate.weight",
        "linear_attn.in_proj_a.weight" => "ssm_alpha.weight",
        "linear_attn.in_proj_b.weight" => "ssm_beta.weight",
        "linear_attn.in_proj_qkvz.weight" => "ssm_in.weight",
        "linear_attn.in_proj_ba.weight" => "ssm_ba.weight",
        "linear_attn.A_log" => "ssm_a",
        "linear_attn.dt_bias" => "ssm_dt.bias",
        "linear_attn.conv1d.weight" => "ssm_conv1d.weight",
        "linear_attn.norm.weight" => "ssm_norm.weight",
        "linear_attn.out_proj.weight" => "ssm_out.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{role}"))
}
