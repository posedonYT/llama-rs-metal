use llama_rs::{TextDecoder, Tokenizer};
use serde_json::json;

#[test]
fn hf_json_normalization_special_tokens_and_split_utf8() {
    let path = std::env::temp_dir().join(format!("llama-rs-hf-tokenizer-{}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    let level =
        json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true});
    let definition = json!({
        "version":"1.0","truncation":null,"padding":null,
        "added_tokens":[{"id":2,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":3,"content":"<停止>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "normalizer":{"type":"NFC"},"pre_tokenizer":level,"post_processor":null,"decoder":level,
        "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"Ã":0,"©":1,"<eos>":2,"<停止>":3},"merges":[]}
    });
    std::fs::write(path.join("tokenizer.json"), definition.to_string()).unwrap();
    std::fs::write(
        path.join("tokenizer_config.json"),
        r#"{"eos_token":{"content":"<eos>"},"add_bos_token":false}"#,
    )
    .unwrap();
    let tokenizer = Tokenizer::from_huggingface(&path).unwrap();
    assert_eq!(tokenizer.encode("é", false, true).unwrap(), vec![0, 1]);
    assert_eq!(
        tokenizer.encode("e\u{301}", false, true).unwrap(),
        vec![0, 1]
    );
    assert_eq!(tokenizer.eos, vec![2]);
    assert!(tokenizer.decode_piece(2, false).unwrap().is_empty());
    assert_eq!(tokenizer.decode_piece(2, true).unwrap(), b"<eos>");
    assert_eq!(
        tokenizer.decode_piece(3, true).unwrap(),
        "<停止>".as_bytes()
    );
    assert!(tokenizer.decode_piece(3, false).unwrap().is_empty());
    assert!(tokenizer.decode_piece(1000, false).is_err());
    let mut decoder = TextDecoder::default();
    assert_eq!(decoder.push(&tokenizer.decode_piece(0, false).unwrap()), "");
    assert_eq!(
        decoder.push(&tokenizer.decode_piece(1, false).unwrap()),
        "é"
    );
    std::fs::write(
        path.join("generation_config.json"),
        r#"{"eos_token_id":4294967296}"#,
    )
    .unwrap();
    assert!(Tokenizer::from_huggingface(&path).is_err());
    std::fs::write(
        path.join("generation_config.json"),
        r#"{"eos_token_id":1000}"#,
    )
    .unwrap();
    assert!(Tokenizer::from_huggingface(&path).is_err());
    std::fs::remove_file(path.join("generation_config.json")).unwrap();
    let mut unsupported = definition;
    unsupported["decoder"] = json!({"type":"ByteFallback"});
    std::fs::write(path.join("tokenizer.json"), unsupported.to_string()).unwrap();
    let error = Tokenizer::from_huggingface(&path)
        .err()
        .expect("decoder rejected");
    assert!(error.to_string().contains("ByteLevel"));
    std::fs::remove_dir_all(path).unwrap();
}
