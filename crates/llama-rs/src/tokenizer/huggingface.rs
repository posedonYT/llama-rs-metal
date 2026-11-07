use super::*;
impl Tokenizer {
    pub fn from_huggingface(directory: &std::path::Path) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let vocab = tokenizer.get_vocab(true);
        let size = vocab
            .values()
            .copied()
            .max()
            .map(|v| v as usize + 1)
            .unwrap_or(0);
        if size == 0 || size > 1_000_000 {
            return Err(Error::Tokenizer("invalid vocabulary size".into()));
        }
        let mut tokens = vec![String::new(); size];
        for (token, id) in &vocab {
            tokens[*id as usize] = token.clone()
        }
        let mut types = vec![1; size];
        for (id, token) in tokenizer.get_added_tokens_decoder() {
            if token.special {
                types[id as usize] = 3
            }
        }
        let load = |name: &str| -> Result<serde_json::Value> {
            let path = directory.join(name);
            if path.exists() {
                serde_json::from_slice(
                    &std::fs::read(path).map_err(|e| Error::Tokenizer(e.to_string()))?,
                )
                .map_err(|e| Error::Tokenizer(e.to_string()))
            } else {
                Ok(serde_json::Value::Null)
            }
        };
        let settings = load("tokenizer_config.json")?;
        let config = load("config.json")?;
        let generation = load("generation_config.json")?;
        let text = config.get("text_config").unwrap_or(&config);
        let token_name = |v: &serde_json::Value| {
            v.as_str()
                .or_else(|| v.get("content").and_then(serde_json::Value::as_str))
                .map(str::to_string)
        };
        let checked_id = |value: u64| -> Result<u32> {
            let id = u32::try_from(value)
                .map_err(|_| Error::Tokenizer("token ID exceeds u32".into()))?;
            if id as usize >= size {
                return Err(Error::Tokenizer(
                    "special token ID outside vocabulary".into(),
                ));
            }
            Ok(id)
        };
        let named_bos = settings
            .get("bos_token")
            .and_then(token_name)
            .and_then(|s| vocab.get(&s).copied());
        let bos = if let Some(id) = named_bos {
            Some(id)
        } else {
            text.get("bos_token_id")
                .and_then(serde_json::Value::as_u64)
                .map(checked_id)
                .transpose()?
        };
        let mut eos = Vec::new();
        for value in [text.get("eos_token_id"), generation.get("eos_token_id")]
            .into_iter()
            .flatten()
        {
            if let Some(id) = value.as_u64() {
                eos.push(checked_id(id)?)
            } else if let Some(ids) = value.as_array() {
                for id in ids {
                    if let Some(id) = id.as_u64() {
                        eos.push(checked_id(id)?)
                    }
                }
            }
        }
        if let Some(id) = settings
            .get("eos_token")
            .and_then(token_name)
            .and_then(|s| vocab.get(&s).copied())
        {
            eos.push(id)
        }
        for name in ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>"] {
            if let Some(id) = vocab.get(name) {
                eos.push(*id)
            }
        }
        eos.sort_unstable();
        eos.dedup();
        let mut base: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
        let mut chars = base.clone();
        let mut n = 0;
        for b in 0..=255 {
            if !base.contains(&b) {
                base.push(b);
                chars.push(256 + n);
                n += 1
            }
        }
        let byte_decoder = base
            .into_iter()
            .zip(chars)
            .map(|(b, c)| (char::from_u32(c).unwrap(), b as u8))
            .collect();
        let definition = load("tokenizer.json")?;
        fn has_byte_level(v: &serde_json::Value) -> bool {
            if v.get("type").and_then(serde_json::Value::as_str) == Some("ByteLevel") {
                return true;
            }
            v.as_object()
                .is_some_and(|o| o.values().any(has_byte_level))
                || v.as_array().is_some_and(|a| a.iter().any(has_byte_level))
        }
        let byte_level = definition.get("decoder").is_some_and(has_byte_level);
        if !byte_level {
            return Err(Error::Tokenizer(
                "native MLX text streaming currently requires a ByteLevel decoder".into(),
            ));
        }
        Ok(Self {
            tokens,
            ids: vocab.into_iter().collect(),
            scores: vec![],
            types,
            kind: Kind::HuggingFace {
                tokenizer: Box::new(tokenizer),
                byte_level,
                byte_decoder,
            },
            bos,
            eos,
            add_bos: settings
                .get("add_bos_token")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(bos.is_some()),
            prefix: false,
            special: vec![],
        })
    }
}
