use super::*;
impl Tokenizer {
    pub fn from_gguf(f: &Gguf) -> Result<Self> {
        let list = f
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(Value::array)
            .ok_or_else(|| Error::Tokenizer("missing vocabulary".into()))?;
        let tokens: Vec<String> = list
            .iter()
            .map(|v| {
                v.string()
                    .map(str::to_string)
                    .ok_or_else(|| Error::Tokenizer("non-string vocabulary entry".into()))
            })
            .collect::<Result<_>>()?;
        let ids: HashMap<_, _> = tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();
        let scores = f
            .metadata
            .get("tokenizer.ggml.scores")
            .and_then(Value::array)
            .map(|a| a.iter().map(|v| v.float().unwrap_or(-1e9) as f32).collect())
            .unwrap_or_else(|| vec![0.; tokens.len()]);
        let types: Vec<u64> = f
            .metadata
            .get("tokenizer.ggml.token_type")
            .and_then(Value::array)
            .map(|a| a.iter().map(|v| v.uint().unwrap_or(1)).collect())
            .unwrap_or_else(|| vec![1; tokens.len()]);
        if scores.len() != tokens.len() || types.len() != tokens.len() {
            return Err(Error::Tokenizer("vocabulary metadata lengths".into()));
        }
        let model = f.str("tokenizer.ggml.model").unwrap_or("llama");
        let kind = match model {
            "llama" => Kind::SentencePiece,
            "gpt2" => {
                let pre = f.str("tokenizer.ggml.pre").unwrap_or("gpt-2");
                let pattern = match pre {
                    "llama-bpe" => {
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                    }
                    "qwen35" => {
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                    }
                    "qwen2" => {
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                    }
                    "gpt-2" | "default" => {
                        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
                    }
                    _ => return Err(Error::Tokenizer(format!("unsupported pre-tokenizer {pre}"))),
                };
                let merges = f
                    .metadata
                    .get("tokenizer.ggml.merges")
                    .and_then(Value::array)
                    .ok_or_else(|| Error::Tokenizer("missing BPE merges".into()))?;
                let mut ranks = HashMap::new();
                for (i, m) in merges.iter().enumerate() {
                    let (a, b) = m
                        .string()
                        .and_then(|s| s.split_once(' '))
                        .ok_or_else(|| Error::Tokenizer("invalid BPE merge".into()))?;
                    ranks.insert((a.into(), b.into()), i);
                }
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
                let mut encode = vec!['\0'; 256];
                let mut decode = HashMap::new();
                for (b, ch) in base.into_iter().zip(chars) {
                    let ch = char::from_u32(ch).unwrap();
                    encode[b as usize] = ch;
                    decode.insert(ch, b as u8);
                }
                Kind::ByteBpe {
                    pattern: Box::new(
                        Regex::new(pattern).map_err(|e| Error::Tokenizer(e.to_string()))?,
                    ),
                    ranks,
                    encode,
                    decode,
                }
            }
            _ => return Err(Error::Tokenizer(format!("unsupported tokenizer {model}"))),
        };
        let id = |key: &str| -> Result<Option<u32>> {
            let v = f.uint(key);
            if v.is_some_and(|v| v >= tokens.len()) {
                return Err(Error::Tokenizer(format!("{key} outside vocabulary")));
            }
            Ok(v.map(|v| v as u32))
        };
        let bos = id("tokenizer.ggml.bos_token_id")?;
        let mut eos = Vec::new();
        for key in [
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.eom_token_id",
        ] {
            if let Some(i) = id(key)? {
                eos.push(i)
            }
        }
        for name in [
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|im_end|>",
            "<|endoftext|>",
        ] {
            if let Some(i) = ids.get(name) {
                eos.push(*i)
            }
        }
        eos.sort_unstable();
        eos.dedup();
        let add_bos = match f.metadata.get("tokenizer.ggml.add_bos_token") {
            Some(Value::Bool(b)) => *b,
            _ => model == "llama",
        };
        let prefix = match f.metadata.get("tokenizer.ggml.add_space_prefix") {
            Some(Value::Bool(b)) => *b,
            _ => true,
        };
        let mut special: Vec<_> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| types[*i] == 3 || types[*i] == 4)
            .map(|(i, t)| (t.clone(), i as u32))
            .filter(|(s, _)| !s.is_empty())
            .collect();
        special.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        Ok(Self {
            tokens,
            ids,
            scores,
            types,
            kind,
            bos,
            eos,
            add_bos,
            prefix,
            special,
        })
    }
}
