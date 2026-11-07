use super::*;
impl Tokenizer {
    pub fn decode_piece(&self, id: u32, special: bool) -> Result<Vec<u8>> {
        if id as usize >= self.tokens.len() {
            return Err(Error::Tokenizer("unknown token".into()));
        }
        if let Kind::HuggingFace {
            tokenizer,
            byte_level,
            byte_decoder,
        } = &self.kind
        {
            if !special && self.types.get(id as usize) == Some(&3) {
                return Ok(Vec::new());
            }
            if *byte_level {
                if let Some(token) = tokenizer.id_to_token(id) {
                    // ByteLevel's decoder falls back to the token's literal
                    // UTF-8 if any character lies outside its byte alphabet
                    // (notably Unicode added tokens). Delay UTF-8 assembly to
                    // TextDecoder so ordinary byte pieces remain lossless.
                    let bytes: Option<Vec<u8>> = token
                        .chars()
                        .map(|c| byte_decoder.get(&c).copied())
                        .collect();
                    return Ok(bytes.unwrap_or_else(|| token.into_bytes()));
                }
                return Err(Error::Tokenizer("unknown token".into()));
            }
            return tokenizer
                .decode(&[id], !special)
                .map(String::into_bytes)
                .map_err(|e| Error::Tokenizer(e.to_string()));
        }
        let token = self
            .tokens
            .get(id as usize)
            .ok_or_else(|| Error::Tokenizer("unknown token".into()))?;
        if !special && [3, 4].contains(&self.types[id as usize]) {
            return Ok(Vec::new());
        }
        match &self.kind {
            Kind::HuggingFace { .. } => unreachable!("handled above"),

            Kind::SentencePiece => {
                if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
                    Ok(vec![
                        u8::from_str_radix(&token[3..5], 16)
                            .map_err(|e| Error::Tokenizer(e.to_string()))?,
                    ])
                } else {
                    Ok(token.replace('▁', " ").into_bytes())
                }
            }
            Kind::ByteBpe { decode, .. } => token
                .chars()
                .map(|c| {
                    decode
                        .get(&c)
                        .copied()
                        .ok_or_else(|| Error::Tokenizer("invalid byte alphabet".into()))
                })
                .collect(),
        }
    }
}

/// Preserves UTF-8 sequences split across generated tokens.
#[derive(Default)]
pub struct TextDecoder {
    pending: Vec<u8>,
}
impl TextDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut result = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    result.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    result.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    match e.error_len() {
                        Some(n) => {
                            result.push('�');
                            self.pending.drain(..valid + n);
                        }
                        None => {
                            self.pending.drain(..valid);
                            break;
                        }
                    }
                }
            }
        }
        result
    }
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_utf8() {
        let mut d = TextDecoder::default();
        assert_eq!(d.push(&[0xd0]), "");
        assert_eq!(d.push(&[0x9f]), "П");
        assert_eq!(d.finish(), "");
    }
}
