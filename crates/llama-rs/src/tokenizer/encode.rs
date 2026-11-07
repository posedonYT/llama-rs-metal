use super::*;
impl Tokenizer {
    pub fn encode(&self, text: &str, add_bos: bool, parse_special: bool) -> Result<Vec<u32>> {
        if let Kind::HuggingFace { tokenizer, .. } = &self.kind {
            let encoding = if parse_special {
                tokenizer.encode(text, add_bos)
            } else {
                let mut plain = tokenizer.clone();
                plain.set_encode_special_tokens(true);
                plain.encode(text, add_bos)
            }
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
            return Ok(encoding.get_ids().to_vec());
        }
        let mut out = Vec::new();
        if add_bos && self.add_bos {
            out.push(
                self.bos
                    .ok_or_else(|| Error::Tokenizer("BOS required but missing".into()))?,
            )
        }
        if !parse_special {
            self.encode_plain(text, &mut out, true)?;
            return Ok(out);
        }
        let mut pos = 0;
        let mut first = true;
        while pos < text.len() {
            let next = self
                .special
                .iter()
                .filter_map(|(s, id)| text[pos..].find(s).map(|i| (pos + i, s, *id)))
                .min_by(|a, b| a.0.cmp(&b.0).then(b.1.len().cmp(&a.1.len())));
            if let Some((start, s, id)) = next {
                if start > pos {
                    self.encode_plain(&text[pos..start], &mut out, first)?;
                    first = false;
                }
                out.push(id);
                pos = start + s.len();
            } else {
                self.encode_plain(&text[pos..], &mut out, first)?;
                break;
            }
        }
        Ok(out)
    }
    pub(super) fn encode_plain(&self, text: &str, out: &mut Vec<u32>, first: bool) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        match &self.kind {
            Kind::HuggingFace { .. } => {
                return Err(Error::Tokenizer("internal tokenizer dispatch".into()));
            }

            Kind::SentencePiece => {
                let text = format!(
                    "{}{}",
                    if first && self.prefix { "▁" } else { "" },
                    text.replace(' ', "▁")
                );
                let mut parts: Vec<String> = text.chars().map(|c| c.to_string()).collect();
                loop {
                    let best = parts
                        .windows(2)
                        .enumerate()
                        .filter_map(|(i, p)| {
                            let merged = format!("{}{}", p[0], p[1]);
                            self.ids
                                .get(&merged)
                                .filter(|id| self.types[**id as usize] == 1)
                                .map(|id| (i, self.scores[*id as usize], merged))
                        })
                        .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)));
                    if let Some((i, _, merged)) = best {
                        parts[i] = merged;
                        parts.remove(i + 1);
                    } else {
                        break;
                    }
                }
                for p in parts {
                    if let Some(id) = self.ids.get(&p) {
                        out.push(*id)
                    } else {
                        for byte in p.as_bytes() {
                            let name = format!("<0x{byte:02X}>");
                            out.push(*self.ids.get(&name).ok_or_else(|| {
                                Error::Tokenizer(format!("missing byte fallback {name}"))
                            })?)
                        }
                    }
                }
            }
            Kind::ByteBpe {
                pattern,
                ranks,
                encode,
                ..
            } => {
                let mut end = 0;
                for m in pattern.find_iter(text) {
                    let m = m.map_err(|e| Error::Tokenizer(e.to_string()))?;
                    if m.start() != end {
                        return Err(Error::Tokenizer("pre-tokenizer skipped bytes".into()));
                    }
                    end = m.end();
                    let mut parts: Vec<String> = m
                        .as_str()
                        .bytes()
                        .map(|b| encode[b as usize].to_string())
                        .collect();
                    loop {
                        let best = parts
                            .windows(2)
                            .enumerate()
                            .filter_map(|(i, p)| {
                                ranks.get(&(p[0].clone(), p[1].clone())).map(|r| (i, *r))
                            })
                            .min_by_key(|(i, r)| (*r, *i));
                        if let Some((i, _)) = best {
                            let right = parts.remove(i + 1);
                            parts[i].push_str(&right)
                        } else {
                            break;
                        }
                    }
                    for part in parts {
                        out.push(*self.ids.get(&part).ok_or_else(|| {
                            Error::Tokenizer(format!("missing BPE token {part:?}"))
                        })?)
                    }
                }
                if end != text.len() {
                    return Err(Error::Tokenizer("pre-tokenizer incomplete".into()));
                }
            }
        }
        Ok(())
    }
}
