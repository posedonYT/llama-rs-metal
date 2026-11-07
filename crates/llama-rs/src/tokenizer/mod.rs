use crate::{Error, Result};
use fancy_regex::Regex;
use model_format::{Gguf, Value};
use std::collections::HashMap;
#[derive(Debug)]
enum Kind {
    SentencePiece,
    HuggingFace {
        tokenizer: Box<tokenizers::Tokenizer>,
        byte_level: bool,
        byte_decoder: HashMap<char, u8>,
    },
    ByteBpe {
        pattern: Box<Regex>,
        ranks: HashMap<(String, String), usize>,
        encode: Vec<char>,
        decode: HashMap<char, u8>,
    },
}
pub struct Tokenizer {
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    scores: Vec<f32>,
    types: Vec<u64>,
    kind: Kind,
    pub bos: Option<u32>,
    pub eos: Vec<u32>,
    pub add_bos: bool,
    prefix: bool,
    special: Vec<(String, u32)>,
}
impl Tokenizer {
    pub fn from_checkpoint(checkpoint: &model_format::Checkpoint) -> Result<Self> {
        if let Some(file) = &checkpoint.gguf {
            return Self::from_gguf(file);
        }
        Self::from_huggingface(
            checkpoint
                .directory
                .as_deref()
                .ok_or_else(|| Error::Tokenizer("missing tokenizer directory".into()))?,
        )
    }
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }
}

mod decode;
mod encode;
mod gguf;
mod huggingface;
pub use decode::TextDecoder;
