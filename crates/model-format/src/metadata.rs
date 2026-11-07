use serde::Serialize;
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum Value {
    Uint(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}
impl Value {
    pub fn uint(&self) -> Option<u64> {
        match self {
            Self::Uint(v) => Some(*v),
            Self::Int(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            _ => self.uint().map(|v| v as f64),
        }
    }
    pub fn string(&self) -> Option<&str> {
        if let Self::String(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn array(&self) -> Option<&[Value]> {
        if let Self::Array(a) = self {
            Some(a)
        } else {
            None
        }
    }
}
