use thiserror::Error;
#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid GGUF: {0}")]
    Invalid(String),
    #[error("unsupported GGUF feature: {0}")]
    Unsupported(String),
}
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
