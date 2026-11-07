//! Synchronous submission boundaries with batched, ordered GPU operations.
//! CPU access requires an exclusive backend borrow; buffers cannot outlive mapped storage.
use thiserror::Error;
#[derive(Debug, Error)]
pub enum Error {
    #[error("Metal unavailable: {0}")]
    Unavailable(String),
    #[error("GPU error: {0}")]
    Gpu(String),
    #[error("invalid GPU operation: {0}")]
    Shape(String),
    #[error("allocation of {0} bytes exceeds the device limit or failed")]
    Allocation(usize),
}
pub type Result<T> = std::result::Result<T, Error>;
#[cfg(target_os = "macos")]
mod apple;
#[cfg(target_os = "macos")]
pub use apple::*;
#[cfg(not(target_os = "macos"))]
compile_error!("metal-backend requires macOS on Apple Silicon");
