use crate::error::invalid;
use crate::{Error, Result};
use serde::Serialize;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[repr(u32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    BF16 = 30,
}
impl DType {
    pub fn from_id(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            30 => Self::BF16,
            _ => return Err(Error::Unsupported(format!("tensor type {v}"))),
        })
    }
    pub fn block(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q8_0 => (32, 34),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
        }
    }
    pub fn row_bytes(self, n: usize) -> Result<usize> {
        let (block, bytes) = self.block();
        if n == 0 || n % block != 0 {
            return Err(invalid(format!("row width {n} incompatible with {self:?}")));
        }
        (n / block)
            .checked_mul(bytes)
            .ok_or_else(|| invalid("row size overflow"))
    }
}
