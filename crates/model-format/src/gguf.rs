use crate::dequantize;
use crate::error::invalid;
use crate::{DType, Error, Result, Value};
use memmap2::Mmap;
use serde::Serialize;
use std::{collections::BTreeMap, fs::File, path::Path, sync::Arc};
#[derive(Clone, Debug, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: DType,
    pub offset: usize,
    pub bytes: usize,
}
impl TensorInfo {
    pub fn elements(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn row_bytes(&self) -> usize {
        self.dtype
            .row_bytes(self.dims[0])
            .expect("validated tensor")
    }
}
pub struct Gguf {
    pub metadata: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub data_offset: usize,
    map: Arc<Mmap>,
}
impl Gguf {
    /// The caller must not modify or truncate the model file while any model/session is alive.
    /// A read-only mapping avoids copying multi-gigabyte weights into a second allocation.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the mapping is read-only; mutation of mapped model files is forbidden by API contract.
        let map = Arc::new(unsafe { Mmap::map(&file)? });
        let mut r = Reader { data: &map, pos: 0 };
        if r.take(4)? != b"GGUF" {
            return Err(invalid("magic"));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Unsupported(format!("GGUF version {version}")));
        }
        let nt = r.count(24)?;
        let nm = r.count(12)?;
        let mut metadata = BTreeMap::new();
        for _ in 0..nm {
            let key = r.string()?;
            let ty = r.u32()?;
            let value = r.value(ty, 0)?;
            if metadata.insert(key, value).is_some() {
                return Err(invalid("duplicate metadata key"));
            }
        }
        if metadata
            .get("split.count")
            .and_then(Value::uint)
            .unwrap_or(1)
            > 1
        {
            return Err(Error::Unsupported(
                "split GGUF: merge shards before loading".into(),
            ));
        }
        let align = metadata
            .get("general.alignment")
            .and_then(Value::uint)
            .unwrap_or(32);
        let align = usize::try_from(align).map_err(|_| invalid("alignment overflow"))?;
        if !align.is_power_of_two() || align > 1024 * 1024 {
            return Err(invalid("alignment"));
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..nt {
            let name = r.string()?;
            let nd = r.u32()? as usize;
            if !(1..=4).contains(&nd) {
                return Err(invalid("tensor rank"));
            }
            let mut dims = Vec::new();
            let mut elements = 1usize;
            for _ in 0..nd {
                let d = usize::try_from(r.u64()?).map_err(|_| invalid("dimension overflow"))?;
                if d == 0 {
                    return Err(invalid("zero dimension"));
                }
                elements = elements
                    .checked_mul(d)
                    .ok_or_else(|| invalid("element count overflow"))?;
                dims.push(d);
            }
            let dtype = DType::from_id(r.u32()?)?;
            let offset = usize::try_from(r.u64()?).map_err(|_| invalid("offset overflow"))?;
            if offset % align != 0 {
                return Err(invalid(format!("unaligned tensor {name}")));
            }
            let bytes = dtype
                .row_bytes(dims[0])?
                .checked_mul(elements / dims[0])
                .ok_or_else(|| invalid("tensor size overflow"))?;
            if tensors
                .insert(
                    name.clone(),
                    TensorInfo {
                        name,
                        dims,
                        dtype,
                        offset,
                        bytes,
                    },
                )
                .is_some()
            {
                return Err(invalid("duplicate tensor"));
            }
        }
        let data_offset = r
            .pos
            .checked_add(align - 1)
            .ok_or_else(|| invalid("alignment overflow"))?
            & !(align - 1);
        let mut ranges = Vec::new();
        for t in tensors.values_mut() {
            t.offset = t
                .offset
                .checked_add(data_offset)
                .ok_or_else(|| invalid("offset overflow"))?;
            let end = t
                .offset
                .checked_add(t.bytes)
                .ok_or_else(|| invalid("size overflow"))?;
            if end > map.len() {
                return Err(invalid(format!("tensor {} outside file", t.name)));
            }
            ranges.push((t.offset, end));
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err(invalid("overlapping tensors"));
        }
        Ok(Self {
            metadata,
            tensors,
            data_offset,
            map,
        })
    }
    pub fn bytes(&self) -> &[u8] {
        &self.map
    }
    pub fn mapping(&self) -> Arc<Mmap> {
        self.map.clone()
    }
    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))
    }
    pub fn data(&self, t: &TensorInfo) -> Result<&[u8]> {
        self.map
            .get(
                t.offset
                    ..t.offset
                        .checked_add(t.bytes)
                        .ok_or_else(|| invalid("offset overflow"))?,
            )
            .ok_or_else(|| invalid("tensor range"))
    }
    pub fn decode(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        dequantize(t.dtype, self.data(t)?, t.elements())
    }
    pub fn str(&self, k: &str) -> Option<&str> {
        self.metadata.get(k).and_then(Value::string)
    }
    pub fn uint(&self, k: &str) -> Option<usize> {
        self.metadata
            .get(k)
            .and_then(Value::uint)
            .and_then(|v| usize::try_from(v).ok())
    }
    pub fn float(&self, k: &str) -> Option<f32> {
        self.metadata
            .get(k)
            .and_then(Value::float)
            .map(|v| v as f32)
    }
}
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}
impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| invalid("length overflow"))?;
        let b = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| invalid("truncated file"))?;
        self.pos = end;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn count(&mut self, min: usize) -> Result<usize> {
        let n = usize::try_from(self.u64()?).map_err(|_| invalid("count overflow"))?;
        if n > (self.data.len() - self.pos) / min.max(1) {
            return Err(invalid("impossible count"));
        }
        Ok(n)
    }
    fn string(&mut self) -> Result<String> {
        let n = self.count(1)?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| invalid("non-UTF8 metadata"))
    }
    fn value(&mut self, ty: u32, depth: u8) -> Result<Value> {
        if depth > 8 {
            return Err(invalid("metadata nesting"));
        }
        Ok(match ty {
            0 => Value::Uint(self.take(1)?[0] as u64),
            1 => Value::Int(self.take(1)?[0] as i8 as i64),
            2 => Value::Uint(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Value::Int(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::Uint(self.u32()? as u64),
            5 => Value::Int(self.u32()? as i32 as i64),
            6 => Value::Float(f32::from_bits(self.u32()?) as f64),
            7 => {
                let b = self.take(1)?[0];
                if b > 1 {
                    return Err(invalid("boolean"));
                }
                Value::Bool(b != 0)
            }
            8 => Value::String(self.string()?),
            9 => {
                let elem = self.u32()?;
                if elem == 9 {
                    return Err(invalid("nested arrays"));
                }
                let n = self.count(1)?;
                let mut a = Vec::new();
                for _ in 0..n {
                    a.push(self.value(elem, depth + 1)?)
                }
                Value::Array(a)
            }
            10 => Value::Uint(self.u64()?),
            11 => Value::Int(self.u64()? as i64),
            12 => Value::Float(f64::from_bits(self.u64()?)),
            _ => return Err(invalid(format!("metadata type {ty}"))),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequantize;
    use half::f16;
    #[test]
    fn quant_block_layouts() {
        let mut b = vec![0; 18];
        b[..2].copy_from_slice(&f16::from_f32(2.).to_le_bytes());
        b[2..].fill(0xf0);
        let v = dequantize(DType::Q4_0, &b, 32).unwrap();
        assert_eq!(&v[..16], &[-16.; 16]);
        assert_eq!(&v[16..], &[14.; 16]);
        assert!(dequantize(DType::Q4K, &b, 32).is_err());
    }
    #[test]
    fn truncated_metadata() {
        let mut r = Reader {
            data: &[255; 8],
            pos: 0,
        };
        assert!(r.string().is_err());
        assert!(DType::Q6K.row_bytes(257).is_err());
    }
}
