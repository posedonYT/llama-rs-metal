//! The runtime-facing checkpoint contract. File adapters own naming/layout conversion.
use crate::{DType, Error, Gguf, ModelConfig, Result, dequantize};
use memmap2::Mmap;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointFormat {
    Gguf,
    Mlx,
}

#[derive(Clone, Debug)]
enum ByteStorage {
    Mapped {
        map: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
    Owned(Arc<[u8]>),
}
/// Immutable bytes. A GPU view retains the mapping rather than a borrowed pointer.
#[derive(Clone, Debug)]
pub struct TensorBytes(ByteStorage);
impl TensorBytes {
    pub fn owned(bytes: Vec<u8>) -> Self {
        Self(ByteStorage::Owned(bytes.into()))
    }
    pub(crate) fn mapped(map: Arc<Mmap>, offset: usize, len: usize) -> Result<Self> {
        if offset.checked_add(len).is_none_or(|end| end > map.len()) {
            return Err(Error::Invalid("tensor mapping outside file".into()));
        }
        Ok(Self(ByteStorage::Mapped { map, offset, len }))
    }
    pub fn as_slice(&self) -> &[u8] {
        match &self.0 {
            ByteStorage::Mapped { map, offset, len } => &map[*offset..*offset + *len],
            ByteStorage::Owned(b) => b,
        }
    }
    pub fn len(&self) -> usize {
        match &self.0 {
            ByteStorage::Mapped { len, .. } => *len,
            ByteStorage::Owned(b) => b.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn mapping(&self) -> Option<(&Arc<Mmap>, usize)> {
        match &self.0 {
            ByteStorage::Mapped { map, offset, .. } => Some((map, *offset)),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AffineQuantization {
    pub bits: u32,
    pub group_size: usize,
    pub scale_type: DType,
    pub scales: TensorBytes,
    pub biases: TensorBytes,
}
#[derive(Clone, Debug)]
pub enum WeightData {
    Ggml {
        bytes: TensorBytes,
        dtype: DType,
    },
    Affine {
        packed: TensorBytes,
        quantization: AffineQuantization,
    },
}
#[derive(Clone, Debug)]
pub struct Tensor {
    dims: Vec<usize>,
    data: WeightData,
}
impl Tensor {
    pub fn new(dims: Vec<usize>, data: WeightData) -> Result<Self> {
        let n = dims
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Error::Invalid("tensor size overflow".into()))?;
        if dims.is_empty() || dims.len() > 4 || dims.contains(&0) {
            return Err(Error::Invalid("tensor dimensions".into()));
        }
        match &data {
            WeightData::Ggml { bytes, dtype } => {
                if dtype.row_bytes(dims[0])?.checked_mul(n / dims[0]) != Some(bytes.len()) {
                    return Err(Error::Invalid("tensor byte length".into()));
                }
            }
            WeightData::Affine {
                packed,
                quantization: q,
            } => {
                if ![2, 4, 8].contains(&q.bits) || ![32, 64, 128].contains(&q.group_size) {
                    return Err(Error::Unsupported(format!(
                        "MLX affine bits={}, group_size={}",
                        q.bits, q.group_size
                    )));
                }
                if ![DType::F16, DType::BF16, DType::F32].contains(&q.scale_type)
                    || dims[0] % q.group_size != 0
                {
                    return Err(Error::Invalid("affine scale type/group alignment".into()));
                }
                let expected = n
                    .checked_mul(q.bits as usize)
                    .and_then(|n| n.checked_div(8));
                let scales = (n / q.group_size).checked_mul(q.scale_type.block().1);
                if expected != Some(packed.len())
                    || scales != Some(q.scales.len())
                    || q.scales.len() != q.biases.len()
                {
                    return Err(Error::Invalid("affine packed/scale/bias length".into()));
                }
            }
        }
        Ok(Self { dims, data })
    }
    pub fn from_f32(dims: Vec<usize>, values: &[f32]) -> Result<Self> {
        Self::new(
            dims,
            WeightData::Ggml {
                bytes: TensorBytes::owned(values.iter().flat_map(|v| v.to_le_bytes()).collect()),
                dtype: DType::F32,
            },
        )
    }
    pub fn reshape(&self, dims: Vec<usize>) -> Result<Self> {
        if dims.iter().try_fold(1usize, |n, d| n.checked_mul(*d)) != Some(self.elements()) {
            return Err(Error::Invalid("reshape element count".into()));
        }
        Self::new(dims, self.data.clone())
    }
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    pub fn data(&self) -> &WeightData {
        &self.data
    }
    pub fn elements(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn storage_bytes(&self) -> usize {
        match &self.data {
            WeightData::Ggml { bytes, .. } => bytes.len(),
            WeightData::Affine {
                packed,
                quantization: q,
            } => packed.len() + q.scales.len() + q.biases.len(),
        }
    }
    pub fn decode(&self) -> Result<Vec<f32>> {
        match &self.data {
            WeightData::Ggml { bytes, dtype } => {
                dequantize(*dtype, bytes.as_slice(), self.elements())
            }
            WeightData::Affine {
                packed,
                quantization: q,
            } => {
                let scales = dequantize(
                    q.scale_type,
                    q.scales.as_slice(),
                    self.elements() / q.group_size,
                )?;
                let biases = dequantize(
                    q.scale_type,
                    q.biases.as_slice(),
                    self.elements() / q.group_size,
                )?;
                let per_word = 32 / q.bits;
                let mask = (1u32 << q.bits) - 1;
                let bytes = packed.as_slice();
                Ok((0..self.elements())
                    .map(|i| {
                        let word = i / (per_word as usize) * 4;
                        let value = u32::from_le_bytes(bytes[word..word + 4].try_into().unwrap());
                        let code = (value >> ((i % per_word as usize) as u32 * q.bits)) & mask;
                        code as f32 * scales[i / q.group_size] + biases[i / q.group_size]
                    })
                    .collect())
            }
        }
    }
    pub(crate) fn float_storage(&self) -> Result<Self> {
        match &self.data {
            WeightData::Ggml {
                dtype: DType::F32, ..
            } => Ok(self.clone()),
            WeightData::Ggml { .. } => Self::from_f32(self.dims.clone(), &self.decode()?),
            WeightData::Affine {
                packed,
                quantization: q,
            } => {
                let n = self.elements() / q.group_size;
                let bytes = |b: &TensorBytes| -> Result<TensorBytes> {
                    Ok(TensorBytes::owned(
                        dequantize(q.scale_type, b.as_slice(), n)?
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    ))
                };
                Self::new(
                    self.dims.clone(),
                    WeightData::Affine {
                        packed: packed.clone(),
                        quantization: AffineQuantization {
                            bits: q.bits,
                            group_size: q.group_size,
                            scale_type: DType::F32,
                            scales: bytes(&q.scales)?,
                            biases: bytes(&q.biases)?,
                        },
                    },
                )
            }
        }
    }
}

/// Only validated semantic tensors reach the inference engine. Architecture
/// conventions live in ModelConfig, not in backend checks for a file format.
pub struct Checkpoint {
    pub config: ModelConfig,
    pub format: CheckpointFormat,
    tensors: BTreeMap<String, Tensor>,
    pub directory: Option<PathBuf>,
    pub gguf: Option<Gguf>,
    pub chat_template: Option<String>,
    pub rope_factors: Vec<f32>,
    pub provenance: serde_json::Value,
}
impl Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_float_storage(path, false)
    }
    /// Matched-precision benchmarks can cast float tensors and affine metadata to
    /// F32 without changing integer codes or requantizing model weights.
    pub fn open_with_float_storage(path: impl AsRef<Path>, float32: bool) -> Result<Self> {
        let path = path.as_ref();
        let mut checkpoint = if path.is_dir() {
            crate::mlx::open(path)?
        } else {
            Self::from_gguf(Gguf::open(path)?)?
        };
        if float32 && checkpoint.format == CheckpointFormat::Mlx {
            for tensor in checkpoint.tensors.values_mut() {
                *tensor = tensor.float_storage()?
            }
        }
        Ok(checkpoint)
    }
    pub fn from_gguf(file: Gguf) -> Result<Self> {
        let config = ModelConfig::from_gguf(&file)?;
        let mut tensors = BTreeMap::new();
        let map = file.mapping();
        for (name, t) in &file.tensors {
            tensors.insert(
                name.clone(),
                Tensor::new(
                    t.dims.clone(),
                    WeightData::Ggml {
                        bytes: TensorBytes::mapped(map.clone(), t.offset, t.bytes)?,
                        dtype: t.dtype,
                    },
                )?,
            );
        }
        let rope_factors = if tensors.contains_key("rope_freqs.weight") {
            tensors["rope_freqs.weight"].decode()?
        } else if tensors.contains_key("blk.0.rope_freqs.weight") {
            tensors["blk.0.rope_freqs.weight"].decode()?
        } else if file.str(&format!("{}.rope.scaling.type", config.architecture)) == Some("llama3")
        {
            return Err(Error::Unsupported(
                "scaled Llama3 GGUF requires rope_freqs.weight".into(),
            ));
        } else {
            vec![1.; config.rotary / 2]
        };
        let chat_template = file.str("tokenizer.chat_template").map(str::to_string);
        Ok(Self {
            config,
            format: CheckpointFormat::Gguf,
            tensors,
            directory: None,
            gguf: Some(file),
            chat_template,
            rope_factors,
            provenance: serde_json::json!({"format":"gguf"}),
        })
    }
    pub fn tensors(&self) -> &BTreeMap<String, Tensor> {
        &self.tensors
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("missing tensor {name}")))
    }
    pub fn decode(&self, name: &str) -> Result<Vec<f32>> {
        self.tensor(name)?.decode()
    }
    pub fn weight_bytes(&self) -> Result<usize> {
        self.tensors.values().try_fold(0usize, |n, t| {
            n.checked_add(t.storage_bytes())
                .ok_or_else(|| Error::Invalid("checkpoint size overflow".into()))
        })
    }
    pub(crate) fn mlx(
        config: ModelConfig,
        tensors: BTreeMap<String, Tensor>,
        directory: PathBuf,
        chat_template: Option<String>,
        rope_factors: Vec<f32>,
        provenance: serde_json::Value,
    ) -> Self {
        Self {
            config,
            tensors,
            directory: Some(directory),
            gguf: None,
            format: CheckpointFormat::Mlx,
            chat_template,
            rope_factors,
            provenance,
        }
    }
}
