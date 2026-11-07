use super::*;
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use model_format::{Checkpoint, DType, Tensor, TensorBytes, WeightData};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::cell::Cell;
use std::{collections::HashMap, ffi::c_void, ptr::NonNull, rc::Rc, sync::Arc};
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}
type Metal<T> = Retained<ProtocolObject<T>>;
struct Allocations {
    live: Cell<usize>,
}
struct Storage {
    raw: Metal<dyn MTLBuffer>,
    _mapping: Option<Arc<Mmap>>,
    owner: Rc<Allocations>,
}
impl Storage {
    fn new(raw: Metal<dyn MTLBuffer>, mapping: Option<Arc<Mmap>>, owner: Rc<Allocations>) -> Self {
        owner.live.set(owner.live.get() + raw.length());
        Self {
            raw,
            _mapping: mapping,
            owner,
        }
    }
}
impl Drop for Storage {
    fn drop(&mut self) {
        self.owner
            .live
            .set(self.owner.live.get() - self.raw.length())
    }
}
#[derive(Clone)]
pub struct Buffer {
    storage: Rc<Storage>,
    offset: usize,
    len: usize,
}
impl Buffer {
    pub fn len(&self) -> usize {
        self.len / 4
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn view(&self, start: usize, len: usize) -> Result<Self> {
        let end = start
            .checked_add(len)
            .ok_or_else(|| shape("view overflow"))?;
        if end > self.len() || len == 0 {
            return Err(shape("view bounds"));
        }
        Ok(Self {
            storage: self.storage.clone(),
            offset: self.offset + start * 4,
            len: len * 4,
        })
    }
}
#[derive(Clone)]
struct AffineWeight {
    scales: Buffer,
    biases: Buffer,
    bits: u32,
    group_size: usize,
    scale_type: DType,
}
#[derive(Clone)]
pub struct Weight {
    buffer: Buffer,
    dtype: DType,
    dims: Vec<usize>,
    affine: Option<AffineWeight>,
}
impl Weight {
    pub fn dtype(&self) -> Option<DType> {
        if self.affine.is_some() {
            None
        } else {
            Some(self.dtype)
        }
    }
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    pub fn elements(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn reshape(&self, dims: &[usize]) -> Result<Self> {
        let n = dims
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| shape("weight reshape overflow"))?;
        if dims.is_empty() || dims.contains(&0) || n != self.elements() {
            return Err(shape("weight reshape"));
        }
        if self.affine.is_some() && dims[0] != self.dims[0] {
            return Err(shape("affine reshape must preserve row width"));
        }
        self.dtype
            .row_bytes(dims[0])
            .map_err(|e| shape(e.to_string()))?;
        Ok(Self {
            buffer: self.buffer.clone(),
            dtype: self.dtype,
            dims: dims.to_vec(),
            affine: self.affine.clone(),
        })
    }
}
fn shape(s: impl Into<String>) -> Error {
    Error::Shape(s.into())
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Params {
    n: u32,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    e: u32,
    f: u32,
    g: u32,
    h: u32,
    i: u32,
    j: u32,
    k: u32,
    l: u32,
    m: u32,
    o: u32,
    t: u32,
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}
fn u(v: usize) -> Result<u32> {
    u32::try_from(v).map_err(|_| shape("dimension exceeds u32"))
}
fn count(dims: &[usize]) -> Result<usize> {
    let n = dims
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| shape("element count overflow"))?;
    u(n)?;
    Ok(n)
}
pub struct Gpu {
    device: Metal<dyn MTLDevice>,
    queue: Metal<dyn MTLCommandQueue>,
    library: Metal<dyn MTLLibrary>,
    pipelines: HashMap<&'static str, Metal<dyn MTLComputePipelineState>>,
    owner: Rc<Allocations>,
    pub name: String,
    pub recommended_working_set: usize,
    pub max_buffer_length: usize,
    profile: bool,
}
impl Gpu {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| Error::Unavailable("no Metal device".into()))?;
        if !device.hasUnifiedMemory() || !device.supportsFamily(MTLGPUFamily::Apple7) {
            return Err(Error::Unavailable(
                "requires Apple Silicon (Apple7+)".into(),
            ));
        }
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| Error::Gpu("command queue".into()))?;
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(concat!(
                    include_str!("shaders/common.metal"),
                    "\n",
                    include_str!("shaders/matrix.metal"),
                    "\n",
                    include_str!("shaders/elementwise.metal"),
                    "\n",
                    include_str!("shaders/attention.metal"),
                    "\n",
                    include_str!("shaders/recurrent.metal"),
                    "\n",
                    include_str!("shaders/sampling.metal"),
                    "\n",
                    include_str!("shaders/affine.metal"),
                )),
                None,
            )
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let name = device.name().to_string();
        let recommended_working_set = device.recommendedMaxWorkingSetSize() as usize;
        let max_buffer_length = device.maxBufferLength();
        Ok(Self {
            device,
            queue,
            library,
            pipelines: HashMap::new(),
            owner: Rc::new(Allocations { live: Cell::new(0) }),
            name,
            recommended_working_set,
            max_buffer_length,
            profile: std::env::var_os("LLAMA_METAL_PROFILE").is_some(),
        })
    }
}

pub struct Batch<'a> {
    gpu: &'a mut Gpu,
    cmd: Metal<dyn MTLCommandBuffer>,
    enc: Metal<dyn MTLComputeCommandEncoder>,
    keep: Vec<Buffer>,
    ended: bool,
    pub dispatches: usize,
    profiled_seconds: f64,
}

mod commands;
mod memory;
mod ops;
mod weights;
