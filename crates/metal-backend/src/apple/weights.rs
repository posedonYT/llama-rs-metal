use super::*;
impl Gpu {
    pub fn weight_f32(&mut self, data: &[f32], dims: &[usize]) -> Result<Weight> {
        let mut n = 1usize;
        for &d in dims {
            n = n
                .checked_mul(d)
                .ok_or_else(|| shape("weight dimension overflow"))?;
        }
        if n != data.len() || dims.is_empty() || dims.contains(&0) {
            return Err(shape("weight dimensions"));
        }
        Ok(Weight {
            buffer: self.upload(data)?,
            dtype: DType::F32,
            dims: dims.to_vec(),
            affine: None,
        })
    }
    /// Upload a validated packed tensor, primarily for custom operators and reference tests.
    pub fn weight_bytes(&mut self, data: &[u8], dtype: DType, dims: &[usize]) -> Result<Weight> {
        let n = dims
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| shape("weight size overflow"))?;
        if dims.is_empty()
            || dims.contains(&0)
            || dtype
                .row_bytes(dims[0])
                .map_err(|e| shape(e.to_string()))?
                .checked_mul(n / dims[0])
                != Some(data.len())
            || data.len() > self.max_buffer_length
        {
            return Err(shape("packed weight length"));
        }
        // SAFETY: Metal copies the validated slice synchronously, retaining its own allocation.
        let raw = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
                data.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(Error::Allocation(data.len()))?;
        Ok(Weight {
            buffer: Buffer {
                storage: Rc::new(Storage::new(raw, None, self.owner.clone())),
                offset: 0,
                len: data.len(),
            },
            dtype,
            dims: dims.to_vec(),
            affine: None,
        })
    }
    /// Conservative retained Metal bytes, including whole no-copy shard pages
    /// (which can contain unused vision tensors) and separately copied tails.
    /// The copy fallback cannot exceed the corresponding validated shard span.
    pub fn weight_allocation_bound(&self, checkpoint: &Checkpoint) -> Result<usize> {
        let mut mappings = HashMap::new();
        let mut copied = 0usize;
        let mut account = |bytes: &TensorBytes| -> Result<()> {
            if let Some((map, offset)) = bytes.mapping() {
                let length = map.len() / 16384 * 16384;
                let key = map.as_ptr() as usize;
                if length > 0
                    && length <= self.max_buffer_length
                    && key % 16384 == 0
                    && offset % 4 == 0
                    && offset + bytes.len() <= length
                {
                    mappings.insert(key, length);
                    return Ok(());
                }
            }
            copied = copied
                .checked_add(bytes.len())
                .ok_or(Error::Allocation(usize::MAX))?;
            Ok(())
        };
        for tensor in checkpoint.tensors().values() {
            match tensor.data() {
                WeightData::Ggml { bytes, .. } => account(bytes)?,
                WeightData::Affine {
                    packed,
                    quantization: q,
                } => {
                    account(packed)?;
                    account(&q.scales)?;
                    account(&q.biases)?;
                }
            }
        }
        mappings.values().try_fold(copied, |total, bytes| {
            total
                .checked_add(*bytes)
                .ok_or(Error::Allocation(usize::MAX))
        })
    }
    pub fn weights(&mut self, checkpoint: &Checkpoint) -> Result<HashMap<String, Weight>> {
        let mut mappings = HashMap::new();
        let mut weights = HashMap::new();
        for (name, tensor) in checkpoint.tensors() {
            weights.insert(name.clone(), self.tensor_weight(tensor, &mut mappings)?);
        }
        Ok(weights)
    }
    pub fn weight_tensor(&mut self, tensor: &Tensor) -> Result<Weight> {
        self.tensor_weight(tensor, &mut HashMap::new())
    }
    pub(super) fn tensor_weight(
        &self,
        tensor: &Tensor,
        mappings: &mut HashMap<usize, Rc<Storage>>,
    ) -> Result<Weight> {
        let (buffer, dtype, affine) = match tensor.data() {
            WeightData::Ggml { bytes, dtype } => {
                (self.upload_region(bytes, mappings)?, *dtype, None)
            }
            WeightData::Affine {
                packed,
                quantization: q,
            } => {
                let a = AffineWeight {
                    scales: self.upload_region(&q.scales, mappings)?,
                    biases: self.upload_region(&q.biases, mappings)?,
                    bits: q.bits,
                    group_size: q.group_size,
                    scale_type: q.scale_type,
                };
                (self.upload_region(packed, mappings)?, DType::F32, Some(a))
            }
        };
        Ok(Weight {
            buffer,
            dtype,
            affine,
            dims: tensor.dims().to_vec(),
        })
    }
    pub(super) fn upload_region(
        &self,
        bytes: &TensorBytes,
        mappings: &mut HashMap<usize, Rc<Storage>>,
    ) -> Result<Buffer> {
        if let Some((map, offset)) = bytes.mapping() {
            let page = 16384;
            let length = map.len() / page * page;
            let key = map.as_ptr() as usize;
            if length > 0
                && length <= self.max_buffer_length
                && key % page == 0
                && offset % 4 == 0
                && offset + bytes.len() <= length
            {
                if let std::collections::hash_map::Entry::Vacant(entry) = mappings.entry(key) {
                    // SAFETY: complete page-aligned mapped pages, retained by Storage. Kernels read only.
                    let raw = unsafe {
                        self.device
                            .newBufferWithBytesNoCopy_length_options_deallocator(
                                NonNull::new(map.as_ptr() as *mut c_void).unwrap(),
                                length,
                                MTLResourceOptions::StorageModeShared,
                                None,
                            )
                    };
                    if let Some(raw) = raw {
                        entry.insert(Rc::new(Storage::new(
                            raw,
                            Some(map.clone()),
                            self.owner.clone(),
                        )));
                    }
                }
                if let Some(storage) = mappings.get(&key) {
                    return Ok(Buffer {
                        storage: storage.clone(),
                        offset,
                        len: bytes.len(),
                    });
                }
            }
        }
        let data = bytes.as_slice();
        if data.is_empty() || data.len() > self.max_buffer_length {
            return Err(Error::Allocation(data.len()));
        }
        // SAFETY: Metal copies the validated byte slice before this call returns.
        let raw = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
                data.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(Error::Allocation(data.len()))?;
        Ok(Buffer {
            storage: Rc::new(Storage::new(raw, None, self.owner.clone())),
            offset: 0,
            len: data.len(),
        })
    }
}
