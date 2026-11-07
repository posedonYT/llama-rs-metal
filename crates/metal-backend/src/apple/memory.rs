use super::*;
impl Gpu {
    pub(super) fn own(&self, b: &Buffer) -> Result<()> {
        if !Rc::ptr_eq(&self.owner, &b.storage.owner) {
            return Err(shape("buffer belongs to a different engine"));
        }
        Ok(())
    }
    pub(super) fn raw_buffer(&self, len: usize) -> Result<Buffer> {
        if len == 0 || len > self.max_buffer_length || len / 4 > u32::MAX as usize {
            return Err(Error::Allocation(len));
        }
        let raw = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Allocation(len))?;
        // SAFETY: fresh, CPU-accessible shared allocation; no command buffer can reference it yet.
        unsafe { std::ptr::write_bytes(raw.contents().as_ptr().cast::<u8>(), 0, len) };
        Ok(Buffer {
            storage: Rc::new(Storage::new(raw, None, self.owner.clone())),
            offset: 0,
            len,
        })
    }
    pub fn zeros(&mut self, n: usize) -> Result<Buffer> {
        self.raw_buffer(n.checked_mul(4).ok_or(Error::Allocation(usize::MAX))?)
    }
    pub fn upload(&mut self, data: &[f32]) -> Result<Buffer> {
        let b = self.raw_buffer(
            data.len()
                .checked_mul(4)
                .ok_or(Error::Allocation(usize::MAX))?,
        )?;
        // SAFETY: allocation is fresh, exact length, and not submitted to a GPU.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                b.storage.raw.contents().as_ptr().cast(),
                data.len(),
            )
        };
        Ok(b)
    }
    pub fn read(&mut self, b: &Buffer) -> Result<Vec<f32>> {
        self.own(b)?;
        // SAFETY: all submissions wait before releasing the exclusive Gpu borrow.
        Ok(unsafe {
            std::slice::from_raw_parts(
                b.storage
                    .raw
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(b.offset)
                    .cast::<f32>(),
                b.len(),
            )
        }
        .to_vec())
    }
    pub fn clear(&mut self, b: &Buffer) -> Result<()> {
        self.own(b)?;
        // SAFETY: no GPU submission is in flight while Gpu is exclusively borrowed here.
        unsafe {
            std::ptr::write_bytes(
                b.storage.raw.contents().as_ptr().cast::<u8>().add(b.offset),
                0,
                b.len,
            )
        };
        Ok(())
    }
    pub fn allocated_bytes(&self) -> usize {
        self.owner.live.get()
    }
}
