use super::*;
impl Gpu {
    pub fn warmup(&mut self) -> Result<()> {
        for n in [
            "argmax",
            "affine_embedding",
            "affine_mv_2",
            "affine_mv_4",
            "affine_mv_8",
            "affine_mm",
            "affine_expert_2",
            "affine_expert_4",
            "affine_expert_8",
            "embedding",
            "gemv_f32",
            "gemv_f16",
            "gemv_q4",
            "gemv_q8",
            "packed_q4k",
            "packed_q5k",
            "packed_q6k",
            "gemv_bf16",
            "add",
            "activation",
            "gate",
            "copy_f32",
            "norm",
            "rope",
            "attention",
            "attention_partial",
            "attention_merge",
            "conv",
            "delta",
            "route",
            "expert_mv",
            "mix",
            "gather",
            "gate_scalar",
            "gemm",
            "gemm_simd",
            "gemm_simd_wide",
            "embedding_many",
        ] {
            self.pipeline(n)?;
        }
        Ok(())
    }
    pub(super) fn pipeline(
        &mut self,
        name: &'static str,
    ) -> Result<Metal<dyn MTLComputePipelineState>> {
        if let Some(p) = self.pipelines.get(name) {
            return Ok(p.clone());
        }
        let f = self
            .library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| Error::Gpu(format!("missing kernel {name}")))?;
        let p = self
            .device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| Error::Gpu(e.to_string()))?;
        self.pipelines.insert(name, p.clone());
        Ok(p)
    }
    /// Encode and complete a submission within an Objective-C autorelease pool.
    pub fn with_batch<T, E>(
        &mut self,
        f: impl FnOnce(&mut Batch<'_>) -> std::result::Result<T, E>,
    ) -> std::result::Result<(T, f64), E>
    where
        E: From<Error>,
    {
        objc2::rc::autoreleasepool(|_| {
            let mut b = self.batch()?;
            let value = f(&mut b)?;
            let seconds = b.finish()?;
            Ok((value, seconds))
        })
    }
    pub fn batch(&mut self) -> Result<Batch<'_>> {
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or_else(|| Error::Gpu("command buffer".into()))?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or_else(|| Error::Gpu("encoder".into()))?;
        Ok(Batch {
            gpu: self,
            cmd,
            enc,
            keep: Vec::new(),
            ended: false,
            dispatches: 0,
            profiled_seconds: 0.,
        })
    }
}
impl Batch<'_> {
    pub(super) fn dispatch(
        &mut self,
        name: &'static str,
        buffers: &[&Buffer],
        p: Params,
        groups: (usize, usize),
        simds: usize,
    ) -> Result<()> {
        for b in buffers {
            self.gpu.own(b)?
        }
        let (reads, writes, inplace): (&[usize], &[usize], bool) = match name {
            "argmax" | "embedding" => (&[0], &[1], false),
            "embedding_many" => (&[0, 1], &[2], false),
            "affine_embedding" | "affine_mv_2" | "affine_mv_4" | "affine_mv_8" | "affine_mm" => {
                (&[0, 1, 2, 3], &[4], false)
            }
            "affine_expert_2" | "affine_expert_4" | "affine_expert_8" => {
                (&[0, 1, 2, 3, 4], &[5], false)
            }
            "conv" => (&[0, 1], &[2, 3], false),
            "delta" => (&[0, 1, 2, 3, 4], &[5, 6], false),
            "attention" | "attention_partial" => (&[0, 1, 2], &[3], false),
            "attention_merge" => (&[0], &[1], false),
            "expert_mv" => (&[0, 1, 2], &[3], false),
            "route" => (&[0], &[1, 2], false),
            "gather" | "copy_f32" => (&[0], &[1], false),
            "rope" => (&[0, 2], &[1], false),
            "add" | "activation" | "gate" | "gate_scalar" => (&[0, 1], &[2], true),
            _ => (&[0, 1], &[2], false),
        };
        let overlaps = |a: &Buffer, b: &Buffer| {
            Rc::ptr_eq(&a.storage, &b.storage)
                && a.offset < b.offset + b.len
                && b.offset < a.offset + a.len
        };
        for (i, &wi) in writes.iter().enumerate() {
            for &wj in &writes[i + 1..] {
                if overlaps(buffers[wi], buffers[wj]) {
                    return Err(shape("overlapping GPU outputs"));
                }
            }
            for &ri in reads {
                let a = buffers[wi];
                let b = buffers[ri];
                if overlaps(a, b) && !(inplace && a.offset == b.offset && a.len == b.len) {
                    return Err(shape("GPU input/output alias"));
                }
            }
        }
        let pipeline = self.gpu.pipeline(name)?;
        let width = pipeline.threadExecutionWidth();
        let threads = width * simds;
        if (name == "gemm" && threads != 256) || threads > pipeline.maxTotalThreadsPerThreadgroup()
        {
            return Err(shape("threadgroup limit"));
        }
        self.enc.setComputePipelineState(&pipeline);
        // SAFETY: high-level operations below check all shapes; offsets come from checked views.
        // Ordered dispatch plus a buffer barrier establishes every read-after-write dependency.
        unsafe {
            for (i, b) in buffers.iter().enumerate() {
                self.enc
                    .setBuffer_offset_atIndex(Some(&b.storage.raw), b.offset, i);
                self.keep.push((*b).clone());
            }
            self.enc.setBytes_length_atIndex(
                NonNull::from(&p).cast(),
                std::mem::size_of::<Params>(),
                8,
            );
        }
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups.0,
                height: groups.1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        self.enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        self.dispatches += 1;
        if self.gpu.profile {
            self.ended = true;
            self.enc.endEncoding();
            self.cmd.commit();
            self.cmd.waitUntilCompleted();
            if let Some(e) = self.cmd.error() {
                self.ended = true;
                return Err(Error::Gpu(e.to_string()));
            }
            let seconds = self.cmd.GPUEndTime() - self.cmd.GPUStartTime();
            self.profiled_seconds += seconds;
            eprintln!("METAL_PROFILE {name} {seconds:.9}");
            self.cmd = self
                .gpu
                .queue
                .commandBuffer()
                .ok_or_else(|| Error::Gpu("profile command buffer".into()))?;
            self.enc = self
                .cmd
                .computeCommandEncoder()
                .ok_or_else(|| Error::Gpu("profile encoder".into()))?;
            self.ended = false;
        }
        Ok(())
    }
    pub(super) fn linear(&mut self, name: &'static str, b: &[&Buffer], p: Params) -> Result<()> {
        self.dispatch(name, b, p, ((p.n as usize).div_ceil(32), 1), 1)
    }
    pub fn finish(mut self) -> Result<f64> {
        self.enc.endEncoding();
        self.ended = true;
        self.cmd.commit();
        self.cmd.waitUntilCompleted();
        if let Some(e) = self.cmd.error() {
            return Err(Error::Gpu(e.to_string()));
        }
        Ok(self.profiled_seconds + self.cmd.GPUEndTime() - self.cmd.GPUStartTime())
    }
}
impl Drop for Batch<'_> {
    fn drop(&mut self) {
        if !self.ended {
            self.enc.endEncoding();
        }
    }
}
