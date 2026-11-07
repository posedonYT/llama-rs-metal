use super::*;
impl Batch<'_> {
    pub(super) fn binary(
        &mut self,
        name: &'static str,
        a: &Buffer,
        b: &Buffer,
        y: &Buffer,
    ) -> Result<()> {
        if a.len() != b.len() || a.len() != y.len() {
            return Err(shape(name));
        }
        self.linear(
            name,
            &[a, b, y],
            Params {
                n: u(a.len())?,
                ..Params::default()
            },
        )
    }
    pub fn add(&mut self, a: &Buffer, b: &Buffer, y: &Buffer) -> Result<()> {
        self.binary("add", a, b, y)
    }
    pub fn swiglu(&mut self, a: &Buffer, b: &Buffer, y: &Buffer) -> Result<()> {
        self.binary("activation", a, b, y)
    }
    pub fn gate(&mut self, a: &Buffer, b: &Buffer, y: &Buffer) -> Result<()> {
        self.binary("gate", a, b, y)
    }
    pub fn copy(&mut self, a: &Buffer, b: &Buffer, offset: usize) -> Result<()> {
        if offset.checked_add(a.len()).is_none_or(|n| n > b.len()) {
            return Err(shape("copy"));
        }
        self.linear(
            "copy_f32",
            &[a, b],
            Params {
                n: u(a.len())?,
                a: u(offset)?,
                ..Params::default()
            },
        )
    }
    pub fn gather(
        &mut self,
        x: &Buffer,
        y: &Buffer,
        block: usize,
        stride: usize,
        offset: usize,
    ) -> Result<()> {
        if block == 0
            || y.len() % block != 0
            || stride < block
            || offset
                .checked_add(
                    (y.len() / block - 1)
                        .checked_mul(stride)
                        .ok_or_else(|| shape("gather overflow"))?,
                )
                .and_then(|n| n.checked_add(block))
                .is_none_or(|n| n > x.len())
        {
            return Err(shape("gather"));
        }
        self.linear(
            "gather",
            &[x, y],
            Params {
                n: u(y.len())?,
                a: u(block)?,
                b: u(stride)?,
                c: u(offset)?,
                ..Params::default()
            },
        )
    }
    pub fn gate_scalar(&mut self, x: &Buffer, g: &Buffer, y: &Buffer) -> Result<()> {
        if x.len() != y.len() || g.len() != 1 {
            return Err(shape("scalar gate"));
        }
        self.linear(
            "gate_scalar",
            &[x, g, y],
            Params {
                n: u(x.len())?,
                ..Params::default()
            },
        )
    }
    pub fn mix(&mut self, x: &Buffer, w: &Buffer, y: &Buffer) -> Result<()> {
        if x.len() != w.len() * y.len() {
            return Err(shape("mix"));
        }
        self.linear(
            "mix",
            &[x, w, y],
            Params {
                n: u(y.len())?,
                a: u(w.len())?,
                ..Params::default()
            },
        )
    }
}
