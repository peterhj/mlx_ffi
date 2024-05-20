use crate::{MlxArray, MlxStream};
use crate::bindings::*;

use std::ops::{Add, Sub, Mul, Div, Neg};

impl<R: AsRef<MlxArray>> Add<R> for MlxArray {
  type Output = MlxArray;

  fn add(self, rhs: R) -> MlxArray {
    (&self).add(rhs)
  }
}

impl<'this, R: AsRef<MlxArray>> Add<R> for &'this MlxArray {
  type Output = MlxArray;

  fn add(self, rhs: R) -> MlxArray {
    // FIXME: default stream.
    let stm = MlxStream::default_cpu();
    let out_raw = unsafe { mlx_add(self.raw, rhs.as_ref().raw, stm.raw) };
    MlxArray{raw: out_raw}
  }
}

impl<R: AsRef<MlxArray>> Sub<R> for MlxArray {
  type Output = MlxArray;

  fn sub(self, rhs: R) -> MlxArray {
    (&self).sub(rhs)
  }
}

impl<'this, R: AsRef<MlxArray>> Sub<R> for &'this MlxArray {
  type Output = MlxArray;

  fn sub(self, rhs: R) -> MlxArray {
    // FIXME: default stream.
    let stm = MlxStream::default_cpu();
    let out_raw = unsafe { mlx_subtract(self.raw, rhs.as_ref().raw, stm.raw) };
    MlxArray{raw: out_raw}
  }
}

impl<R: AsRef<MlxArray>> Mul<R> for MlxArray {
  type Output = MlxArray;

  fn mul(self, rhs: R) -> MlxArray {
    (&self).mul(rhs)
  }
}

impl<'this, R: AsRef<MlxArray>> Mul<R> for &'this MlxArray {
  type Output = MlxArray;

  fn mul(self, rhs: R) -> MlxArray {
    // FIXME: default stream.
    let stm = MlxStream::default_cpu();
    let out_raw = unsafe { mlx_multiply(self.raw, rhs.as_ref().raw, stm.raw) };
    MlxArray{raw: out_raw}
  }
}

impl<R: AsRef<MlxArray>> Div<R> for MlxArray {
  type Output = MlxArray;

  fn div(self, rhs: R) -> MlxArray {
    (&self).div(rhs)
  }
}

impl<'this, R: AsRef<MlxArray>> Div<R> for &'this MlxArray {
  type Output = MlxArray;

  fn div(self, rhs: R) -> MlxArray {
    // FIXME: default stream.
    let stm = MlxStream::default_cpu();
    let out_raw = unsafe { mlx_divide(self.raw, rhs.as_ref().raw, stm.raw) };
    MlxArray{raw: out_raw}
  }
}

impl Neg for MlxArray {
  type Output = MlxArray;

  fn neg(self) -> MlxArray {
    (&self).neg()
  }
}

impl<'this> Neg for &'this MlxArray {
  type Output = MlxArray;

  fn neg(self) -> MlxArray {
    // FIXME: default stream.
    let stm = MlxStream::default_cpu();
    let out_raw = unsafe { mlx_negative(self.raw, stm.raw) };
    MlxArray{raw: out_raw}
  }
}

impl MlxArray {
  pub fn copy_(&self) -> MlxArray {
    // FIXME: default stream.
    let stm = MlxStream::default_cpu();
    let out_raw = unsafe { mlx_copy(self.raw, stm.raw) };
    MlxArray{raw: out_raw}
  }

  pub fn abs(&self) -> MlxArray {
    // FIXME: default stream.
    let stm = MlxStream::default_cpu();
    let out_raw = unsafe { mlx_abs(self.raw, stm.raw) };
    MlxArray{raw: out_raw}
  }

  pub fn exp(&self) -> MlxArray {
    // FIXME: default stream.
    let stm = MlxStream::default_cpu();
    let out_raw = unsafe { mlx_exp(self.raw, stm.raw) };
    MlxArray{raw: out_raw}
  }
}
