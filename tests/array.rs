extern crate mlx_ffi;

use mlx_ffi::*;
use mlx_ffi::ops::*;

#[test]
fn test_ones() {
  let x = MlxArray::ones(&[10], MlxDtype::Float32);
  let y = x.clone() + x;
}
