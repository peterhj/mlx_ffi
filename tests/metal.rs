extern crate mlx_ffi;

use mlx_ffi::*;

#[test]
fn test_metal() {
  assert!(metal_is_available());
}
