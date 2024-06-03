extern crate half;

use crate::bindings::*;

use std::cell::{RefCell};
use std::ffi::{CStr};
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::slice::{from_raw_parts};

pub mod bindings;
pub mod ops;
pub mod prelude;

pub fn metal_is_available() -> bool {
  unsafe { mlx_metal_is_available() }
}

pub struct MlxString {
  raw:  mlx_string,
}

impl Drop for MlxString {
  fn drop(&mut self) {
    unsafe { mlx_free(self.raw as _) };
  }
}

impl Clone for MlxString {
  fn clone(&self) -> MlxString {
    unsafe { mlx_retain(self.raw as _) };
    MlxString{raw: self.raw}
  }
}

impl Debug for MlxString {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{:?}", self.as_cstr().to_string_lossy())
  }
}

impl Display for MlxString {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{}", self.as_cstr().to_string_lossy())
  }
}

impl MlxString {
  pub fn as_cstr<'this>(&'this self) -> &'this CStr {
    let raw_cstr = unsafe { mlx_string_data(self.raw) };
    assert!(!raw_cstr.is_null());
    unsafe { CStr::from_ptr(raw_cstr) }
  }

  pub fn get_raw(&self) -> mlx_string {
    self.raw
  }
}

thread_local! {
  static TL_CPU_DEV: MlxDevice = MlxDevice::cpu(0);
  static TL_CPU_STM: MlxStream = MlxStream::default_cpu();
  static TL_GPU_DEV: MlxDevice = MlxDevice::gpu(0);
  static TL_GPU_STM: MlxStream = MlxStream::default_gpu();
  static TL_DEF_STM: RefCell<MlxStream> = RefCell::new(MlxStream::default_cpu());
  //static TL_DEF_STM: RefCell<DevStm> = RefCell::new(MlxStream::default_cpu());
}

pub struct DevStm {
  dev:  MlxDevice,
  stm:  MlxStream,
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
#[repr(i32)]
pub enum MlxDeviceType {
  Cpu = 0,
  Gpu,
}

impl MlxDeviceType {
  pub fn from_raw(raw_devty: mlx_device_type) -> MlxDeviceType {
    match raw_devty {
      MLX_CPU => MlxDeviceType::Cpu,
      MLX_GPU => MlxDeviceType::Gpu,
      _ => unimplemented!()
    }
  }

  pub fn to_raw(&self) -> mlx_device_type {
    match *self {
      MlxDeviceType::Cpu => MLX_CPU,
      MlxDeviceType::Gpu => MLX_GPU,
      _ => unimplemented!()
    }
  }
}

pub struct MlxDevice {
  raw:  mlx_device,
}

impl Debug for MlxDevice {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "MlxDevice({})", self.to_debug_string())
  }
}

impl MlxDevice {
  pub fn cpu(index: i32) -> MlxDevice {
    let raw = unsafe { mlx_device_new(MLX_CPU, index) };
    MlxDevice{raw}
  }

  pub fn gpu(index: i32) -> MlxDevice {
    let raw = unsafe { mlx_device_new(MLX_GPU, index) };
    MlxDevice{raw}
  }

  pub fn default() -> MlxDevice {
    let raw = unsafe { mlx_default_device() };
    MlxDevice{raw}
  }

  pub fn set_default(&self) -> MlxDevice {
    let raw = unsafe { mlx_set_default_device(self.raw) };
    MlxDevice{raw}
  }

  pub fn type_(&self) -> MlxDeviceType {
    let raw_devty = unsafe { mlx_device_get_type(self.raw) };
    MlxDeviceType::from_raw(raw_devty)
  }

  pub fn to_debug_string(&self) -> MlxString {
    let str_raw = unsafe { mlx_tostring(self.raw as _) };
    MlxString{raw: str_raw}
  }

  pub fn get_raw(&self) -> mlx_device {
    self.raw
  }
}

pub struct MlxStream {
  raw:  mlx_stream,
}

impl Drop for MlxStream {
  fn drop(&mut self) {
    unsafe { mlx_free(self.raw as _) };
  }
}

impl Clone for MlxStream {
  fn clone(&self) -> MlxStream {
    unsafe { mlx_retain(self.raw as _) };
    MlxStream{raw: self.raw}
  }
}

impl PartialEq for MlxStream {
  fn eq(&self, rstm: &MlxStream) -> bool {
    if self.raw == rstm.raw {
      return true;
    }
    unsafe { mlx_stream_equal(self.raw, rstm.raw) }
  }
}

impl Eq for MlxStream {}

impl Debug for MlxStream {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "MlxStream({})", self.to_debug_string())
  }
}

impl MlxStream {
  pub fn new(index: i32, dev: MlxDevice) -> MlxStream {
    let raw = unsafe { mlx_stream_new(index, dev.raw) };
    MlxStream{raw}
  }

  pub fn new_on_device(dev: MlxDevice) -> MlxStream {
    let raw = unsafe { mlx_stream_new_on_device(dev.raw) };
    MlxStream{raw}
  }

  pub fn default_cpu() -> MlxStream {
    let raw = unsafe { mlx_cpu_stream() };
    MlxStream{raw}
  }

  pub fn default_gpu() -> MlxStream {
    let raw = unsafe { mlx_gpu_stream() };
    MlxStream{raw}
  }

  pub fn default(dev: MlxDevice) -> MlxStream {
    let raw = unsafe { mlx_default_stream(dev.raw) };
    MlxStream{raw}
  }

  pub fn set_default(&self, dev: MlxDevice) -> MlxStream {
    let raw = unsafe { mlx_set_default_stream(dev.raw, self.raw) };
    MlxStream{raw}
  }

  pub fn tl_default() -> MlxStream {
    TL_DEF_STM.with(|defstm| {
      let defstm = defstm.borrow().clone();
      defstm
    })
  }

  pub fn set_tl_default(&self) -> MlxStream {
    TL_DEF_STM.with(|defstm| {
      let mut defstm = defstm.borrow_mut();
      let odefstm = defstm.clone();
      *defstm = self.clone();
      odefstm
    })
  }

  pub fn get_device(&self) -> MlxDevice {
    let raw = unsafe { mlx_stream_get_device(self.raw) };
    MlxDevice{raw}
  }

  pub fn synchronize(&self) {
    unsafe { mlx_stream_synchronize(self.raw) };
  }

  pub fn to_debug_string(&self) -> MlxString {
    let str_raw = unsafe { mlx_tostring(self.raw as _) };
    MlxString{raw: str_raw}
  }

  pub fn get_raw(&self) -> mlx_stream {
    self.raw
  }
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
#[repr(i32)]
pub enum MlxDtype {
  Bool = 0,
  Uint8,
  Uint16,
  Uint32,
  Uint64,
  Int8,
  Int16,
  Int32,
  Int64,
  Float16,
  Float32,
  Bfloat16,
  Complex64,
}

impl MlxDtype {
  pub fn from_raw(raw_dty: mlx_array_dtype) -> MlxDtype {
    match raw_dty {
      MLX_BOOL => MlxDtype::Bool,
      MLX_UINT8 => MlxDtype::Uint8,
      MLX_UINT16 => MlxDtype::Uint16,
      MLX_UINT32 => MlxDtype::Uint32,
      MLX_UINT64 => MlxDtype::Uint64,
      MLX_INT8 => MlxDtype::Int8,
      MLX_INT16 => MlxDtype::Int16,
      MLX_INT32 => MlxDtype::Int32,
      MLX_INT64 => MlxDtype::Int64,
      MLX_FLOAT16 => MlxDtype::Float16,
      MLX_FLOAT32 => MlxDtype::Float32,
      MLX_BFLOAT16 => MlxDtype::Bfloat16,
      MLX_COMPLEX64 => MlxDtype::Complex64,
      _ => unimplemented!()
    }
  }

  pub fn to_raw(&self) -> mlx_array_dtype {
    match *self {
      MlxDtype::Bool => MLX_BOOL,
      MlxDtype::Uint8 => MLX_UINT8,
      MlxDtype::Uint16 => MLX_UINT16,
      MlxDtype::Uint32 => MLX_UINT32,
      MlxDtype::Uint64 => MLX_UINT64,
      MlxDtype::Int8 => MLX_INT8,
      MlxDtype::Int16 => MLX_INT16,
      MlxDtype::Int32 => MLX_INT32,
      MlxDtype::Int64 => MLX_INT64,
      MlxDtype::Float16 => MLX_FLOAT16,
      MlxDtype::Float32 => MLX_FLOAT32,
      MlxDtype::Bfloat16 => MLX_BFLOAT16,
      MlxDtype::Complex64 => MLX_COMPLEX64,
      _ => unimplemented!()
    }
  }
}

pub struct MlxArray {
  raw:  mlx_array,
}

impl Drop for MlxArray {
  fn drop(&mut self) {
    unsafe { mlx_free(self.raw as _) };
  }
}

impl Clone for MlxArray {
  fn clone(&self) -> MlxArray {
    unsafe { mlx_retain(self.raw as _) };
    MlxArray{raw: self.raw}
  }
}

impl AsRef<MlxArray> for MlxArray {
  fn as_ref(&self) -> &MlxArray {
    self
  }
}

impl Debug for MlxArray {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "MlxArray({})", self.to_debug_string())
  }
}

impl MlxArray {
  pub fn flat_len(&self) -> usize {
    unsafe { mlx_array_size(self.raw) }
  }

  pub fn ndim(&self) -> usize {
    unsafe { mlx_array_ndim(self.raw) }
  }

  pub fn shape(&self) -> Box<[i64]> {
    let ndim = unsafe { mlx_array_ndim(self.raw) };
    let mut shape = Vec::with_capacity(ndim);
    for k in 0 .. ndim {
      let v = unsafe { mlx_array_dim(self.raw, k as _) };
      shape.push(v as _);
    }
    shape.into()
  }

  pub fn dtype(&self) -> MlxDtype {
    let raw_dty = unsafe { mlx_array_get_dtype(self.raw) };
    MlxDtype::from_raw(raw_dty)
  }

  pub fn eval(&self) {
    unsafe { mlx_array_eval(self.raw) };
  }

  pub fn item_f32(&self) -> f32 {
    unsafe { mlx_array_item_float32(self.raw) }
  }

  /// SAFETY: this only returns a sensible result if the array layout
  /// is packed/non-strided.
  pub fn get_data_f32_unchecked<'this>(&'this self) -> Option<&'this [f32]> {
    let flat_len = self.flat_len();
    let ptr = unsafe { mlx_array_data_float32(self.raw) };
    if ptr.is_null() {
      return None;
    }
    Some(unsafe { from_raw_parts(ptr, flat_len) })
  }

  pub fn to_debug_string(&self) -> MlxString {
    let str_raw = unsafe { mlx_tostring(self.raw as _) };
    MlxString{raw: str_raw}
  }

  pub fn get_raw(&self) -> mlx_array {
    self.raw
  }
}

pub struct MlxVecArray {
  raw:  mlx_vector_array,
}

impl Drop for MlxVecArray {
  fn drop(&mut self) {
    unsafe { mlx_free(self.raw as _) };
  }
}

impl Clone for MlxVecArray {
  fn clone(&self) -> MlxVecArray {
    unsafe { mlx_retain(self.raw as _) };
    MlxVecArray{raw: self.raw}
  }
}

impl Debug for MlxVecArray {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "MlxVecArray({})", self.to_debug_string())
  }
}

impl MlxVecArray {
  pub fn new() -> MlxVecArray {
    let raw = unsafe { mlx_vector_array_new() };
    MlxVecArray{raw}
  }

  pub fn len(&self) -> usize {
    unsafe { mlx_vector_array_size(self.raw) }
  }

  pub fn get(&self, index: usize) -> Option<MlxArray> {
    if index >= self.len() {
      return None;
    }
    let arr_raw = unsafe { mlx_vector_array_get(self.raw, index) };
    Some(MlxArray{raw: arr_raw})
  }

  pub fn push<A: AsRef<MlxArray>>(&self, arr: A) {
    let arr = arr.as_ref();
    unsafe { mlx_vector_array_add(self.raw, arr.raw) };
  }

  pub fn to_debug_string(&self) -> MlxString {
    let str_raw = unsafe { mlx_tostring(self.raw as _) };
    MlxString{raw: str_raw}
  }

  pub fn get_raw(&self) -> mlx_vector_array {
    self.raw
  }
}

pub struct MlxVecVecArray {
  raw:  mlx_vector_vector_array,
}

impl Drop for MlxVecVecArray {
  fn drop(&mut self) {
    unsafe { mlx_free(self.raw as _) };
  }
}

impl Clone for MlxVecVecArray {
  fn clone(&self) -> MlxVecVecArray {
    unsafe { mlx_retain(self.raw as _) };
    MlxVecVecArray{raw: self.raw}
  }
}

impl Debug for MlxVecVecArray {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "MlxVecVecArray({})", self.to_debug_string())
  }
}

impl MlxVecVecArray {
  pub fn new() -> MlxVecVecArray {
    let raw = unsafe { mlx_vector_vector_array_new() };
    MlxVecVecArray{raw}
  }

  pub fn len(&self) -> usize {
    unsafe { mlx_vector_vector_array_size(self.raw) }
  }

  pub fn get(&self, index: usize) -> Option<MlxVecArray> {
    if index >= self.len() {
      return None;
    }
    let varr_raw = unsafe { mlx_vector_vector_array_get(self.raw, index) };
    Some(MlxVecArray{raw: varr_raw})
  }

  pub fn push<V: AsRef<MlxVecArray>>(&self, varr: V) {
    let varr = varr.as_ref();
    unsafe { mlx_vector_vector_array_add(self.raw, varr.raw) };
  }

  pub fn to_debug_string(&self) -> MlxString {
    let str_raw = unsafe { mlx_tostring(self.raw as _) };
    MlxString{raw: str_raw}
  }

  pub fn get_raw(&self) -> mlx_vector_vector_array {
    self.raw
  }
}

pub struct MlxClosure {
  raw:  mlx_closure,
}

impl Drop for MlxClosure {
  fn drop(&mut self) {
    unsafe { mlx_free(self.raw as _) };
  }
}

impl Clone for MlxClosure {
  fn clone(&self) -> MlxClosure {
    unsafe { mlx_retain(self.raw as _) };
    MlxClosure{raw: self.raw}
  }
}

impl Debug for MlxClosure {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "MlxClosure({})", self.to_debug_string())
  }
}

impl MlxClosure {
  pub fn new(raw_fun: unsafe extern "C" fn (mlx_vector_array_const) -> mlx_vector_array) -> MlxClosure {
    let raw = unsafe { mlx_closure_new(raw_fun) };
    MlxClosure{raw}
  }

  pub fn new_unary(raw_fun: unsafe extern "C" fn (mlx_array_const) -> mlx_array) -> MlxClosure {
    let raw = unsafe { mlx_closure_new_unary(raw_fun) };
    MlxClosure{raw}
  }

  pub fn apply<V: AsRef<MlxVecArray>>(&self, args: V) -> MlxVecArray {
    let args = args.as_ref();
    let outs_raw = unsafe { mlx_closure_apply(self.raw, args.raw) };
    MlxVecArray{raw: outs_raw}
  }

  pub fn value_and_grad(&self) -> MlxClosureValueAndGrad {
    // FIXME: argnums.
    let argnums = [0];
    let out_raw = unsafe { mlx_value_and_grad(self.raw, argnums.as_ptr(), argnums.len()) };
    MlxClosureValueAndGrad{raw: out_raw}
  }

  pub fn vjp<U: AsRef<MlxVecArray>, V: AsRef<MlxVecArray>>(&self, primals: U, cotangents: V) -> MlxVecVecArray {
    let primals = primals.as_ref();
    let cotangents = cotangents.as_ref();
    let out_raw = unsafe { mlx_vjp(self.raw, primals.raw, cotangents.raw) };
    MlxVecVecArray{raw: out_raw}
  }

  pub fn jvp<U: AsRef<MlxVecArray>, V: AsRef<MlxVecArray>>(&self, primals: U, tangents: V) -> MlxVecVecArray {
    let primals = primals.as_ref();
    let tangents = tangents.as_ref();
    let out_raw = unsafe { mlx_jvp(self.raw, primals.raw, tangents.raw) };
    MlxVecVecArray{raw: out_raw}
  }

  pub fn to_debug_string(&self) -> MlxString {
    let str_raw = unsafe { mlx_tostring(self.raw as _) };
    MlxString{raw: str_raw}
  }

  pub fn get_raw(&self) -> mlx_closure {
    self.raw
  }
}

pub struct MlxClosureValueAndGrad {
  raw:  mlx_closure_value_and_grad,
}

impl Drop for MlxClosureValueAndGrad {
  fn drop(&mut self) {
    unsafe { mlx_free(self.raw as _) };
  }
}

impl Clone for MlxClosureValueAndGrad {
  fn clone(&self) -> MlxClosureValueAndGrad {
    unsafe { mlx_retain(self.raw as _) };
    MlxClosureValueAndGrad{raw: self.raw}
  }
}

impl Debug for MlxClosureValueAndGrad {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "MlxClosureValueAndGrad({})", self.to_debug_string())
  }
}

impl MlxClosureValueAndGrad {
  pub fn apply<V: AsRef<MlxVecArray>>(&self, args: V) -> (MlxVecArray, MlxVecArray) {
    let args = args.as_ref();
    let outs_raw = unsafe { mlx_closure_value_and_grad_apply(self.raw, args.raw) };
    let outs = MlxVecVecArray{raw: outs_raw};
    let value = outs.get(0).unwrap();
    let grad = outs.get(1).unwrap();
    (value, grad)
  }

  pub fn to_debug_string(&self) -> MlxString {
    let str_raw = unsafe { mlx_tostring(self.raw as _) };
    MlxString{raw: str_raw}
  }

  pub fn get_raw(&self) -> mlx_closure_value_and_grad {
    self.raw
  }
}
