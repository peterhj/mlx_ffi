#![allow(non_camel_case_types)]

use half::{bf16, f16};

use std::ffi::{c_char, c_int, c_void};

pub type mlx_string = *mut mlx_string_;

#[repr(C)]
pub struct mlx_string_ { _unused: [u8; 0] }

extern "C" { pub fn mlx_string_data(str_: mlx_string) -> *const c_char; }

extern "C" { pub fn mlx_tostring(obj: *mut c_void) -> mlx_string; }
extern "C" { pub fn mlx_retain(obj: *mut c_void); }
extern "C" { pub fn mlx_free(obj: *mut c_void); }

extern "C" { pub fn mlx_metal_is_available() -> bool; }
extern "C" { pub fn mlx_metal_clear_cache(); }
extern "C" { pub fn mlx_metal_get_active_memory() -> usize; }
extern "C" { pub fn mlx_metal_get_cache_memory() -> usize; }
extern "C" { pub fn mlx_metal_get_peak_memory() -> usize; }

pub type mlx_device = *mut mlx_device_;

#[repr(C)]
pub struct mlx_device_ { _unused: [u8; 0] }

pub type mlx_device_type = mlx_device_type_;
pub type mlx_device_type_ = c_int;
pub const MLX_CPU: mlx_device_type_ = 0;
pub const MLX_GPU: mlx_device_type_ = 1;

extern "C" { pub fn mlx_device_new(type_: mlx_device_type, index: c_int) -> mlx_device; }
extern "C" { pub fn mlx_device_get_type(dev: mlx_device) -> mlx_device_type; }
extern "C" { pub fn mlx_default_device() -> mlx_device; }
extern "C" { pub fn mlx_set_default_device(dev: mlx_device) -> mlx_device; }

pub type mlx_stream = *mut mlx_stream_;

#[repr(C)]
pub struct mlx_stream_ { _unused: [u8; 0] }

extern "C" { pub fn mlx_stream_new(index: c_int, dev: mlx_device) -> mlx_stream; }
extern "C" { pub fn mlx_stream_new_on_device(dev: mlx_device) -> mlx_stream; }
extern "C" { pub fn mlx_stream_equal(lstm: mlx_stream, rstm: mlx_stream) -> bool; }
extern "C" { pub fn mlx_stream_synchronize(stm: mlx_stream); }
extern "C" { pub fn mlx_default_stream(dev: mlx_device) -> mlx_stream; }
extern "C" { pub fn mlx_set_default_stream(dev: mlx_device, stm: mlx_stream) -> mlx_stream; }
extern "C" { pub fn mlx_cpu_stream() -> mlx_stream; }
extern "C" { pub fn mlx_gpu_stream() -> mlx_stream; }

pub type mlx_array = *mut mlx_array_;
pub type mlx_array_const = *const mlx_array_;

#[repr(C)]
pub struct mlx_array_ { _unused: [u8; 0] }

pub type mlx_array_dtype = mlx_array_dtype_;
pub type mlx_array_dtype_ = c_int;
pub const MLX_BOOL: mlx_array_dtype_ = 0;
pub const MLX_UINT8: mlx_array_dtype_ = 1;
pub const MLX_UINT16: mlx_array_dtype_ = 2;
pub const MLX_UINT32: mlx_array_dtype_ = 3;
pub const MLX_UINT64: mlx_array_dtype_ = 4;
pub const MLX_INT8: mlx_array_dtype_ = 5;
pub const MLX_INT16: mlx_array_dtype_ = 6;
pub const MLX_INT32: mlx_array_dtype_ = 7;
pub const MLX_INT64: mlx_array_dtype_ = 8;
pub const MLX_FLOAT16: mlx_array_dtype_ = 9;
pub const MLX_FLOAT32: mlx_array_dtype_ = 10;
pub const MLX_BFLOAT16: mlx_array_dtype_ = 11;
pub const MLX_COMPLEX64: mlx_array_dtype_ = 12;

// TODO: dtype constructors.
extern "C" { pub fn mlx_array_from_int(val: c_int) -> mlx_array; }
extern "C" { pub fn mlx_array_from_float(val: f32) -> mlx_array; }
extern "C" { pub fn mlx_array_from_data(
    data: *const c_void,
    shape: *const c_int,
    dim: c_int,
    dtype: mlx_array_dtype,
) -> mlx_array; }
extern "C" { pub fn mlx_array_itemsize(arr: mlx_array) -> usize; }
extern "C" { pub fn mlx_array_size(arr: mlx_array) -> usize; }
extern "C" { pub fn mlx_array_nbytes(arr: mlx_array) -> usize; }
extern "C" { pub fn mlx_array_ndim(arr: mlx_array) -> usize; }
extern "C" { pub fn mlx_array_shape(arr: mlx_array) -> *mut c_int; }
extern "C" { pub fn mlx_array_strides(arr: mlx_array) -> *mut usize; }
extern "C" { pub fn mlx_array_dim(arr: mlx_array, dim: c_int) -> c_int; }
extern "C" { pub fn mlx_array_get_dtype(arr: mlx_array) -> mlx_array_dtype; }
extern "C" { pub fn mlx_array_eval(arr: mlx_array); }

extern "C" { pub fn mlx_array_item_bool(arr: mlx_array) -> bool; }
extern "C" { pub fn mlx_array_item_uint8(arr: mlx_array) -> u8; }
extern "C" { pub fn mlx_array_item_uint16(arr: mlx_array) -> u16; }
extern "C" { pub fn mlx_array_item_uint32(arr: mlx_array) -> u32; }
extern "C" { pub fn mlx_array_item_uint64(arr: mlx_array) -> u64; }
extern "C" { pub fn mlx_array_item_int8(arr: mlx_array) -> i8; }
extern "C" { pub fn mlx_array_item_int16(arr: mlx_array) -> i16; }
extern "C" { pub fn mlx_array_item_int32(arr: mlx_array) -> i32; }
extern "C" { pub fn mlx_array_item_int64(arr: mlx_array) -> i64; }
extern "C" { pub fn mlx_array_item_float32(arr: mlx_array) -> f32; }
// FIXME: ABI compat?
//extern "C" { pub fn mlx_array_item_float16(arr: mlx_array) -> f16; }
//extern "C" { pub fn mlx_array_item_bfloat16(arr: mlx_array) -> bf16; }

extern "C" { pub fn mlx_array_data_bool(arr: mlx_array) -> *const bool; }
extern "C" { pub fn mlx_array_data_uint8(arr: mlx_array) -> *const u8; }
extern "C" { pub fn mlx_array_data_uint16(arr: mlx_array) -> *const u16; }
extern "C" { pub fn mlx_array_data_uint32(arr: mlx_array) -> *const u32; }
extern "C" { pub fn mlx_array_data_uint64(arr: mlx_array) -> *const u64; }
extern "C" { pub fn mlx_array_data_int8(arr: mlx_array) -> *const i8; }
extern "C" { pub fn mlx_array_data_int16(arr: mlx_array) -> *const i16; }
extern "C" { pub fn mlx_array_data_int32(arr: mlx_array) -> *const i32; }
extern "C" { pub fn mlx_array_data_int64(arr: mlx_array) -> *const i64; }
extern "C" { pub fn mlx_array_data_float32(arr: mlx_array) -> *const f32; }
extern "C" { pub fn mlx_array_data_float16(arr: mlx_array) -> *const f16; }
extern "C" { pub fn mlx_array_data_bfloat16(arr: mlx_array) -> *const bf16; }

pub type mlx_vector_array = *mut mlx_vector_array_;
pub type mlx_vector_array_const = *const mlx_vector_array_;

#[repr(C)]
pub struct mlx_vector_array_ { _unused: [u8; 0] }

extern "C" { pub fn mlx_vector_array_new() -> mlx_vector_array; }
extern "C" { pub fn mlx_vector_array_from_arrays(arrs: *mut mlx_array, num_arrs: usize) -> mlx_vector_array; }
extern "C" { pub fn mlx_vector_array_from_array(arr: mlx_array_const) -> mlx_vector_array; }
extern "C" { pub fn mlx_vector_array_add(vec: mlx_vector_array, arr: mlx_array_const); }
extern "C" { pub fn mlx_vector_array_add_arrays(vec: mlx_vector_array, arrs: *mut mlx_array_const, num_arrs: usize); }
extern "C" { pub fn mlx_vector_array_get(vec: mlx_vector_array, index: usize) -> mlx_array; }
extern "C" { pub fn mlx_vector_array_size(vec: mlx_vector_array) -> usize; }

pub type mlx_vector_vector_array = *mut mlx_vector_vector_array_;
pub type mlx_vector_vector_array_const = *const mlx_vector_vector_array_;

#[repr(C)]
pub struct mlx_vector_vector_array_ { _unused: [u8; 0] }

extern "C" { pub fn mlx_vector_vector_array_new() -> mlx_vector_vector_array; }
extern "C" { pub fn mlx_vector_vector_array_add(vec2: mlx_vector_vector_array, vec: mlx_vector_array_const); }
extern "C" { pub fn mlx_vector_vector_array_get(vec2: mlx_vector_vector_array, index: usize) -> mlx_vector_array; }
extern "C" { pub fn mlx_vector_vector_array_get2d(vec2: mlx_vector_vector_array, index: usize, arr_index: usize) -> mlx_array; }
extern "C" { pub fn mlx_vector_vector_array_size(vec2: mlx_vector_vector_array) -> usize; }

pub type mlx_closure = *mut mlx_closure_;

#[repr(C)]
pub struct mlx_closure_ { _unused: [u8; 0] }

extern "C" { pub fn mlx_closure_new(fun: unsafe extern "C" fn (mlx_vector_array_const) -> mlx_vector_array) -> mlx_closure; }
extern "C" { pub fn mlx_closure_new_unary(fun: unsafe extern "C" fn (mlx_array_const) -> mlx_array) -> mlx_closure; }
extern "C" { pub fn mlx_closure_apply(cls: mlx_closure, inputs: mlx_vector_array_const) -> mlx_vector_array; }

pub type mlx_closure_value_and_grad = *mut mlx_closure_value_and_grad_;

#[repr(C)]
pub struct mlx_closure_value_and_grad_ { _unused: [u8; 0] }

extern "C" { pub fn mlx_closure_value_and_grad_apply(cls: mlx_closure_value_and_grad, inputs: mlx_vector_array_const) -> mlx_vector_vector_array; }

extern "C" { pub fn mlx_async_eval(outputs: mlx_vector_array_const); }
extern "C" { pub fn mlx_eval(outputs: mlx_vector_array_const); }
extern "C" { pub fn mlx_value_and_grad(fun: mlx_closure, argnums: *const c_int, num_argnums: usize) -> mlx_closure_value_and_grad; }
extern "C" { pub fn mlx_vjp(fun: mlx_closure, primals: mlx_vector_array_const, cotangents: mlx_vector_array_const) -> mlx_vector_vector_array; }
extern "C" { pub fn mlx_jvp(fun: mlx_closure, primals: mlx_vector_array_const, tangents: mlx_vector_array_const) -> mlx_vector_vector_array; }

extern "C" { pub fn mlx_abs(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_add(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_addmm(c: mlx_array, a: mlx_array, b: mlx_array, alpha: f32, beta: f32, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_all_axes(a: mlx_array, axes: *const c_int, num_axes: usize, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_all_axis(a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_all_all(a: mlx_array, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_allclose(a: mlx_array, b: mlx_array, rtol: f64, atol: f64, equal_nan: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_any(a: mlx_array, axes: *const c_int, num_axes: usize, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_any_all(a: mlx_array, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_arange(start: f64, stop: f64, step: f64, dtype: mlx_array_dtype, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_arccos(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_arccosh(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_arcsin(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_arcsinh(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_arctan(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_arctanh(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_argmax(a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_argmax_all(a: mlx_array, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_argmin(a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_argmin_all(a: mlx_array, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_argpartition(a: mlx_array, kth: c_int, axis: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_argpartition_all(a: mlx_array, kth: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_argsort(a: mlx_array, axis: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_argsort_all(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_array_equal(a: mlx_array, b: mlx_array, equal_nan: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_as_strided(a: mlx_array, shape: *const c_int, num_shape: usize, strides: *const usize, num_strides: usize, offset: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_astype(a: mlx_array, dtype: mlx_array_dtype, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_atleast_1d(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_atleast_2d(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_atleast_3d(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_block_masked_mm(a: mlx_array, b: mlx_array, block_size: c_int, mask_out: mlx_array, mask_lhs: mlx_array, mask_rhs: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_broadcast_arrays(inputs: mlx_vector_array_const, s: mlx_stream) -> mlx_vector_array; }
extern "C" { pub fn mlx_broadcast_to(a: mlx_array, shape: *const c_int, num_shape: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_ceil(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_clip(a: mlx_array, a_min: mlx_array, a_max: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_concatenate(arrays: mlx_vector_array_const, axis: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_concatenate_all(arrays: mlx_vector_array_const, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_conv1d(input: mlx_array, weight: mlx_array, stride: c_int, padding: c_int, dilation: c_int, groups: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_conv2d(input: mlx_array, weight: mlx_array, f_stride: c_int, s_stride: c_int, f_padding: c_int, s_padding: c_int, f_dilation: c_int, s_dilation: c_int, groups: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_conv_general(input: mlx_array, weight: mlx_array, stride: *const c_int, num_stride: usize, padding_lo: *const c_int, num_padding_lo: usize, padding_hi: *const c_int, num_padding_hi: usize, kernel_dilation: *const c_int, num_kernel_dilation: usize, input_dilation: *const c_int, num_input_dilation: usize, groups: c_int, flip: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_copy(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_cos(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_cosh(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_cummax(a: mlx_array, axis: c_int, reverse: bool, inclusive: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_cummin(a: mlx_array, axis: c_int, reverse: bool, inclusive: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_cumprod(a: mlx_array, axis: c_int, reverse: bool, inclusive: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_cumsum(a: mlx_array, axis: c_int, reverse: bool, inclusive: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_degrees(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_depends(inputs: *const mlx_vector_array, dependencies: *const mlx_vector_array) -> mlx_vector_array; }
extern "C" { pub fn mlx_dequantize(w: mlx_array, scales: mlx_array, biases: mlx_array, group_size: c_int, bits: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_diag(a: mlx_array, k: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_diagonal(a: mlx_array, offset: c_int, axis1: c_int, axis2: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_divide(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_divmod(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_vector_array; }
extern "C" { pub fn mlx_equal(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_erf(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_erfinv(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_exp(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_expand_dims(a: mlx_array, axes: *const c_int, num_axes: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_expm1(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_eye(n: c_int, m: c_int, k: c_int, dtype: mlx_array_dtype, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_flatten(a: mlx_array, start_axis: c_int, end_axis: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_floor(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_floor_divide(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_full(shape: *const c_int, num_shape: usize, vals: mlx_array, dtype: mlx_array_dtype, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_gather(a: mlx_array, indices: *const mlx_vector_array, axes: *const c_int, num_axes: usize, slice_sizes: *const c_int, num_slice_sizes: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_greater(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_greater_equal(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_identity(n: c_int, dtype: mlx_array_dtype, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_inner(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_isclose(a: mlx_array, b: mlx_array, rtol: f64, atol: f64, equal_nan: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_isinf(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_isnan(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_isneginf(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_isposinf(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_less(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_less_equal(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_linspace(start: f64, stop: f64, num: c_int, dtype: mlx_array_dtype, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_log(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_log10(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_log1p(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_log2(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_logaddexp(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_logical_and(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_logical_not(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_logical_or(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_logsumexp(a: mlx_array, axes: *const c_int, num_axes: usize, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_logsumexp_all(a: mlx_array, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_matmul(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_max(a: mlx_array, axes: *const c_int, num_axes: usize, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_max_all(a: mlx_array, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_maximum(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_mean(a: mlx_array, axes: *const c_int, num_axes: usize, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_mean_all(a: mlx_array, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_meshgrid(arrays: mlx_vector_array_const, sparse: bool, indexing: mlx_string, s: mlx_stream) -> mlx_vector_array; }
extern "C" { pub fn mlx_min(a: mlx_array, axes: *const c_int, num_axes: usize, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_min_all(a: mlx_array, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_minimum(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_moveaxis(a: mlx_array, source: c_int, destination: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_multiply(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_negative(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_not_equal(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_number_of_elements(a: mlx_array, axes: *const c_int, num_axes: usize, inverted: bool, dtype: mlx_array_dtype, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_ones(shape: *const c_int, num_shape: usize, dtype: mlx_array_dtype, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_ones_like(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_outer(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_pad(a: mlx_array, axes: *const c_int, num_axes: usize, low_pad_size: *const c_int, num_low_pad_size: usize, high_pad_size: *const c_int, num_high_pad_size: usize, pad_value: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_partition(a: mlx_array, kth: c_int, axis: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_partition_all(a: mlx_array, kth: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_power(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_prod(a: mlx_array, axes: *const c_int, num_axes: usize, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_prod_all(a: mlx_array, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_quantize(w: mlx_array, group_size: c_int, bits: c_int, s: mlx_stream) -> mlx_vector_array; }
extern "C" { pub fn mlx_quantized_matmul(x: mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, transpose: bool, group_size: c_int, bits: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_radians(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_reciprocal(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_remainder(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_repeat(arr: mlx_array, repeats: c_int, axis: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_repeat_all(arr: mlx_array, repeats: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_reshape(a: mlx_array, shape: *const c_int, num_shape: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_round(a: mlx_array, decimals: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_rsqrt(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_scatter(a: mlx_array, indices: mlx_vector_array_const, updates: mlx_array, axes: *const c_int, num_axes: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_scatter_add(a: mlx_array, indices: mlx_vector_array_const, updates: mlx_array, axes: *const c_int, num_axes: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_scatter_max(a: mlx_array, indices: mlx_vector_array_const, updates: mlx_array, axes: *const c_int, num_axes: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_scatter_min(a: mlx_array, indices: mlx_vector_array_const, updates: mlx_array, axes: *const c_int, num_axes: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_scatter_prod(a: mlx_array, indices: mlx_vector_array_const, updates: mlx_array, axes: *const c_int, num_axes: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_sigmoid(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_sign(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_sin(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_sinh(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_slice(a: mlx_array, start: *const c_int, num_start: usize, stop: *const c_int, num_stop: usize, strides: *const c_int, num_strides: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_slice_update(src: mlx_array, update: mlx_array, start: *const c_int, num_start: usize, stop: *const c_int, num_stop: usize, strides: *const c_int, num_strides: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_softmax(a: mlx_array, axes: *const c_int, num_axes: usize, precise: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_softmax_all(a: mlx_array, precise: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_sort(a: mlx_array, axis: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_sort_all(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_split_equal_parts(a: mlx_array, num_splits: c_int, axis: c_int, s: mlx_stream) -> mlx_vector_array; }
extern "C" { pub fn mlx_split(a: mlx_array, indices: *const c_int, num_indices: usize, axis: c_int, s: mlx_stream) -> mlx_vector_array; }
extern "C" { pub fn mlx_sqrt(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_square(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_squeeze(a: mlx_array, axes: *const c_int, num_axes: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_squeeze_all(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_stack(arrays: mlx_vector_array_const, axis: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_stack_all(arrays: mlx_vector_array_const, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_std(a: mlx_array, axes: *const c_int, num_axes: usize, keepdims: bool, ddof: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_std_all(a: mlx_array, keepdims: bool, ddof: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_stop_gradient(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_subtract(a: mlx_array, b: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_sum(a: mlx_array, axes: *const c_int, num_axes: usize, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_sum_all(a: mlx_array, keepdims: bool, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_swapaxes(a: mlx_array, axis1: c_int, axis2: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_take(a: mlx_array, indices: mlx_array, axis: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_take_all(a: mlx_array, indices: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_take_along_axis(a: mlx_array, indices: mlx_array, axis: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_tan(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_tanh(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_tensordot(a: mlx_array, b: mlx_array, axes_a: *const c_int, num_axes_a: usize, axes_b: *const c_int, num_axes_b: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_tensordot_along_axis(a: mlx_array, b: mlx_array, axis: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_tile(arr: mlx_array, reps: *const c_int, num_reps: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_topk(a: mlx_array, k: c_int, axis: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_topk_all(a: mlx_array, k: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_transpose(a: mlx_array, axes: *const c_int, num_axes: usize, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_transpose_all(a: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_tri(n: c_int, m: c_int, k: c_int, type_: mlx_array_dtype, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_tril(x: mlx_array, k: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_triu(x: mlx_array, k: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_var(a: mlx_array, axes: *const c_int, num_axes: usize, keepdims: bool, ddof: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_var_all(a: mlx_array, keepdims: bool, ddof: c_int, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_where(condition: mlx_array, x: mlx_array, y: mlx_array, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_zeros(shape: *const c_int, num_shape: usize, dtype: mlx_array_dtype, s: mlx_stream) -> mlx_array; }
extern "C" { pub fn mlx_zeros_like(a: mlx_array, s: mlx_stream) -> mlx_array; }
