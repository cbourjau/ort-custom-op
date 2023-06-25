use std::ffi::CString;

use crate::bindings::*;
use crate::error::status_to_result;

use anyhow::Result;
use ndarray::{Array, ArrayD, ArrayView, ArrayViewD, ArrayViewMut, ArrayViewMutD};

pub const API_VERSION: u32 = 16;

#[derive(Debug)]
pub struct KernelInfo<'s> {
    api: &'static OrtApi,
    info: &'s OrtKernelInfo,
}

pub enum ElementType {
    Bool,
    F32,
    F64,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    String,
}

/// Create a new custom domain with the operators `ops`.
pub fn create_custom_op_domain(
    session_options: &mut OrtSessionOptions,
    api_base: &mut OrtApiBase,
    domain: &str,
    ops: &[&'static OrtCustomOp],
) -> OrtStatusPtr {
    let api = unsafe { api_base.GetApi.unwrap()(API_VERSION).as_ref().unwrap() };

    let fun_ptr = api.CreateCustomOpDomain.unwrap();
    let mut domain_ptr: *mut OrtCustomOpDomain = std::ptr::null_mut();

    // Copies and leaks!
    let c_op_domain = CString::new(domain).unwrap().into_raw();
    let domain = unsafe {
        // According to docs: "Must be freed with OrtApi::ReleaseCustomOpDomain"
        let ptr = fun_ptr(c_op_domain, &mut domain_ptr);
        if !ptr.is_null() {
            return ptr;
        }
        let ptr = api.AddCustomOpDomain.unwrap()(session_options, domain_ptr);
        if !ptr.is_null() {
            return ptr;
        }
        domain_ptr.as_mut().unwrap()
    };
    for op in ops {
        let ptr = add_op_to_domain(api, domain, op);
        if !ptr.is_null() {
            return ptr;
        }
    }
    std::ptr::null_mut()
}

/// Explicit struct around OrtTypeAndShapeInfo pointer since we are
/// responsible for properly dropping it.
#[derive(Debug)]
struct TensorTypeAndShapeInfo<'s> {
    api: &'s OrtApi,
    // must be mut so that we can later drop it
    info: &'s mut OrtTensorTypeAndShapeInfo,
}

/// Impls which simplify useful operation.
impl OrtApi {
    pub(crate) fn get_input_array<'s, T>(
        &self,
        context: &'s OrtKernelContext,
        index: usize,
    ) -> Result<ArrayViewD<'s, T>> {
        let value = self.get_input(context, index)?;
        let shape: Vec<_> = self
            .get_tensor_type_and_shape(value)?
            .get_dimensions()?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let mut_tensor = self.get_tensor_data_mut::<T>(value, &shape)?;
        let n_elems = mut_tensor.len();
        let s = mut_tensor.into_slice().unwrap();
        Ok(ArrayView::<'s, T, ndarray::Ix1>::from_shape((n_elems,), s)?
            .into_dyn()
            .into_shape(shape.as_slice())
            .unwrap())
    }

    pub(crate) fn get_input_array_string(
        &self,
        context: &OrtKernelContext,
        index: usize,
    ) -> Result<ArrayD<String>> {
        let value = self.get_input(context, index)?;
        self.get_string_tensor_data(value)
    }

    pub(crate) fn fill_string_tensor(
        &self,
        context: &mut OrtKernelContext,
        index: usize,
        array: ArrayD<String>,
    ) -> Result<()> {
        let shape = array.shape();
        let shape_i64: Vec<_> = shape.iter().map(|v| *v as i64).collect();

        let cstrings = array.mapv(|s| CString::new(s).unwrap());
        let pointers = cstrings.map(|s| s.as_ptr());

        // Make sure that the vector is not dealocated before the ptr is used!
        let vec_of_ptrs = pointers.into_raw_vec();
        let ptr_of_ptrs = vec_of_ptrs.as_ptr();
        let n_items = array.len();
        let val = unsafe { self.get_output(context, index, &shape_i64) }?;

        let fun = self.FillStringTensor.unwrap();
        status_to_result(unsafe { fun(val, ptr_of_ptrs, n_items) }, self)?;
        Ok(())
    }

    pub(crate) fn get_input_count(&self, context: &OrtKernelContext) -> Result<usize> {
        let fun = self.KernelContext_GetInputCount.unwrap();
        let mut out: usize = 0;
        status_to_result(unsafe { fun(context, &mut out) }, self)?;
        Ok(out)
    }
}

impl OrtApi {
    /// Get `OrtValue` for input with index `idx`.
    fn get_input<'s>(&self, ctx: &'s OrtKernelContext, idx: usize) -> Result<&'s mut OrtValue> {
        let fun = self.KernelContext_GetInput.unwrap();

        let mut value: *const OrtValue = std::ptr::null();
        status_to_result(unsafe { fun(ctx, idx, &mut (value)) }, self)?;

        // Code crime!
        let value = unsafe { &mut *(value as *mut OrtValue) };
        Ok(value)
        // match value.as_mut() {
        //     None => anyhow::bail!("failed to get input"),
        //     Some(r) => Ok(r),
        // }
    }

    /// Get `OrtValue` for output with index `idx`.
    pub unsafe fn get_output<'s>(
        &self,
        ctx: &'s mut OrtKernelContext,
        idx: usize,
        shape: &[i64],
    ) -> Result<&'s mut OrtValue> {
        let fun = self.KernelContext_GetOutput.unwrap();

        let mut value: *mut OrtValue = std::ptr::null_mut();
        status_to_result(
            unsafe { fun(ctx, idx, shape.as_ptr(), shape.len(), &mut value) },
            self,
        )?;
        match unsafe { value.as_mut() } {
            None => anyhow::bail!("failed to get input"),
            Some(r) => Ok(r),
        }
    }

    pub fn get_tensor_data_mut<'s, T>(
        &self,
        value: &'s mut OrtValue,
        shape: &[usize],
    ) -> Result<ArrayViewMutD<'s, T>> {
        // This needs a refactor! The shape should be passed here,
        // rather than when creating the `Value`.
        let element_count = {
            let info = self.get_tensor_type_and_shape(value)?;
            info.get_tensor_shape_element_count()?
        };

        let fun = self.GetTensorMutableData.unwrap();
        let mut ptr: *mut _ = std::ptr::null_mut();
        let data = unsafe {
            fun(value, &mut ptr);
            std::slice::from_raw_parts_mut(ptr as *mut T, element_count)
        };
        let a = ArrayViewMut::from(data).into_shape(shape).unwrap();
        Ok(a)
    }

    fn get_tensor_type_and_shape<'s>(
        &'s self,
        value: &'s OrtValue,
    ) -> Result<TensorTypeAndShapeInfo<'s>> {
        let fun = self.GetTensorTypeAndShape.unwrap();

        let mut info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let ort_info = unsafe {
            status_to_result(fun(value, &mut info), self)?;
            info.as_mut().unwrap()
        };
        Ok(TensorTypeAndShapeInfo {
            api: self,
            info: ort_info,
        })
    }

    /// Total number of bytes of all concatenated strings (no trailing nulls!)
    fn get_string_tensor_data_length(&self, value: &OrtValue) -> Result<usize> {
        let mut non_null_bytes = 0;
        let fun_ptr = self.GetStringTensorDataLength.unwrap();
        status_to_result(unsafe { fun_ptr(value, &mut non_null_bytes) }, self)?;
        Ok(non_null_bytes)
    }

    fn get_string_tensor_data(&self, value: &OrtValue) -> Result<ArrayD<String>> {
        let fun_ptr = self.GetStringTensorContent.unwrap();

        let info = self.get_tensor_type_and_shape(value)?;
        let item_count = info.get_tensor_shape_element_count()?;
        let non_null_bytes = self.get_string_tensor_data_length(value)?;

        let mut buf = vec![0u8; non_null_bytes];
        let mut offsets = vec![0usize; item_count];
        unsafe {
            fun_ptr(
                value,
                buf.as_mut_ptr() as *mut _,
                non_null_bytes,
                offsets.as_mut_ptr() as *mut _,
                offsets.len(),
            );
        }

        // Compute windows with the start and end of each
        // substring and then scan the buffer.
        let very_end = [non_null_bytes];
        let starts = offsets.iter();
        let ends = offsets.iter().chain(very_end.iter()).skip(1);
        let windows = starts.zip(ends);
        let strings: Vec<_> = windows
            .scan(buf.as_slice(), |buf: &mut &[u8], (start, end)| {
                let (this, rest) = buf.split_at(end - start);
                *buf = rest;
                // The following allocation could be avoided
                String::from_utf8(this.to_vec()).ok()
            })
            .collect();

        let shape: Vec<_> = info
            .get_dimensions()?
            .into_iter()
            .map(|v| v as usize)
            .collect();

        Ok(Array::from(strings)
            .into_shape(shape)
            .expect("Shape information was incorrect."))
    }
}

impl<'info> KernelInfo<'info> {
    pub(crate) fn from_ort(api: &'static OrtApi, info: &'info OrtKernelInfo) -> Self {
        KernelInfo { api, info }
    }

    /// Read a `f32` attribute.
    pub fn get_attribute_f32(&self, name: &str) -> Result<f32> {
        let name = CString::new(name)?;
        let fun = self.api.KernelInfoGetAttribute_float.unwrap();
        let mut out = 0.0;
        status_to_result(unsafe { fun(self.info, name.as_ptr(), &mut out) }, self.api)?;
        Ok(out)
    }

    /// Read a `i64` attribute.
    pub fn get_attribute_i64(&self, name: &str) -> Result<i64> {
        let name = CString::new(name)?;
        let fun = self.api.KernelInfoGetAttribute_int64.unwrap();
        let mut out = 0;
        status_to_result(unsafe { fun(self.info, name.as_ptr(), &mut out) }, self.api)?;
        Ok(out)
    }

    /// Read a `String` attribute
    pub fn get_attribute_string(&self, name: &str) -> Result<String> {
        let name = CString::new(name)?;
        // Get size first
        let fun = self.api.KernelInfoGetAttribute_string.unwrap();
        let mut size = {
            let mut size = 0;
            unsafe {
                status_to_result(
                    fun(self.info, name.as_ptr(), std::ptr::null_mut(), &mut size),
                    self.api,
                )?;
                size
            }
        };

        let mut buf = vec![0u8; size as _];
        unsafe {
            status_to_result(
                fun(
                    self.info,
                    name.as_ptr(),
                    buf.as_mut_ptr() as *mut i8,
                    &mut size,
                ),
                self.api,
            )?
        };
        Ok(CString::from_vec_with_nul(buf)?.into_string()?)
    }

    pub fn get_attribute_f32s(&self, name: &str) -> Result<Vec<f32>> {
        let name = CString::new(name)?;
        // Get size first
        let fun = self.api.KernelInfoGetAttributeArray_float.unwrap();
        let mut size = {
            let mut size = 0;
            unsafe {
                status_to_result(
                    fun(self.info, name.as_ptr(), std::ptr::null_mut(), &mut size),
                    self.api,
                )?;
                size
            }
        };

        let mut buf = vec![0f32; size as _];
        unsafe {
            status_to_result(
                fun(self.info, name.as_ptr(), buf.as_mut_ptr(), &mut size),
                self.api,
            )?
        };
        Ok(buf)
    }

    pub fn get_attribute_i64s(&self, name: &str) -> Result<Vec<i64>> {
        let name = CString::new(name)?;
        // Get size first
        let fun = self.api.KernelInfoGetAttributeArray_int64.unwrap();
        let mut size = {
            let mut size = 0;
            unsafe {
                status_to_result(
                    fun(self.info, name.as_ptr(), std::ptr::null_mut(), &mut size),
                    self.api,
                )?;
                size
            }
        };

        let mut buf = vec![0i64; size as _];
        unsafe {
            status_to_result(
                fun(self.info, name.as_ptr(), buf.as_mut_ptr(), &mut size),
                self.api,
            )?
        };
        Ok(buf)
    }

    // Not implemented for string?
    #[allow(unused)]
    fn get_attribute_array<T>(&self, name: &str) -> Result<&[T]> {
        unimplemented!()
    }
}

impl<'s> TensorTypeAndShapeInfo<'s> {
    fn get_dimensions(&self) -> Result<Vec<i64>> {
        let mut n_dim = 0;
        unsafe { self.api.GetDimensionsCount.unwrap()(self.info, &mut n_dim) };
        let mut out = Vec::with_capacity(n_dim);
        unsafe {
            self.api.GetDimensions.unwrap()(self.info, out.as_mut_ptr(), n_dim);
            out.set_len(n_dim);
        }
        Ok(out)
    }

    fn get_tensor_shape_element_count(&self) -> Result<usize> {
        let mut element_count = 0;
        status_to_result(
            unsafe { self.api.GetTensorShapeElementCount.unwrap()(self.info, &mut element_count) },
            self.api,
        )?;
        Ok(element_count)
    }

    #[allow(unused)]
    fn get_tensor_element_type(&self) -> Result<ONNXTensorElementDataType> {
        unimplemented!()
    }
}

impl<'s> Drop for TensorTypeAndShapeInfo<'s> {
    fn drop(&mut self) {
        unsafe { self.api.ReleaseTensorTypeAndShapeInfo.unwrap()(&mut *self.info) }
    }
}

impl ElementType {
    pub fn to_ort_encoding(&self) -> u32 {
        match self {
            Self::Bool => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,

            Self::F32 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            Self::F64 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,

            Self::I32 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            Self::I64 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,

            Self::U8 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
            Self::U16 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
            Self::U32 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
            Self::U64 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,

            Self::String => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
        }
    }
}

fn add_op_to_domain(
    api: &OrtApi,
    domain: &mut OrtCustomOpDomain,
    op: &'static OrtCustomOp,
) -> OrtStatusPtr {
    let fun_ptr = api.CustomOpDomain_Add.unwrap();
    unsafe { fun_ptr(domain, op) }
}
