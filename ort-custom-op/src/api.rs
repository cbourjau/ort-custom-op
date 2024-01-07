use std::ffi::CString;

use anyhow::{bail, Result};
use ndarray::{Array, ArrayD, ArrayView, ArrayViewD, ArrayViewMut, ArrayViewMutD};

use crate::bindings::*;
use crate::error::ErrorStatus;
use crate::value::{TensorString, Value};

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

macro_rules! bail_non_null {
    ($ptr:expr) => {
        if !$ptr.is_null() {
            return $ptr;
        }
    };
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
        // Create domain
        // According to docs: "Must be freed with OrtApi::ReleaseCustomOpDomain"
        bail_non_null!(fun_ptr(c_op_domain, &mut domain_ptr));
        domain_ptr.as_mut().unwrap()
    };
    // Add ops to domain
    for op in ops {
        bail_non_null!(add_op_to_domain(api, domain, op));
    }
    // Add domain to session options
    unsafe { api.AddCustomOpDomain.unwrap()(session_options, domain_ptr) }
}

/// Explicit struct around OrtTypeAndShapeInfo pointer since we are
/// responsible for properly dropping it.
#[derive(Debug)]
struct TensorTypeAndShapeInfo<'s> {
    api: &'s OrtApi,
    // must be mut so that we can later drop it
    info: &'s mut OrtTensorTypeAndShapeInfo,
}

impl OrtValue {
    fn shape(&self, api: &OrtApi) -> Result<Vec<usize>> {
        let shape: Vec<_> = self
            .get_tensor_type_and_shape(api)?
            .get_dimensions()?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        Ok(shape)
    }

    /// Get the ONNX type such as 'Tensor' or 'Map'.
    fn onnx_type(&self, api: &OrtApi) -> Result<ONNXType> {
        let fun = api.GetValueType.unwrap();
        let mut num = 0;
        let num = unsafe {
            api.status_to_result(fun(self, &mut num))?;
            num
        };

        Ok(num)
    }

    pub fn load_value<'s>(&'s mut self, api: &OrtApi) -> Result<Value<'s>> {
        Ok(match self.onnx_type(&api)? {
            ONNXType_ONNX_TYPE_TENSOR => {
                let (ty, shape) = {
                    let info = self.get_tensor_type_and_shape(&api)?;
                    (info.get_element_type()?, info.get_dimensions()?)
                };
                match ty {
                    ElementType::I32 => Value::TensorI32(self.as_array(api)?),
                    ElementType::String => {
                        let (buf, offsets) = self.get_string_tensor_single_buf(api)?;
                        Value::TensorString(TensorString {
                            buf,
                            shape: shape.into_iter().map(|n| n as usize).collect(),
                            offsets,
                        })
                    }
                    _ => panic!(),
                }
            }
            _ => panic!(),
        })
    }

    pub fn as_array<T>(&mut self, api: &OrtApi) -> Result<ArrayViewD<T>> {
        let shape = self.shape(api)?;
        let data = self.get_data_mut(api)?;
        Ok(ArrayView::from(data as &_).into_shape(shape.as_slice())?)
    }

    pub fn as_array_mut<T>(&mut self, api: &OrtApi) -> Result<ArrayViewMutD<T>> {
        let shape = self.shape(api)?;
        let data = self.get_data_mut(api)?;
        Ok(ArrayViewMut::from(data).into_shape(shape.as_slice())?)
    }

    fn get_data_mut<'s, T>(&'s mut self, api: &OrtApi) -> Result<&'s mut [T]> {
        if self.onnx_type(api)? != ONNXType_ONNX_TYPE_TENSOR {
            bail!("OrtValue is not a tensor")
        }
        let element_count = {
            let info = self.get_tensor_type_and_shape(api)?;
            info.get_tensor_shape_element_count()?
        };

        let fun = api.GetTensorMutableData.unwrap();
        let mut ptr: *mut _ = std::ptr::null_mut();
        let data = unsafe {
            fun(self, &mut ptr);
            std::slice::from_raw_parts_mut(ptr as *mut T, element_count)
        };
        Ok(data)
    }

    fn get_tensor_type_and_shape<'s>(
        &'s self,
        api: &'s OrtApi,
    ) -> Result<TensorTypeAndShapeInfo<'s>> {
        let fun = api.GetTensorTypeAndShape.unwrap();

        let mut info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let ort_info = unsafe {
            api.status_to_result(fun(self, &mut info))?;
            info.as_mut().unwrap()
        };
        Ok(TensorTypeAndShapeInfo {
            api,
            info: ort_info,
        })
    }

    /// Total number of bytes of all concatenated strings (no trailing nulls!)
    fn get_string_tensor_data_length(&self, api: &OrtApi) -> Result<usize> {
        let mut non_null_bytes = 0;
        let fun_ptr = api.GetStringTensorDataLength.unwrap();
        api.status_to_result(unsafe { fun_ptr(self, &mut non_null_bytes) })?;
        Ok(non_null_bytes)
    }

    /// Get string data as a single buffer along with the offsets to
    /// the beginning of each element.
    fn get_string_tensor_single_buf(&self, api: &OrtApi) -> Result<(Vec<u8>, Vec<usize>)> {
        let fun_ptr = api.GetStringTensorContent.unwrap();

        let info = self.get_tensor_type_and_shape(api)?;
        let item_count = info.get_tensor_shape_element_count()?;
        let non_null_bytes = self.get_string_tensor_data_length(api)?;

        let mut buf = vec![0u8; non_null_bytes];
        let mut offsets = vec![0usize; item_count];
        api.status_to_result(unsafe {
            fun_ptr(
                self,
                buf.as_mut_ptr() as *mut _,
                non_null_bytes,
                offsets.as_mut_ptr() as *mut _,
                offsets.len(),
            )
        })?;
        Ok((buf, offsets))
    }

    fn get_string_tensor_data(&self, api: &OrtApi) -> Result<ArrayD<String>> {
        let fun_ptr = api.GetStringTensorContent.unwrap();

        let info = self.get_tensor_type_and_shape(api)?;
        let item_count = info.get_tensor_shape_element_count()?;
        let non_null_bytes = self.get_string_tensor_data_length(api)?;

        let mut buf = vec![0u8; non_null_bytes];
        let mut offsets = vec![0usize; item_count];
        unsafe {
            fun_ptr(
                self,
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

impl OrtKernelContext {
    pub(crate) fn get_input_values<'s>(&'s self, api: &OrtApi) -> Result<Vec<Value<'s>>> {
        let n_inputs = self.get_input_count(api)?;
        let mut inputs = Vec::with_capacity(n_inputs);
        for idx in 0..n_inputs {
            let value = self.get_input(api, idx)?;
            inputs.push(value.load_value(api)?);
        }
        Ok(inputs)
    }

    pub(crate) fn get_input_array<'s, T>(
        &'s self,
        api: &OrtApi,
        index: usize,
    ) -> Result<ArrayViewD<'s, T>> {
        let value = self.get_input(api, index)?;
        value.as_array(api)
    }

    pub(crate) fn get_input_array_string(
        &self,
        api: &OrtApi,
        index: usize,
    ) -> Result<ArrayD<String>> {
        let value = self.get_input(api, index)?;
        value.get_string_tensor_data(api)
    }

    pub(crate) fn fill_string_tensor(
        &mut self,
        api: &OrtApi,
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
        let val = unsafe { self.get_output(api, index, &shape_i64) }?;

        let fun = api.FillStringTensor.unwrap();
        api.status_to_result(unsafe { fun(val, ptr_of_ptrs, n_items) })?;
        Ok(())
    }

    pub(crate) fn get_input_count(&self, api: &OrtApi) -> Result<usize> {
        let fun = api.KernelContext_GetInputCount.unwrap();
        let mut out: usize = 0;
        api.status_to_result(unsafe { fun(self, &mut out) })?;
        Ok(out)
    }

    /// Get `OrtValue` for output with index `idx`.
    pub(crate) unsafe fn get_output<'s>(
        &'s mut self,
        api: &OrtApi,
        idx: usize,
        shape: &[i64],
    ) -> Result<&'s mut OrtValue> {
        let fun = api.KernelContext_GetOutput.unwrap();

        let mut value: *mut OrtValue = std::ptr::null_mut();
        api.status_to_result(unsafe { fun(self, idx, shape.as_ptr(), shape.len(), &mut value) })?;
        match unsafe { value.as_mut() } {
            None => anyhow::bail!("failed to get input"),
            Some(r) => Ok(r),
        }
    }

    /// Get `OrtValue` for input with index `idx`.
    fn get_input<'s>(&'s self, api: &OrtApi, idx: usize) -> Result<&'s mut OrtValue> {
        let fun = api.KernelContext_GetInput.unwrap();

        let mut value: *const OrtValue = std::ptr::null();
        api.status_to_result(unsafe { fun(self, idx, &mut (value)) })?;

        // Code crime!
        let value = unsafe { &mut *(value as *mut OrtValue) };
        Ok(value)
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
        self.api
            .status_to_result(unsafe { fun(self.info, name.as_ptr(), &mut out) })?;
        Ok(out)
    }

    /// Read a `i64` attribute.
    pub fn get_attribute_i64(&self, name: &str) -> Result<i64> {
        let name = CString::new(name)?;
        let fun = self.api.KernelInfoGetAttribute_int64.unwrap();
        let mut out = 0;
        self.api
            .status_to_result(unsafe { fun(self.info, name.as_ptr(), &mut out) })?;
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
                self.api.status_to_result(fun(
                    self.info,
                    name.as_ptr(),
                    std::ptr::null_mut(),
                    &mut size,
                ))?;
                size
            }
        };

        let mut buf = vec![0u8; size as _];
        unsafe {
            self.api.status_to_result(fun(
                self.info,
                name.as_ptr(),
                buf.as_mut_ptr() as *mut i8,
                &mut size,
            ))?
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
                self.api.status_to_result(fun(
                    self.info,
                    name.as_ptr(),
                    std::ptr::null_mut(),
                    &mut size,
                ))?;
                size
            }
        };

        let mut buf = vec![0f32; size as _];
        unsafe {
            self.api
                .status_to_result(fun(self.info, name.as_ptr(), buf.as_mut_ptr(), &mut size))?
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
                self.api.status_to_result(fun(
                    self.info,
                    name.as_ptr(),
                    std::ptr::null_mut(),
                    &mut size,
                ))?;
                size
            }
        };

        let mut buf = vec![0i64; size as _];
        unsafe {
            self.api
                .status_to_result(fun(self.info, name.as_ptr(), buf.as_mut_ptr(), &mut size))?
        };
        Ok(buf)
    }

    pub fn get_attribute_tensor<'s, T>(&'s self, name: &str) -> Result<ArrayViewD<'s, T>> {
        let get_alloc = self.api.GetAllocatorWithDefaultOptions.unwrap();
        let mut alloc = std::ptr::null_mut();

        unsafe { self.api.status_to_result(get_alloc(&mut alloc))? };

        let name = CString::new(name)?;
        let fun = self.api.KernelInfoGetAttribute_tensor.unwrap();
        let value = {
            let mut value = std::ptr::null_mut();
            unsafe {
                self.api
                    .status_to_result(fun(self.info, name.as_ptr(), alloc, &mut value))?;
                &mut *value
            }
        };

        value.as_array(self.api)
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
        self.api.status_to_result(unsafe {
            self.api.GetTensorShapeElementCount.unwrap()(self.info, &mut element_count)
        })?;
        Ok(element_count)
    }

    #[allow(unused)]
    fn get_element_type(&self) -> Result<ElementType> {
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

    pub fn try_from_ort_encoding(type_number: u32) -> Result<Self> {
        #[allow(non_upper_case_globals)]
        Ok(match type_number {
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => Self::Bool,

            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => Self::F32,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => Self::F64,

            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => Self::I32,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => Self::I64,

            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => Self::U8,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => Self::U16,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => Self::U32,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => Self::U64,

            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => Self::String,
            _ => bail!("Unsupported tensor element type: '{}'", type_number),
        })
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

impl OrtApi {
    /// Wraps a status pointer into a result.
    ///
    ///A null pointer is mapped to the `Ok(())`.
    fn status_to_result(&self, ptr: OrtStatusPtr) -> Result<(), ErrorStatus> {
        if ptr.is_null() {
            Ok(())
        } else {
            Err(ErrorStatus::new(ptr, self))
        }
    }
}
