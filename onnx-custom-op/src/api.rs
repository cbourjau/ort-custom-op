use std::ffi::{c_char, CString};

use crate::bindings::*;
use crate::error::ErrorStatusPtr;

use anyhow::Result;
use ndarray::{Array, ArrayD, ArrayView, ArrayViewD, ArrayViewMut, ArrayViewMutD};

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
        index: u64,
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
        index: u64,
    ) -> Result<ArrayD<String>> {
        let value = self.get_input(context, index)?;
        self.get_string_tensor_data(value)
    }
}

impl OrtApi {
    /// Get `OrtValue` for input with index `idx`.
    fn get_input<'s>(&self, ctx: &'s OrtKernelContext, idx: u64) -> Result<&'s mut OrtValue> {
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
        idx: u64,
        shape: &[i64],
    ) -> Result<&'s mut OrtValue> {
        let fun = self.KernelContext_GetOutput.unwrap();

        let mut value: *mut OrtValue = std::ptr::null_mut();
        status_to_result(
            unsafe { fun(ctx, idx, shape.as_ptr(), shape.len() as u64, &mut value) },
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
            std::slice::from_raw_parts_mut(ptr as *mut T, element_count as usize)
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
    fn get_string_tensor_data_length(&self, value: &OrtValue) -> Result<u64> {
        let mut non_null_bytes: u64 = 0;
        let fun_ptr = self.GetStringTensorDataLength.unwrap();
        status_to_result(unsafe { fun_ptr(value, &mut non_null_bytes) }, self)?;
        Ok(non_null_bytes)
    }

    fn get_string_tensor_data(&self, value: &OrtValue) -> Result<ArrayD<String>> {
        let fun_ptr = self.GetStringTensorContent.unwrap();

        let info = self.get_tensor_type_and_shape(value)?;
        let item_count = info.get_tensor_shape_element_count()?;
        let non_null_bytes = self.get_string_tensor_data_length(value)?;

        let mut buf = vec![0u8; non_null_bytes as usize];
        let mut offsets = vec![0usize; item_count as usize];
        unsafe {
            fun_ptr(
                value,
                buf.as_mut_ptr() as *mut _,
                non_null_bytes,
                offsets.as_mut_ptr() as *mut _,
                offsets.len() as u64,
            );
        }

        // Compute windows with the start and end of each
        // substring and then scan the buffer.
        let very_end = [non_null_bytes as usize];
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
    pub fn from_ort(api: &'static OrtApi, info: &'info OrtKernelInfo) -> Self {
        KernelInfo { api, info }
    }

    pub fn get_attribute_string(&self, name: &str) -> Result<String> {
        let name = CString::new(name)?;
        // Get size first
        let mut size = {
            let mut size = 0;
            // let buf: *mut _ = std::ptr::null_mut();
            unsafe {
                status_to_result(
                    self.api.KernelInfoGetAttribute_string.unwrap()(
                        self.info,
                        name.as_ptr(),
                        std::ptr::null_mut(),
                        &mut size,
                    ),
                    self.api,
                )?;
                size
            }
        };

        let mut buf = vec![0u8; size as _];
        unsafe {
            status_to_result(
                self.api.KernelInfoGetAttribute_string.unwrap()(
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

    // Not implemented for string?
    #[allow(unused)]
    fn get_attribute_array<T>(&self, name: &str) -> Result<&[T]> {
        unimplemented!()
    }
}

impl<'s> TensorTypeAndShapeInfo<'s> {
    pub fn get_dimensions(&self) -> Result<Vec<i64>> {
        let mut n_dim = 0;
        unsafe { self.api.GetDimensionsCount.unwrap()(self.info, &mut n_dim) };
        let mut out = Vec::with_capacity(n_dim as usize);
        unsafe {
            self.api.GetDimensions.unwrap()(self.info, out.as_mut_ptr(), n_dim);
            out.set_len(n_dim as usize);
        }
        Ok(out)
    }

    fn get_tensor_shape_element_count(&self) -> Result<u64> {
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
    // TODO: I should actually use this drop impl...
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

/// Create a new custom domain with the operators `ops`.
pub fn create_custom_op_domain(
    session_options: &mut OrtSessionOptions,
    api_base: &mut OrtApiBase,
    domain: &str,
    ops: &[&'static OrtCustomOp],
) -> Result<(), ErrorStatusPtr> {
    let api = unsafe { api_base.GetApi.unwrap()(12).as_ref().unwrap() };

    let fun_ptr = api.CreateCustomOpDomain.unwrap();
    let mut domain_ptr: *mut OrtCustomOpDomain = std::ptr::null_mut();

    // Copies and leaks!
    let c_op_domain = str_to_c_char_ptr(domain);
    let domain = unsafe {
        // According to docs: "Must be freed with OrtApi::ReleaseCustomOpDomain"
        status_to_result(fun_ptr(c_op_domain, &mut domain_ptr), api)?;
        status_to_result(
            api.AddCustomOpDomain.unwrap()(session_options, domain_ptr),
            api,
        )?;
        domain_ptr.as_mut().unwrap()
    };
    for op in ops {
        add_op_to_domain(api, domain, op)?;
    }
    Ok(())
}

fn add_op_to_domain(
    api: &OrtApi,
    domain: &mut OrtCustomOpDomain,
    op: &'static OrtCustomOp,
) -> Result<(), ErrorStatusPtr> {
    let fun_ptr = api.CustomOpDomain_Add.unwrap();
    status_to_result(unsafe { fun_ptr(domain, op) }, api)?;
    Ok(())
}

fn str_to_c_char_ptr(s: &str) -> *const c_char {
    CString::new(s).unwrap().into_raw()
}

/// Wraps a status pointer into a result.
///
///A null pointer is mapped to the `Ok(())`.
fn status_to_result(ptr: OrtStatusPtr, api: &OrtApi) -> Result<(), ErrorStatusPtr> {
    if ptr.is_null() {
        Ok(())
    } else {
        Err(ErrorStatusPtr::new(ptr, api))
    }
}
