use crate::{
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, OrtApiBase,
};

use crate::*;
use ndarray::{ArrayView, ArrayViewMut, IxDyn};

#[derive(Debug)]
pub struct Api {
    api: &'static OrtApi,
}

#[derive(Debug)]
pub struct KernelContext<'s> {
    context: &'s mut OrtKernelContext,
    api: &'s Api,
}

#[derive(Debug)]
pub struct KernelInfo<'s> {
    _kernel_info: &'s OrtKernelInfo,
}

// From context
#[derive(Debug)]
pub struct Value<'s> {
    value: &'s mut OrtValue,
    api: &'s Api,
}

#[derive(Debug)]
pub struct OutputValue<'s> {
    api: Api,
    context: &'s mut OrtKernelContext,
    index: u64,
}

// From Value
#[derive(Debug)]
pub struct TensorTypeAndShapeInfo<'s> {
    api: &'s Api,
    info: &'s mut OrtTensorTypeAndShapeInfo,
}

pub struct CustomOpDomain<'s> {
    api: &'s OrtApi,
    custom_op_domain: &'s mut OrtCustomOpDomain,
}

pub enum ElementType {
    F32,
    F64,
    I32,
    I64,
    String,
}

pub struct SessionOptions<'s> {
    api: &'static OrtApi,
    session_options: &'s mut OrtSessionOptions,
}

impl std::ops::Deref for Api {
    type Target = OrtApi;

    fn deref(&self) -> &Self::Target {
        self.api
    }
}

impl Api {
    pub fn from_raw(api: &'static OrtApi) -> Self {
        Self { api }
    }

    fn create_error_status(&self, code: u32, msg: &str) -> *mut OrtStatus {
        let c_char_ptr = str_to_c_char_ptr(msg);
        unsafe { self.CreateStatus.unwrap()(code, c_char_ptr) }
    }
}

impl<'s> KernelContext<'s> {
    pub fn from_raw(api: &'s Api, context: *mut OrtKernelContext) -> Self {
        unsafe {
            Self {
                api,
                context: &mut *context,
            }
        }
    }

    pub fn get_input_value(&self, index: u64) -> Result<Value<'s>> {
        let mut value: *const OrtValue = std::ptr::null();
        status_to_result(unsafe {
            self.api.KernelContext_GetInput.unwrap()(self.context, index, &mut value)
        })?;
        if value.is_null() {
            status_to_result(
                self.api
                    .create_error_status(OrtErrorCode_ORT_FAIL, "Failed to get input"),
            )?;
        }
        let value = unsafe { &mut *(value as *mut OrtValue) };
        Ok(Value {
            value,
            api: self.api,
        })
    }

    // pub fn get_input<T>(&self, index: u64) -> Result<ArrayView<T, IxDyn>> {
    //     let mut value = self.get_input_value(index)?;
    //     value.get_tensor_data::<T>()
    // }

    pub unsafe fn get_safe_output<'inner>(self, index: u64) -> OutputValue<'s> {
        OutputValue::<'s> {
            api: Api::from_raw(self.api.api),
            context: self.context,
            index,
        }
    }

    pub fn get_output_count(&self) -> Result<u64> {
        let mut val = 0;
        status_to_result(unsafe {
            self.api.KernelContext_GetOutputCount.unwrap()(self.context, &mut val)
        })?;
        Ok(val)
    }

    #[allow(unused)]
    pub fn get_input_count(&self) -> Result<u64> {
        let mut val = 0;
        status_to_result(unsafe {
            self.api.KernelContext_GetInputCount.unwrap()(self.context, &mut val)
        })?;
        Ok(val)
    }
}

impl<'s> KernelInfo<'s> {
    #[allow(unused)]
    fn get_attribute<T>(&self, name: &str) -> Result<&T> {
        unimplemented!()
    }

    // Not implemented for string?
    #[allow(unused)]
    fn get_attribute_array<T>(&self, name: &str) -> Result<&[T]> {
        unimplemented!()
    }
}

// Should be enum
#[allow(unused)]
type Type = ONNXType;

impl<'s> Value<'s> {
    pub fn get_tensor_data<T>(self) -> Result<ArrayView<'s, T, IxDyn>> {
        // Get the data
        let data = {
            let element_count = self
                .get_tensor_type_and_shape()?
                .get_tensor_shape_element_count()?;

            let mut ptr: *mut _ = std::ptr::null_mut();
            unsafe {
                self.api.GetTensorMutableData.unwrap()(self.value, &mut ptr);
                std::slice::from_raw_parts(ptr as *mut T, element_count as usize)
            }
        };
        // Figure out the correct shape
        let dims: Vec<_> = {
            let info = self.get_tensor_type_and_shape()?;
            info.get_dimensions()?
                .into_iter()
                .map(|el| el as usize)
                .collect()
        };
        Ok(ArrayView::from(data)
            .into_shape(dims)
            .expect("Shape information was incorrect."))
    }

    pub fn get_tensor_type_and_shape(&self) -> Result<TensorTypeAndShapeInfo<'s>> {
        let mut info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        unsafe {
            self.api.GetTensorTypeAndShape.unwrap()(self.value, &mut info);
            Ok(TensorTypeAndShapeInfo {
                api: self.api,
                info: &mut *info,
            })
        }
    }

    #[allow(unused)]
    pub fn get_value_type(&self) -> Result<Type> {
        unimplemented!()
    }

    // /// Not clear what that is...
    // fn get_type_info(&self) -> Result<TypeInfo> {
    // 	unimplemented!()
    // }
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
        status_to_result(unsafe {
            self.api.GetTensorShapeElementCount.unwrap()(self.info, &mut element_count)
        })?;
        Ok(element_count)
    }

    #[allow(unused)]
    fn get_tensor_element_type(&self) -> Result<ONNXTensorElementDataType> {
        unimplemented!()
    }
    //...
}

impl<'s> Drop for TensorTypeAndShapeInfo<'s> {
    fn drop(&mut self) {
        unsafe { self.api.ReleaseTensorTypeAndShapeInfo.unwrap()(&mut *self.info) }
    }
}

impl<'s> Drop for Value<'s> {
    fn drop(&mut self) {
        // This is unused in the official example and crashes if used...
    }
}

impl<'s> CustomOpDomain<'s> {
    pub fn add_op_to_domain(&mut self, op: &'static OrtCustomOp) -> Result<()> {
        let fun_ptr = self.api.CustomOpDomain_Add.unwrap();
        status_to_result(unsafe { fun_ptr(self.custom_op_domain, op) })?;
        Ok(())
    }
}

impl<'s> OutputValue<'s> {
    pub fn get_tensor_data_mut<T>(self, shape: &[usize]) -> Result<ArrayViewMut<T, IxDyn>> {
        let value = unsafe {
            let mut value: *mut OrtValue = std::ptr::null_mut();
            let shape: Vec<_> = shape.iter().map(|el| *el as i64).collect();
            status_to_result(self.api.KernelContext_GetOutput.unwrap()(
                self.context,
                self.index,
                shape.as_ptr(),
                shape.len() as u64,
                &mut value,
            ))?;
            if value.is_null() {
                return Err(self.api.create_error_status(0, "No value found"));
            }
            Value {
                value: &mut *value,
                api: &self.api,
            }
        };
        // This needs a refactor! The shape should be passed here,
        // rather than when creating the `Value`.
        let element_count = value
            .get_tensor_type_and_shape()?
            .get_tensor_shape_element_count()?;

        let mut ptr: *mut _ = std::ptr::null_mut();
        let data = unsafe {
            self.api.GetTensorMutableData.unwrap()(value.value, &mut ptr);
            std::slice::from_raw_parts_mut(ptr as *mut T, element_count as usize)
        };

        let a = ArrayViewMut::from(data).into_shape(shape).unwrap();
        Ok(a)
    }
}

impl ElementType {
    pub fn to_ort_encoding(&self) -> u32 {
        match self {
            Self::F32 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            Self::F64 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
            Self::I32 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            Self::I64 => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            Self::String => ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
        }
    }
}

impl<'s> SessionOptions<'s> {
    pub fn from_ort(api_base: &mut OrtApiBase, options: &'s mut OrtSessionOptions) -> Self {
        // Version 12 is the latest one supported by the installed
        // onnxruntime. I should probably downgrade the c api file.
        let api = unsafe { api_base.GetApi.unwrap()(12) };
        Self {
            api: unsafe { &*api },
            session_options: options,
        }
    }

    pub fn create_custom_op_domain(&mut self, domain: &str) -> Result<CustomOpDomain> {
        let fun_ptr = self.api.CreateCustomOpDomain.unwrap();
        let mut domain_ptr: *mut OrtCustomOpDomain = std::ptr::null_mut();

        // Leak!
        let c_op_domain = str_to_c_char_ptr(domain);
        unsafe {
            status_to_result(fun_ptr(c_op_domain, &mut domain_ptr))?;
            status_to_result(self.api.AddCustomOpDomain.unwrap()(
                self.session_options,
                domain_ptr,
            ))?;
            Ok(CustomOpDomain {
                api: self.api,
                custom_op_domain: &mut *domain_ptr,
            })
        }
    }
}
