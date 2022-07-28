use crate::*;

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

// From Value
#[derive(Debug)]
pub struct TensorTypeAndShapeInfo<'s> {
    api: &'s Api,
    info: &'s mut OrtTensorTypeAndShapeInfo,
}

pub struct CustomOpDomain<'s> {
    api: &'s Api,
    custom_op_domain: &'s mut OrtCustomOpDomain,
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

    pub fn create_custom_op_domain(
        &self,
        domain: &str,
        options: &mut OrtSessionOptions,
    ) -> Result<CustomOpDomain> {
        let fun_ptr = self.CreateCustomOpDomain.unwrap();
        let mut domain_ptr: *mut OrtCustomOpDomain = std::ptr::null_mut();

        // Leak!
        let c_op_domain = str_to_c_char_ptr(domain);
        unsafe {
            status_to_result(fun_ptr(c_op_domain, &mut domain_ptr))?;
            status_to_result(self.AddCustomOpDomain.unwrap()(options, domain_ptr))?;
            Ok(CustomOpDomain {
                api: self,
                custom_op_domain: &mut *domain_ptr,
            })
        }
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

    pub fn get_input(&self, index: u64) -> Result<Value<'s>> {
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

    pub fn get_output(&mut self, index: u64, dimensions: &[i64]) -> Result<Value<'s>> {
        let mut value: *mut OrtValue = std::ptr::null_mut();
        unsafe {
            status_to_result(self.api.KernelContext_GetOutput.unwrap()(
                self.context,
                index,
                dimensions.as_ptr(),
                dimensions.len() as u64,
                &mut value,
            ))?;
            Ok(Value {
                value: &mut *value,
                api: self.api,
            })
        }
    }

    #[allow(unused)]
    fn get_input_count(&self) -> Result<usize> {
        unimplemented!()
    }

    #[allow(unused)]
    fn get_output_count(&self) -> Result<usize> {
        unimplemented!()
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

    /// Get mutable tensor data associated to this `Value`.
    ///
    /// Calling this function repeatedly hands out aliasing mutable slices!
    pub unsafe fn get_tensor_data_mut<T>(&mut self) -> Result<&'s mut [T]> {
        let element_count = self
            .get_tensor_type_and_shape()?
            .get_tensor_shape_element_count()?;

        let mut ptr: *mut _ = std::ptr::null_mut();
        self.api.GetTensorMutableData.unwrap()(self.value, &mut ptr);
        Ok(std::slice::from_raw_parts_mut(
            ptr as *mut T,
            element_count as usize,
        ))
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
