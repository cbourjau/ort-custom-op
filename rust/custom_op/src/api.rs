use crate::*;

#[derive(Debug)]
pub struct Api<'s> {
    api: &'s OrtApi
}

#[derive(Debug)]
pub struct KernelContext<'s> {
    context: &'s mut OrtKernelContext,
    api: &'s Api<'s>,
}

#[derive(Debug)]
pub struct KernelInfo<'s> {
    inner: &'s OrtKernelInfo,
}

// From context
#[derive(Debug)]
pub struct Value<'s> {
    value: &'s mut OrtValue,
    api: &'s Api<'s>,
}

// From Value
#[derive(Debug)]
pub struct TensorTypeAndShapeInfo<'s> {
    api: &'s Api<'s>,
    info: &'s OrtTensorTypeAndShapeInfo,
}

impl<'s> std::ops::Deref for Api<'s> {
    type Target = OrtApi;

    fn deref(&self) -> &Self::Target {
        &self.api
    }
}

impl<'s> Api<'s> {
    pub fn from_raw(api: *const OrtApi) -> Self {
	unsafe{Self {api: &*api}}
    }
}

impl<'s> KernelContext<'s> {
    pub fn from_raw(api: &'s Api<'s>, context: *mut OrtKernelContext) -> Self {
	dbg!(context);
	unsafe{Self {api, context: &mut *context}}
    }

    pub fn get_input(&self, index: u64) -> Result<Value<'s>> {
	let mut value: *const OrtValue = std::ptr::null();
        status_to_result(
	    unsafe{
		self.api.KernelContext_GetInput.unwrap()(self.context, index, &mut value)
	    }
	)?;
	if value.is_null() {
	    status_to_result(
		self.api.create_error_status(OrtErrorCode_ORT_FAIL, "Failed to get input")
	    )?;
	}
	let value = unsafe{dbg!(&mut *(value as *mut OrtValue))};
        Ok(Value{value, api: self.api})
    }

    pub fn get_output(&mut self, index: u64, dimensions: &[i64]) -> Result<Value<'s>> {
        let mut value: *mut OrtValue = std::ptr::null_mut();
        unsafe{
	    status_to_result(
		self.api.KernelContext_GetOutput.unwrap()(
		self.context,
		    index,
		    dimensions.as_ptr(),
		    dimensions.len() as u64,
		    &mut value
		)
	    )?;
	    Ok(Value{value: &mut *value, api: self.api})	
	}
    }
    
    fn get_input_count(&self) -> Result<usize> {
	unimplemented!()
    }
    fn get_output_count(&self) -> Result<usize> {
	unimplemented!()
    }
}

impl<'s> KernelInfo<'s> {
    fn get_attribute<T>(&self, name: &str) -> Result<&T> {
	unimplemented!()
    }

    // Not implemented for string?
    fn get_attribute_array<T>(&self, name: &str) -> Result<&[T]> {
	unimplemented!()
    }    
}

// Should be enum
type Type = ONNXType;

impl<'s> Value<'s> {
    pub fn get_tensor_type_and_shape(&self) -> Result<TensorTypeAndShapeInfo<'s>> {
	let mut info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        unsafe{
	    self.api.GetTensorTypeAndShape.unwrap()(self.value, &mut info)
	};
	// TODO: ReleaseTensorTypeAndShapeInfo ought to be called somewhere!
        Ok(TensorTypeAndShapeInfo{api: self.api, info: unsafe{&*info}})
    }

    pub fn get_value_type(&self) -> Result<Type> {
	unimplemented!()
    }

    pub fn get_tensor_mutable_data<T>(&mut self) -> Result<&'s mut [T]> {
        let element_count = {
	    let info = self.get_tensor_type_and_shape()?;
	    info.get_tensor_shape_element_count()?
	};
        let mut ptr: *mut _ = std::ptr::null_mut();
        unsafe{
	    self.api.GetTensorMutableData.unwrap()(self.value, &mut ptr);
	    Ok(std::slice::from_raw_parts_mut(ptr as *mut T, element_count as usize))
	}
    }

    // /// Not clear what that is...
    // fn get_type_info(&self) -> Result<TypeInfo> {
    // 	unimplemented!()
    // }
}

impl<'s> TensorTypeAndShapeInfo<'s> {
    pub fn get_dimensions(&self) -> Result<Vec<i64>> {
        let mut n_dim = 0;
        unsafe{
	    self.api.GetDimensionsCount.unwrap()(self.info, &mut n_dim)
	};
	let mut out = Vec::with_capacity(n_dim as usize);
        unsafe{
	    self.api.GetDimensions.unwrap()(self.info, out.as_mut_ptr(), n_dim);
	    out.set_len(n_dim as usize);
	}
	Ok(out)
    }

    fn get_tensor_shape_element_count(&self) -> Result<u64> {
	let mut element_count = 0;
	status_to_result(
	    unsafe {
		self.api.GetTensorShapeElementCount.unwrap()(self.info, &mut element_count)
	    }
	)?;
	Ok(element_count)
    }

    fn get_tensor_element_type(&self) -> Result<ONNXTensorElementDataType> {
	unimplemented!()
    }
    //...
}

impl<'s> Drop for TensorTypeAndShapeInfo<'s> {
    fn drop(&mut self) {
    }
}

impl<'s> Drop for Value<'s> {
    fn drop(&mut self) {
    }
}

