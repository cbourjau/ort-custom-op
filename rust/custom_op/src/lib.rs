#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::ffi::CString;
use std::os::raw::{c_char, c_void};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

mod api;

type Result<T> = std::result::Result<T, OrtStatusPtr>;

fn str_to_c_char_ptr(s: &str) -> *const c_char {
    CString::new(s).unwrap().into_raw()
}

/// Wraps a status pointer into a result.
///
///A null pointer is mapped to the `Ok(())`.
fn status_to_result(ptr: OrtStatusPtr) -> Result<()> {
    if ptr == std::ptr::null_mut() {
	Ok(())
    } else {
	Err(ptr)
    }
}

impl OrtApi {
    fn create_error_status(&self, code: u32, msg: &'static str) -> *mut OrtStatus {
        let c_char_ptr = str_to_c_char_ptr(msg);
        unsafe { self.CreateStatus.unwrap()(code, c_char_ptr) }
    }

    fn create_custom_op_domain(
        &self,
        domain: &str,
        options: &mut OrtSessionOptions,
    ) -> *mut OrtCustomOpDomain {
        let fun_ptr = self.CreateCustomOpDomain.unwrap();
        let mut domain_ptr: *mut OrtCustomOpDomain = std::ptr::null_mut();
        let c_op_domain = str_to_c_char_ptr(domain);
        let status = unsafe { fun_ptr(c_op_domain, &mut domain_ptr) };
        if status != std::ptr::null_mut() {
            panic!()
            // return status;
        }
        // Somehow, one must ensure that the domain stays alive?
        // ...
        dbg!(domain_ptr);
        dbg!(unsafe { self.AddCustomOpDomain.unwrap()(options, domain_ptr) });
        domain_ptr
    }

    fn add_op_to_domain(&self, domain: *mut OrtCustomOpDomain) -> Result<()> {
        let fun_ptr = self.CustomOpDomain_Add.unwrap();
        status_to_result(unsafe { fun_ptr(domain, &op_one::OP_ONE) })?;
        status_to_result(unsafe { fun_ptr(domain, &op_two::OP_TWO) })
    }
}

impl OrtKernelContext {
    unsafe fn get_input(&self, api: &OrtApi, index: u64) -> Result<*mut OrtValue> {
        let mut value: *const OrtValue = std::ptr::null();
        status_to_result(api.KernelContext_GetInput.unwrap()(self, index, &mut value))?;
	if value == std::ptr::null() {
	    status_to_result(
		api.create_error_status(OrtErrorCode_ORT_FAIL, "Failed to get input")
	    )?;
	}
        Ok(value as *mut _)
    }

    unsafe fn get_output(
        &mut self,
        api: &OrtApi,
        index: u64,
        shape_like: &OrtTensorTypeAndShapeInfo,
    ) -> *mut OrtValue {
        let dims = {
            let mut n_dim = 0;
            api.GetDimensionsCount.unwrap()(shape_like, &mut n_dim);
	    let mut out = Vec::with_capacity(n_dim as usize);
            dbg!(api.GetDimensions.unwrap()(shape_like, out.as_mut_ptr(), n_dim));
	    out.set_len(n_dim as usize);
            dbg!(out)
        };

        let mut value: *mut OrtValue = std::ptr::null_mut();
        let _status = dbg!(api.KernelContext_GetOutput.unwrap()(
            self,
            index,
            dims.as_ptr(),
            dims.len() as u64,
            &mut value
        ));
        dbg!(value) as *mut _
    }
}

impl OrtValue {
    unsafe fn get_type_and_shape_info(&self, api: &OrtApi) -> *mut OrtTensorTypeAndShapeInfo {
        let mut info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        api.GetTensorTypeAndShape.unwrap()(self as *const _, &mut info);
	// TODO: ReleaseTensorTypeAndShapeInfo ought to be called somewhere!
        dbg!(info)
    }

    unsafe fn get_tensor_mutable_data<T>(&mut self, api: &OrtApi) -> &mut [T] {
        let info = self.get_type_and_shape_info(api);
        let mut element_count = 0;
        api.GetTensorShapeElementCount.unwrap()(info, &mut element_count);

        let mut ptr: *mut _ = std::ptr::null_mut();
        api.GetTensorMutableData.unwrap()(self, &mut ptr);

        std::slice::from_raw_parts_mut(ptr as *mut T, element_count as usize)
    }
}

#[derive(Debug)]
struct Kernel {
    op: *const OrtCustomOp,
    api: *const OrtApi,
    info: *const OrtKernelInfo,
}

unsafe extern "C" fn create_kernel(
    op: *const OrtCustomOp,
    api: *const OrtApi,
    info: *const OrtKernelInfo,
) -> *mut c_void {
    let kernel = Kernel { op, api, info };
    Box::leak(Box::new(kernel)) as *mut _ as *mut c_void
}

unsafe extern "C" fn get_execution_provider_type(
    _op: *const OrtCustomOp,
) -> *const ::std::os::raw::c_char {
    dbg!("provider");
    CString::new("CPUExecutionProvider").unwrap().into_raw()
}


mod op_one {
    use super::*;

    pub const OP_ONE: OrtCustomOp = {
        OrtCustomOp {
            version: 1,
            CreateKernel: Some(create_kernel),
            GetName: Some(get_name),
            GetExecutionProviderType: Some(get_execution_provider_type),
            GetInputType: Some(get_input_type),
            GetInputTypeCount: Some(get_input_type_count),
            GetOutputType: Some(get_output_type),
            GetOutputTypeCount: Some(get_output_type_count),
            KernelCompute: Some(kernel_compute),
            KernelDestroy: Some(kernel_destroy),
            GetInputCharacteristic: Some(get_input_characteristic),
            GetOutputCharacteristic: Some(get_output_characteristic),
        }
    };

    extern "C" fn get_name(_op: *const OrtCustomOp) -> *const c_char {
        dbg!(str_to_c_char_ptr("CustomOpOne"))
    }

    unsafe extern "C" fn get_input_type(
        _op: *const OrtCustomOp,
        _index: size_t,
    ) -> ONNXTensorElementDataType {
        dbg!("input_type");
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    }

    unsafe extern "C" fn get_input_type_count(_op: *const OrtCustomOp) -> size_t {
        dbg!("input_type_count");
        2
    }

    unsafe extern "C" fn get_output_type(
        _op: *const OrtCustomOp,
        _index: size_t,
    ) -> ONNXTensorElementDataType {
        dbg!("output_type");
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    }

    unsafe extern "C" fn get_output_type_count(_op: *const OrtCustomOp) -> size_t {
        dbg!("output_type_count");
        1
    }

    unsafe extern "C" fn kernel_compute(
        op_kernel: *mut c_void,
        context: *mut OrtKernelContext,
    ) {
        let kernel: &Kernel = &*(op_kernel as *const _);
        let api = &*kernel.api;
        let context = &mut *context;

        let (info_x, array_x) = {
            let value = context.get_input(api, 0).unwrap();
            let array = (*value).get_tensor_mutable_data::<f32>(api);
	    let info = (*value).get_type_and_shape_info(api);
            dbg!(info, array)
        };
        let array_y = {
            let value = context.get_input(api, 1).unwrap();
            let array = (*value).get_tensor_mutable_data::<f32>(api);
            dbg!(array)
        };
        let array_z = {
            let value = dbg!(context.get_output(api, 0, &*info_x));
            let array = (*value).get_tensor_mutable_data::<f32>(api);
            dbg!(array)
        };
	for ((x, y), z) in array_x.iter().zip(array_y.iter()).zip(array_z.iter_mut()) {
	    *z  = *x + *y;
	}
        dbg!("kernel_compute-one");
    }
    unsafe extern "C" fn get_input_characteristic(
        _op: *const OrtCustomOp,
        _index: size_t,
    ) -> OrtCustomOpInputOutputCharacteristic {
        dbg!("get_input_characteristic");
        unimplemented!()
    }
    unsafe extern "C" fn get_output_characteristic(
        _op: *const OrtCustomOp,
        _index: size_t,
    ) -> OrtCustomOpInputOutputCharacteristic {
        dbg!("get_out_characteristic");
        unimplemented!()
    }

    unsafe extern "C" fn kernel_destroy(op_kernel: *mut c_void) {
        dbg!("kernel_destroy");
    }
}

mod op_two {
    use super::*;

    pub const OP_TWO: OrtCustomOp = {
        OrtCustomOp {
            version: 1,
            CreateKernel: Some(create_kernel),
            GetName: Some(get_name),
            GetExecutionProviderType: Some(get_execution_provider_type),
            GetInputType: Some(get_input_type),
            GetInputTypeCount: Some(get_input_type_count),
            GetOutputType: Some(get_output_type),
            GetOutputTypeCount: Some(get_output_type_count),
            KernelCompute: Some(kernel_compute),
            KernelDestroy: Some(kernel_destroy),
            GetInputCharacteristic: Some(get_input_characteristic),
            GetOutputCharacteristic: Some(get_output_characteristic),
        }
    };

    extern "C" fn get_name(_op: *const OrtCustomOp) -> *const c_char {
        dbg!(str_to_c_char_ptr("CustomOpTwo"))
    }

    unsafe extern "C" fn get_input_type(
        _op: *const OrtCustomOp,
        _index: size_t,
    ) -> ONNXTensorElementDataType {
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    }

    unsafe extern "C" fn get_input_type_count(_op: *const OrtCustomOp) -> size_t {
        1
    }

    unsafe extern "C" fn get_output_type(
        _op: *const OrtCustomOp,
        _index: size_t,
    ) -> ONNXTensorElementDataType {
        ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    }

    unsafe extern "C" fn get_output_type_count(_op: *const OrtCustomOp) -> size_t {
        1
    }
    unsafe extern "C" fn kernel_compute(
        op_kernel: *mut c_void,
        context: *mut OrtKernelContext,
    ) {
	dbg!("begin compute two");
        let kernel: &Kernel = &*(op_kernel as *const _);
        let api = &*kernel.api;
        let context = &mut *context;

        let (info_x, array_x) = {
            let value = dbg!(context.get_input(api, 0)).unwrap();
            let array = (*value).get_tensor_mutable_data::<f32>(api);
	    let info = (*value).get_type_and_shape_info(api);
            dbg!(info, array)
        };
        let array_z = {
            let value = dbg!(context.get_output(api, 0, &*info_x));
            let array = (*value).get_tensor_mutable_data::<i32>(api);
            dbg!(array)
        };
	for (x, z) in array_x.iter().zip(array_z.iter_mut()) {
	    *z  = x.round() as i32
	}	
        dbg!("kernel_compute-two");
    }
    unsafe extern "C" fn get_input_characteristic(
        _op: *const OrtCustomOp,
        _index: size_t,
    ) -> OrtCustomOpInputOutputCharacteristic {
        dbg!("get_input_characteristic");
        unimplemented!()
    }
    unsafe extern "C" fn get_output_characteristic(
        _op: *const OrtCustomOp,
        _index: size_t,
    ) -> OrtCustomOpInputOutputCharacteristic {
        dbg!("get_out_characteristic");
        unimplemented!()
    }
    unsafe extern "C" fn kernel_destroy(op_kernel: *mut c_void) {
        dbg!("kernel_destroy");
    }
}

#[no_mangle]
pub extern "C" fn RegisterCustomOps(
    options: &mut OrtSessionOptions,
    api_base: &mut OrtApiBase,
) -> *mut OrtStatus {
    // Docs from upstream
    // \brief Register custom ops from a shared library

    // 	Loads a shared library (dll on windows, so on linux, etc)
    //  named 'library_path' and looks for this entry point:

    //     OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);

    // It then passes in the provided session options to this function
    // along with the api base.  The handle to the loaded library is
    // returned in library_handle. It can be freed by the caller after
    // all sessions using the passed in session options are destroyed,
    // or if an error occurs and it is non null.
    let api = unsafe { *api_base.GetApi.unwrap()(12) };
    let domain_ptr = api.create_custom_op_domain("test.customop", options);
    match api.add_op_to_domain(domain_ptr) {
	Ok(_) => std::ptr::null_mut(),
	Err(status) => status,
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     #[test]
//     fn loopy_test() {
//         let api = unsafe{*(*OrtGetApiBase()).GetApi.unwrap()(12)};
// 	let mut so_ptr: *mut OrtSessionOptions = std::ptr::null_mut();
// 	unsafe{api.CreateSessionOptions.unwrap()(&mut so_ptr)};
// 	unsafe{api.RegisterCustomOpsLibrary.unwrap()(so_ptr, str_to_c_char_ptr(""), std::ptr::null_mut())};
//     }

// }
