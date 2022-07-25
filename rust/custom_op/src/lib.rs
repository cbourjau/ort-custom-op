#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::ffi::CString;
use std::os::raw::{c_char, c_void};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

mod api;
use api::*;

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
    ) -> Result<*mut OrtCustomOpDomain> {
        let fun_ptr = self.CreateCustomOpDomain.unwrap();
        let mut domain_ptr: *mut OrtCustomOpDomain = std::ptr::null_mut();
        let c_op_domain = str_to_c_char_ptr(domain);
	unsafe {
            status_to_result(fun_ptr(c_op_domain, &mut domain_ptr))?;
            status_to_result(self.AddCustomOpDomain.unwrap()(options, domain_ptr))?;
	}
        Ok(domain_ptr)
    }

    fn add_op_to_domain(&self, domain: *mut OrtCustomOpDomain) -> Result<()> {
        let fun_ptr = self.CustomOpDomain_Add.unwrap();
        status_to_result(unsafe { fun_ptr(domain, &op_one::OP_ONE) })?;
        status_to_result(unsafe { fun_ptr(domain, &op_two::OP_TWO) })
    }
}

#[derive(Debug)]
struct Kernel {
    _op: *const OrtCustomOp,
    api: *const OrtApi,
    _info: *const OrtKernelInfo,
}

extern "C" fn create_kernel(
    op: *const OrtCustomOp,
    api: *const OrtApi,
    info: *const OrtKernelInfo,
) -> *mut c_void {
    let kernel = Kernel {
        _op: op,
        api,
        _info: info,
    };
    Box::leak(Box::new(kernel)) as *mut _ as *mut c_void
}

unsafe extern "C" fn get_execution_provider_type(
    _op: *const OrtCustomOp,
) -> *const ::std::os::raw::c_char {
    CString::new("CPUExecutionProvider").unwrap().into_raw()
}

unsafe extern "C" fn kernel_destroy(op_kernel: *mut c_void) {
    Box::from_raw(op_kernel as *mut Kernel);
    dbg!("kernel_destroy");
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
        str_to_c_char_ptr("CustomOpOne")
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

    unsafe extern "C" fn kernel_compute(op_kernel: *mut c_void, context: *mut OrtKernelContext) {
        let kernel: &Kernel = &*(op_kernel as *const _);
        let api = Api::from_raw(kernel.api);
        let mut context = KernelContext::from_raw(&api, context);

        let (dims_x, array_x) = {
            let mut value = context.get_input(0).unwrap();
            let array = value.get_tensor_data_mut::<f32>().unwrap();
            let info = value.get_tensor_type_and_shape().unwrap();
            let dims = info.get_dimensions().unwrap();
            dbg!((dims, array))
        };
        let array_y = {
            let mut value = context.get_input(1).unwrap();
            let array = value.get_tensor_data_mut::<f32>().unwrap();
            dbg!(array)
        };
        let array_z = {
            let mut value = context.get_output(0, &dims_x).unwrap();
            let array = value.get_tensor_data_mut::<f32>().unwrap();
            array
        };
        for ((x, y), z) in array_x.iter().zip(array_y.iter()).zip(array_z.iter_mut()) {
            *z = *x + *y;
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
        str_to_c_char_ptr("CustomOpTwo")
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
    unsafe extern "C" fn kernel_compute(op_kernel: *mut c_void, context: *mut OrtKernelContext) {
        let kernel: &Kernel = &*(op_kernel as *const _);
        let api = Api::from_raw(kernel.api);
        let mut context = KernelContext::from_raw(&api, context);

        let (dims_x, array_x) = {
            let mut value = context.get_input(0).unwrap();
            let array = value.get_tensor_data_mut::<f32>().unwrap();
            let info = value.get_tensor_type_and_shape().unwrap();
            let dims = info.get_dimensions().unwrap();
            (dims, array)
        };
        let array_z = {
            let mut value = context.get_output(0, &dims_x).unwrap();
            let array = value.get_tensor_data_mut::<i32>().unwrap();
            array
        };
        for (x, z) in array_x.into_iter().zip(array_z.iter_mut()) {
            *z = x.round() as i32
        }
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
    let api = unsafe { &*api_base.GetApi.unwrap()(12) };
    let status = api.create_custom_op_domain("test.customop", options)
	.and_then(|domain_ptr| {
	    api.add_op_to_domain(domain_ptr)
	});
    match status {
        Ok(_) => std::ptr::null_mut(),
        Err(status) => status,
    }
}
