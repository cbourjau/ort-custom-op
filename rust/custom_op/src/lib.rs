#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::ffi::CString;
use std::os::raw::{c_char, c_void};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

mod api;
mod custom_op;

use api::*;
use custom_op::{build, CustomOp};

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

pub enum ExecutionProviders {
    Cpu,
}

impl ExecutionProviders {
    /// Execution provider as null terminated string with static
    /// lifetime.
    fn into_c_char_ptr(&self) -> *const c_char {
        let null_term_str = match self {
            Self::Cpu => b"CPUExecutionProvider\0".as_ptr(),
        };
        null_term_str as *const _
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
        status_to_result(unsafe { fun_ptr(domain, &op_one::OP_TWO) })
    }
}

mod op_one {
    use super::*;

    pub const OP_ONE: OrtCustomOp = build::<KernelOne>();
    pub const OP_TWO: OrtCustomOp = build::<KernelTwo>();

    #[derive(Debug)]
    struct KernelOne {
        api: Api,
    }

    #[derive(Debug)]
    struct KernelTwo {
        api: Api,
    }

    impl CustomOp for KernelOne {
        const VERSION: u32 = 1;
        const NAME: &'static str = "CustomOpOne";
        const INPUT_TYPES: &'static [u32] = &[
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        ];
        const OUTPUT_TYPES: &'static [u32] =
            &[ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT];

        fn get_api(&self) -> &Api {
            &self.api
        }
        fn kernel_create(_op: &OrtCustomOp, api: Api, _info: &OrtKernelInfo) -> Self {
            KernelOne { api }
        }

        fn kernel_compute(&self, context: &mut KernelContext) {
            let (dims_x, array_x) = {
                let mut value = context.get_input(0).unwrap();
                let array = unsafe { value.get_tensor_data_mut::<f32>().unwrap() };
                let info = value.get_tensor_type_and_shape().unwrap();
                let dims = info.get_dimensions().unwrap();
                dbg!((dims, array))
            };
            let array_y = {
                let mut value = context.get_input(1).unwrap();
                let array = unsafe { value.get_tensor_data_mut::<f32>().unwrap() };
                dbg!(array)
            };
            let array_z = {
                let mut value = context.get_output(0, &dims_x).unwrap();
                let array = unsafe { value.get_tensor_data_mut::<f32>().unwrap() };
                array
            };
            for ((x, y), z) in array_x.iter().zip(array_y.iter()).zip(array_z.iter_mut()) {
                *z = *x + *y;
            }
        }
    }

    impl CustomOp for KernelTwo {
        const VERSION: u32 = 1;
        const NAME: &'static str = "CustomOpTwo";
        const INPUT_TYPES: &'static [u32] =
            &[ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT];
        const OUTPUT_TYPES: &'static [u32] =
            &[ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32];

        fn get_api(&self) -> &Api {
            &self.api
        }
        fn kernel_create(_op: &OrtCustomOp, api: Api, _info: &OrtKernelInfo) -> Self {
            Self { api }
        }

        fn kernel_compute(&self, context: &mut KernelContext) {
            let (dims_x, array_x) = {
                let mut value = context.get_input(0).unwrap();
                let array = unsafe { value.get_tensor_data_mut::<f32>().unwrap() };
                let info = value.get_tensor_type_and_shape().unwrap();
                let dims = info.get_dimensions().unwrap();
                (dims, array)
            };
            let array_z = {
                let mut value = context.get_output(0, &dims_x).unwrap();
                let array = unsafe { value.get_tensor_data_mut::<i32>().unwrap() };
                array
            };
            for (x, z) in array_x.into_iter().zip(array_z.iter_mut()) {
                *z = x.round() as i32
            }
        }
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
    let status = api
        .create_custom_op_domain("test.customop", options)
        .and_then(|domain_ptr| api.add_op_to_domain(domain_ptr));
    match status {
        Ok(_) => std::ptr::null_mut(),
        Err(status) => status,
    }
}
