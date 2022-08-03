#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::ffi::CString;
use std::os::raw::c_char;

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
    if ptr.is_null() {
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
    const fn as_c_char_ptr(&self) -> &'static c_char {
        let null_term_str = match self {
            Self::Cpu => b"CPUExecutionProvider\0".as_ptr(),
        };
        unsafe { &*(null_term_str as *const _) }
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

        fn kernel_compute(&self, context: &KernelContext, mut outputs: Vec<SafeValue>) {
            let (dims_x, array_x) = {
                let value = context.get_input(0).unwrap();
                let info = value.get_tensor_type_and_shape().unwrap();
                let array = unsafe { value.get_tensor_data_mut::<f32>().unwrap() };
                let dims = info.get_dimensions().unwrap();
                dbg!((dims, array))
            };
            let array_y = {
                let value = context.get_input(1).unwrap();
                let array = unsafe { value.get_tensor_data_mut::<f32>().unwrap() };
                dbg!(array)
            };
            let array_z = { outputs[0].get_output_mut::<f32>(&dims_x).unwrap() };
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

        fn kernel_compute(&self, context: &KernelContext, mut outputs: Vec<SafeValue>) {
            let (dims_x, array_x) = {
                let value = context.get_input(0).unwrap();
                let info = value.get_tensor_type_and_shape().unwrap();
                let array = unsafe { value.get_tensor_data_mut::<f32>().unwrap() };
                let dims = info.get_dimensions().unwrap();
                (dims, array)
            };
            let array_z = { outputs[0].get_output_mut::<i32>(&dims_x).unwrap() };
            for (x, z) in array_x.iter_mut().zip(array_z.iter_mut()) {
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
    let api = Api::from_raw(unsafe { &*api_base.GetApi.unwrap()(12) });
    let status = api
        .create_custom_op_domain("test.customop", options)
        .and_then(|mut domain| {
            domain.add_op_to_domain(&op_one::OP_ONE)?;
            domain.add_op_to_domain(&op_one::OP_TWO)
        });
    match status {
        Ok(_) => std::ptr::null_mut(),
        Err(status) => status,
    }
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    /// Run pytest in a subprocess.
    #[test]
    fn test_with_onnxruntime() {
        let output = Command::new("pytest")
            .arg("-s")
            .output()
            .expect("pytest failed");
        dbg!(String::from_utf8(output.stdout).unwrap());
        println!("{}", String::from_utf8(output.stderr).unwrap());
    }
}
