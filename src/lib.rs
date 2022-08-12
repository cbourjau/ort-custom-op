#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::ffi::CString;
use std::os::raw::c_char;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

mod api;
mod custom_op;

use api::*;
use custom_op::{build, CustomOp, Inputs, Outputs};

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
    struct KernelOne;

    #[derive(Debug)]
    struct KernelTwo;

    impl CustomOp for KernelOne {
        const VERSION: u32 = 1;
        const NAME: &'static str = "CustomOpOne";
        const INPUT_TYPES: &'static [ElementType] = &[ElementType::F32, ElementType::F32];
        const OUTPUT_TYPES: &'static [ElementType] = &[ElementType::F32];

        fn kernel_create(_op: &OrtCustomOp, _api: &Api, _info: &OrtKernelInfo) -> Self {
            KernelOne
        }

        fn kernel_compute(&self, _context: &KernelContext, inputs: Inputs, outputs: Outputs) {
            let (array_x, array_y) = inputs.into_2_arrays::<f32, f32>();
            let mut array_z = outputs.into_array(array_x.shape());
            array_z.assign(&(&array_x + &array_y));
        }
    }

    impl CustomOp for KernelTwo {
        const VERSION: u32 = 1;
        const NAME: &'static str = "CustomOpTwo";
        const INPUT_TYPES: &'static [ElementType] = &[ElementType::F32];
        const OUTPUT_TYPES: &'static [ElementType] = &[ElementType::I32];

        fn kernel_create(_op: &OrtCustomOp, _api: &Api, _info: &OrtKernelInfo) -> Self {
            Self
        }

        fn kernel_compute(&self, _context: &KernelContext, inputs: Inputs, outputs: Outputs) {
            let array_x = inputs.into_array::<f32>();
            let mut array_z = outputs.into_array(array_x.shape());
            array_z.assign(&array_x.mapv(|el| el.round() as i32));
        }
    }
}

#[no_mangle]
pub extern "C" fn RegisterCustomOps(
    options: &mut OrtSessionOptions,
    api_base: &mut OrtApiBase,
) -> *mut OrtStatus {
    let mut options = SessionOptions::from_ort(api_base, options);

    let status = options
        .create_custom_op_domain("test.customop")
        .and_then(|mut domain| {
            domain.add_op_to_domain(&op_one::OP_ONE)?;
            domain.add_op_to_domain(&op_one::OP_TWO)
        });
    match status {
        Ok(_) => std::ptr::null_mut(),
        Err(status) => status,
    }
}
