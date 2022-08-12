use std::ffi::CString;
use std::os::raw::c_char;

mod api;
mod bindings;
mod custom_op;

use api::*;
use custom_op::{build, CustomOp, Inputs, Outputs};

use bindings::{OrtApiBase, OrtCustomOp, OrtKernelInfo, OrtSessionOptions, OrtStatus};

// Implement the CustomOp trait for two custom operators. In this
// case, there are no attributes and thus the kernels have no data.
mod op_one {
    use super::*;

    #[derive(Debug)]
    pub struct KernelOne;

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
}

mod op_two {
    use super::*;

    #[derive(Debug)]
    pub struct KernelTwo;

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

// Create static objects conforming to the api expected by the ONNX runtime
pub const OP_ONE: OrtCustomOp = build::<op_one::KernelOne>();
pub const OP_TWO: OrtCustomOp = build::<op_two::KernelTwo>();

// Define the one public function expected by the ONNX runtime when loading the shared library.
#[no_mangle]
pub extern "C" fn RegisterCustomOps(
    options: &mut OrtSessionOptions,
    api_base: &mut OrtApiBase,
) -> *mut OrtStatus {
    let mut options = SessionOptions::from_ort(api_base, options);

    let status = options
        .create_custom_op_domain("test.customop")
        .and_then(|mut domain| {
            domain.add_op_to_domain(&OP_ONE)?;
            domain.add_op_to_domain(&OP_TWO)
        });
    match status {
        Ok(_) => std::ptr::null_mut(),
        Err(status) => status,
    }
}
