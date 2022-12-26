use chrono::NaiveDateTime;
use ndarray::{ArrayD, ArrayViewD};
use onnx_custom_op::prelude::*;

/// Static object defining the custom operator
pub const OP_CUSTOM_ADD: OrtCustomOp = build::<CustomAdd>();
pub const OP_PARSE_DATETIME: OrtCustomOp = build::<ParseDateTime>();

/// Public function which onnxruntime expects to be in the shared library
#[no_mangle]
pub extern "C" fn RegisterCustomOps(
    options: &mut OrtSessionOptions,
    api_base: &mut OrtApiBase,
) -> *mut OrtStatus {
    let status = create_custom_op_domain(
        options,
        api_base,
        "my.domain",
        &[&OP_CUSTOM_ADD, &OP_PARSE_DATETIME],
    );

    match status {
        Ok(_) => std::ptr::null_mut(),
        Err(status) => status.into_pointer(),
    }
}

/// A custom operator which adds its two inputs
#[derive(Debug)]
pub struct CustomAdd;

impl CustomOp for CustomAdd {
    const VERSION: u32 = 1;
    const NAME: &'static str = "CustomAdd";

    type OpInputs<'s> = (ArrayViewD<'s, f32>, ArrayViewD<'s, f32>);
    type OpOutputs<'s> = (ArrayD<f32>,);

    fn kernel_create(_info: &KernelInfo) -> Self {
        CustomAdd
    }

    fn kernel_compute<'s>(&self, (array_x, array_y): Self::OpInputs<'s>) -> Self::OpOutputs<'s> {
        (&array_x + &array_y,)
    }
}

/// Parse input strings as datetimes using the provided format string.
/// Outputs a tensor of unix timestamps.
pub struct ParseDateTime {
    fmt: String,
}

impl CustomOp for ParseDateTime {
    const VERSION: u32 = 1;
    const NAME: &'static str = "ParseDateTime";

    type OpInputs<'s> = (ArrayD<String>,);
    type OpOutputs<'s> = (ArrayD<i64>,);

    fn kernel_create(info: &KernelInfo) -> Self {
        let fmt = info.get_attribute_string("fmt").unwrap();
        Self { fmt }
    }

    fn kernel_compute<'s>(&self, (array_in,): (ArrayD<String>,)) -> Self::OpOutputs<'s> {
        let out = array_in.mapv(|s| {
            NaiveDateTime::parse_from_str(s.as_str(), self.fmt.as_str())
                .unwrap()
                .timestamp()
        });
        (out,)
    }
}
