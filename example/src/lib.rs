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
    let mut options = SessionOptions::from_ort(api_base, options);

    let status = options
        .create_custom_op_domain("my.domain")
        .and_then(|mut domain| {
            domain.add_op_to_domain(&OP_CUSTOM_ADD)?;
            domain.add_op_to_domain(&OP_PARSE_DATETIME)
        });
    match status {
        Ok(_) => std::ptr::null_mut(),
        Err(status) => status,
    }
}

/// A custom operator which adds its two inputs
#[derive(Debug)]
pub struct CustomAdd;

impl CustomOp for CustomAdd {
    const VERSION: u32 = 1;
    const NAME: &'static str = "CustomAdd";
    const OUTPUT_TYPES: &'static [ElementType] = &[ElementType::F32];

    type OpInputs<'s> = (ArrayViewD<'s, f32>, ArrayViewD<'s, f32>);

    fn kernel_create(_info: &KernelInfo) -> Self {
        CustomAdd
    }

    fn kernel_compute<'s>(
        &self,
        _context: &KernelContext<'s>,
        (array_x, array_y): Self::OpInputs<'s>,
        outputs: Outputs,
    ) {
        let mut array_z = outputs.into_array(array_x.shape());
        array_z.assign(&(&array_x + &array_y));
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
    const OUTPUT_TYPES: &'static [ElementType] = &[ElementType::I64];

    type OpInputs<'s> = (ArrayD<String>,);

    fn kernel_create(info: &KernelInfo) -> Self {
        let fmt = dbg!(info.get_attribute_string("fmt")).unwrap();
        Self { fmt }
    }

    fn kernel_compute<'s>(
        &self,
        _context: &KernelContext<'s>,
        (array_in,): (ArrayD<String>,),
        outputs: Outputs,
    ) {
        let mut array_out = outputs.into_array(array_in.shape());

        let out = array_in.mapv(|s| {
            NaiveDateTime::parse_from_str(s.as_str(), self.fmt.as_str())
                .unwrap()
                .timestamp()
        });
        array_out.assign(&out);
    }
}
