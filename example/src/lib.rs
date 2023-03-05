use chrono::NaiveDateTime;
use ndarray::{ArrayD, ArrayViewD};
use ort_custom_op::prelude::*;

/// Static objects defining the custom operators
const OP_CUSTOM_ADD: OrtCustomOp = build::<CustomAdd>();
const OP_PARSE_DATETIME: OrtCustomOp = build::<ParseDateTime>();
const OP_ATTR_SHOWCASE: OrtCustomOp = build::<AttrShowcase>();

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
        &[&OP_CUSTOM_ADD, &OP_PARSE_DATETIME, &OP_ATTR_SHOWCASE],
    );

    match status {
        Ok(_) => std::ptr::null_mut(),
        Err(status) => status.into_pointer(),
    }
}

/// A custom operator which adds its two inputs
struct CustomAdd;

impl CustomOp for CustomAdd {
    const VERSION: u32 = 1;
    const NAME: &'static str = "CustomAdd";

    type OpInputs<'s> = (ArrayViewD<'s, f32>, ArrayViewD<'s, f32>);
    type OpOutputs = (ArrayD<f32>,);

    fn kernel_create(_info: &KernelInfo) -> Self {
        CustomAdd
    }

    fn kernel_compute(&self, (array_x, array_y): Self::OpInputs<'_>) -> Self::OpOutputs {
        (&array_x + &array_y,)
    }
}

/// Parse input strings as datetimes using the provided format string.
/// Outputs a tensor of unix timestamps as a float64. Invalid inputs
/// are mapped to f64::NAN to make the operation infallible.
struct ParseDateTime {
    fmt: String,
}

impl CustomOp for ParseDateTime {
    const VERSION: u32 = 1;
    const NAME: &'static str = "ParseDateTime";

    type OpInputs<'s> = (ArrayD<String>,);
    type OpOutputs = (ArrayD<f64>,);

    fn kernel_create(info: &KernelInfo) -> Self {
        let fmt = info.get_attribute_string("fmt").unwrap();
        Self { fmt }
    }

    fn kernel_compute<'s>(&self, (array_in,): (ArrayD<String>,)) -> Self::OpOutputs {
        let out = array_in.mapv(|s| {
            NaiveDateTime::parse_from_str(s.as_str(), self.fmt.as_str())
                .map(|dt| dt.timestamp() as f64)
                .unwrap_or_else(|_| f64::NAN)
        });
        (out,)
    }
}

/// A custom operator showcasing the available attribute types
#[derive(Debug)]
struct AttrShowcase {
    float_attr: f32,
    int_attr: i64,
    string_attr: String,

    _floats_attr: Vec<f32>,
    _ints_attr: Vec<i64>,
}

impl CustomOp for AttrShowcase {
    const VERSION: u32 = 1;
    const NAME: &'static str = "AttrShowcase";

    type OpInputs<'s> = (ArrayViewD<'s, f32>, ArrayViewD<'s, i64>, ArrayD<String>);
    type OpOutputs = (ArrayD<f32>, ArrayD<i64>, ArrayD<String>);

    fn kernel_create(info: &KernelInfo) -> Self {
        AttrShowcase {
            float_attr: info.get_attribute_f32("float_attr").unwrap(),
            int_attr: info.get_attribute_i64("int_attr").unwrap(),
            string_attr: info.get_attribute_string("string_attr").unwrap(),
            _floats_attr: info.get_attribute_f32s("floats_attr").unwrap(),
            _ints_attr: info.get_attribute_i64s("ints_attr").unwrap(),
        }
    }

    fn kernel_compute(&self, (a, b, c): Self::OpInputs<'_>) -> Self::OpOutputs {
        let a = &a + self.float_attr;
        let b = &b + self.int_attr;
        let c = c.mapv_into(|v| v + " + " + &self.string_attr);
        (a, b, c)
    }
}
