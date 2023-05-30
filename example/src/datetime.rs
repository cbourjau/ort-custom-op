use chrono::NaiveDateTime;
use ndarray::ArrayD;

use ort_custom_op::prelude::*;

/// Parse input strings as datetimes using the provided format string.
/// Outputs a tensor of unix timestamps as a float64. Invalid inputs
/// are mapped to f64::NAN to make the operation infallible.
pub struct ParseDateTime {
    fmt: String,
}

impl CustomOp for ParseDateTime {
    const NAME: &'static str = "ParseDateTime";

    type OpInputs<'s> = (ArrayD<String>,);
    type OpOutputs = (ArrayD<f64>,);

    fn kernel_create(info: &KernelInfo) -> Self {
        // There is no error propagation across the c-api at this
        // point. At least the kernel creation is happening at session
        // creation time so we don't get a nasty surprise at runtime.
        let fmt = info.get_attribute_string("fmt").expect("");
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
