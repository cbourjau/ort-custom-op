use std::convert::Infallible;

use anyhow::Error;
use chrono::NaiveDateTime;
use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// Parse input strings as datetimes using the provided format string.
/// Outputs a tensor of unix timestamps as a float64. Invalid inputs
/// are mapped to f64::NAN to make the operation infallible.
pub struct ParseDateTime {
    fmt: String,
}

impl CustomOp for ParseDateTime {
    type KernelCreateError = Error;
    type ComputeError = Infallible;
    const NAME: &'static str = "ParseDateTime";

    type OpInputs<'s> = (ArrayViewD<'s, &'s str>,);
    type OpOutputs = (ArrayD<f64>,);

    fn kernel_create(info: &KernelInfo) -> Result<Self, Self::KernelCreateError> {
        let fmt = info.get_attribute_string("fmt")?;
        Ok(Self { fmt })
    }

    fn kernel_compute<'s>(
        &self,
        (array_in,): (ArrayViewD<'s, &'s str>,),
    ) -> Result<Self::OpOutputs, Self::ComputeError> {
        let out = array_in.mapv(|s| {
            NaiveDateTime::parse_from_str(s, &self.fmt)
                .map(|dt| dt.timestamp() as f64)
                .unwrap_or_else(|_| f64::NAN)
        });
        Ok((out,))
    }
}
