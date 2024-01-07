use std::convert::Infallible;

use anyhow::Error;
use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator showcasing the available attribute types
#[derive(Debug)]
pub struct AttrShowcase {
    float_attr: f32,
    int_attr: i64,
    string_attr: String,
    _u8_tensor: ArrayD<u8>,
    _floats_attr: Vec<f32>,
    _ints_attr: Vec<i64>,
}

impl CustomOp for AttrShowcase {
    type KernelCreateError = Error;
    type ComputeError = Infallible;

    const NAME: &'static str = "AttrShowcase";

    type OpInputs<'s> = (ArrayViewD<'s, f32>, ArrayViewD<'s, i64>, ArrayD<&'s str>);
    type OpOutputs = (ArrayD<f32>, ArrayD<i64>, ArrayD<String>);

    fn kernel_create(info: &KernelInfo) -> Result<Self, Self::KernelCreateError> {
        Ok(AttrShowcase {
            float_attr: info.get_attribute_f32("float_attr")?,
            int_attr: info.get_attribute_i64("int_attr")?,
            string_attr: info.get_attribute_string("string_attr")?,
            _u8_tensor: info.get_attribute_tensor("u8_tensor")?.to_owned(),
            _floats_attr: info.get_attribute_f32s("floats_attr")?,
            _ints_attr: info.get_attribute_i64s("ints_attr")?,
        })
    }

    fn kernel_compute(
        &self,
        (a, b, c): Self::OpInputs<'_>,
    ) -> Result<Self::OpOutputs, Self::ComputeError> {
        let a = &a + self.float_attr;
        let b = &b + self.int_attr;
        let c = c.mapv_into(|v| v + " + " + &self.string_attr);
        Ok((a, b, c))
    }
}
