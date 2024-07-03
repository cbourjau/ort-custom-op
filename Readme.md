# ort-custom-op

The `ort-custom-op` crate provides a framework to safely write custom operators for the `onnxruntime`. 


## Quick start

The `example` crate in the likewise named directory provides a comprehensive end-to-end example of the functionality of the `ort-custom-op` crate. Below follows a brief quick start tutorial.

Custom operators may be made available in an `onnxruntime` session by providing a shared library that exposes a single function named `RegisterCustomOps`. The task of this function is to register custom operators in the form of function tables; namely `OrtCustomOp` objects. This crate provides tooling to easily create `OrtCustomOp` objects from anything that implements the `CustomOp` trait via the `ort_custom_op::build` function.

The following showcases how one may define a custom operator that adds a constant to each element of an input tensor of data type `i64`.

The containing crate must be compiled as a `cdylib` crate.


```rust
use std::convert::Infallible;

use anyhow::Error;
use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator that adds a constant to every input of an i64 input
/// tensor.
pub struct AddConstant {
    /// State associated with this operator. Derived from the
    /// respective node's attributes.
    constant: i64,
}

impl CustomOp for AddConstant {
    /// Possible error raised when setting up the kernel during
    /// session creation.
    type KernelCreateError = Error;
    /// Possible error raised during compute call.
    type ComputeError = Infallible;

    /// Name of the operator within its domain
    const NAME: &'static str = "AddConstant";

    /// Input array(s) (as a tuple)
    type OpInputs<'s> = (ArrayViewD<'s, i64>,);

    /// Output array(s) (as a tuple)
    type OpOutputs = (ArrayD<i64>,);

    /// Function called once per node during session creation.
    fn kernel_create(info: &KernelInfo) -> Result<Self, Self::KernelCreateError> {
        // Read and parse attribute from kernel info object.
        let constant = info.get_attribute_i64("constant")?;
        Ok(AddConstant { constant })
    }

    /// Function called during inference.
    fn kernel_compute(
        &self,
        (array,): Self::OpInputs<'_>,
    ) -> Result<Self::OpOutputs, Self::ComputeError> {
        Ok((&array + self.constant,))
    }
}

// Create a constant function table as expected by onnxruntime
const ADD_CONSTANT_OP: OrtCustomOp = build::<AddConstant>();

/// Public function which onnxruntime expects to be in the shared library
#[no_mangle]
pub extern "C" fn RegisterCustomOps(
    options: &mut OrtSessionOptions,
    api_base: &mut OrtApiBase,
) -> *mut OrtStatus {
    create_custom_op_domain(options, api_base, "my.domain", &[&ADD_CONSTANT_OP])
}
```
