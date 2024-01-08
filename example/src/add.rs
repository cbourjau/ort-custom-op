use std::{convert::Infallible, marker::PhantomData, ops::Add};

use ndarray::{ArrayD, ArrayViewD};

use ort_custom_op::prelude::*;

/// A custom operator which adds its two inputs
pub struct CustomAdd<T> {
    ty: PhantomData<T>,
}

impl<T> CustomOp for CustomAdd<T>
where
    T: 'static + Add<T, Output = T> + Clone,
    for<'foo> (ArrayViewD<'foo, T>, ArrayViewD<'foo, T>): Inputs<'foo>,
    (ArrayD<T>,): Outputs,
{
    type KernelCreateError = Infallible;
    type ComputeError = Infallible;

    const NAME: &'static str = "CustomAdd";

    type OpInputs<'s> = (ArrayViewD<'s, T>, ArrayViewD<'s, T>);
    type OpOutputs = (ArrayD<T>,);

    fn kernel_create(_info: &KernelInfo) -> Result<Self, Self::KernelCreateError> {
        Ok(CustomAdd { ty: PhantomData })
    }

    fn kernel_compute<'s>(
        &self,
        (array_x, array_y): Self::OpInputs<'s>,
    ) -> Result<Self::OpOutputs, Self::ComputeError> {
        Ok((&array_x + &array_y,))
    }
}
