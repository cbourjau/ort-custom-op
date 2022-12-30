use crate::api::ElementType;
use crate::bindings::{OrtApi, OrtKernelContext};

use ndarray::{ArrayD, ArrayViewD};

/// Trait which qualifies types to be used as input to the
/// `kernel_compute` function of the custom operator.
pub trait Inputs<'s> {
    const INPUT_TYPES: &'static [ElementType];
    fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext) -> Self;
}

/// Trait which qualifies types to be used as outputs by the
/// `kernel_compute` function of the custom operator.
pub trait Outputs {
    const OUTPUT_TYPES: &'static [ElementType];
    fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext);
}

trait Input<'s> {
    const INPUT_TYPE: ElementType;

    fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext, idx: u64) -> Self;
}

macro_rules! impl_input_non_string {
    ($ty:ty, $variant:tt) => {
        impl<'s> Input<'s> for ArrayViewD<'s, $ty> {
            const INPUT_TYPE: ElementType = ElementType::$variant;

            fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext, idx: u64) -> Self {
                api.get_input_array::<$ty>(ctx, idx)
                    .expect("Loading input data of given type failed.")
            }
        }
    };
}

impl_input_non_string!(bool, Bool);
impl_input_non_string!(f32, F32);
impl_input_non_string!(f64, F64);
impl_input_non_string!(i32, I32);
impl_input_non_string!(i64, I64);
impl_input_non_string!(u16, U16);
impl_input_non_string!(u32, U32);
impl_input_non_string!(u64, U64);
impl_input_non_string!(u8, U8);

impl<'s> Input<'s> for ArrayD<String> {
    const INPUT_TYPE: ElementType = ElementType::String;

    fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext, idx: u64) -> Self {
        api.get_input_array_string(ctx, idx)
            .expect("Loading input data of given type failed.")
    }
}

macro_rules! impl_inputs {
    ($($idx:literal),+; $($param:tt),+  ) => {
        impl<'s, $($param,)*> Inputs<'s> for ($($param,)*)
        where
            $($param : Input<'s>,)*
        {
            const INPUT_TYPES: &'static [ElementType] = &[$(<$param as Input>::INPUT_TYPE),*];
            fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext) -> Self {

                (
                    $($param::from_ort(api, ctx, $idx), )*
                )
            }
        }
    };
}
impl_inputs! {0; A}
impl_inputs! {0, 1; A, B}
impl_inputs! {0, 1, 2; A, B, C}
impl_inputs! {0, 1, 2, 3; A, B, C, D}
impl_inputs! {0, 1, 2, 3, 4; A, B, C, D, E}
impl_inputs! {0, 1, 2, 3, 4, 5; A, B, C, D, E, F}

trait Output {
    const OUTPUT_TYPE: ElementType;

    fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext, idx: u64);
}

macro_rules! impl_output_non_string {
    ($ty:ty, $variant:tt) => {
        impl<'s> Output for ArrayD<$ty> {
            const OUTPUT_TYPE: ElementType = ElementType::$variant;

            fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext, idx: u64) {
                let shape = self.shape();
                let shape_i64: Vec<_> = shape.iter().map(|v| *v as i64).collect();
                let val = unsafe { api.get_output(ctx, idx, &shape_i64) }.unwrap();
                let mut arr = api.get_tensor_data_mut(val, &shape).unwrap();
                arr.assign(&self);
            }
        }
    };
}

impl Output for ArrayD<String> {
    const OUTPUT_TYPE: ElementType = ElementType::String;

    fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext, idx: u64) {
        api.fill_string_tensor(ctx, idx, self).unwrap();
    }
}

impl_output_non_string!(bool, Bool);
impl_output_non_string!(f32, F32);
impl_output_non_string!(f64, F64);
impl_output_non_string!(i32, I32);
impl_output_non_string!(i64, I64);
impl_output_non_string!(u16, U16);
impl_output_non_string!(u32, U32);
impl_output_non_string!(u64, U64);
impl_output_non_string!(u8, U8);

// TODO: Impl string output

macro_rules! impl_outputs {
    ($($idx:tt),+; $($param:tt),+  ) => {
        impl<'s, $($param,)*> Outputs for ($($param,)*)
        where
            $($param : Output,)*
        {
            const OUTPUT_TYPES: &'static [ElementType] = &[$(<$param as Output>::OUTPUT_TYPE),*];
            fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext,) {
                $(self.$idx.write_to_ort(api, ctx, $idx);)*
            }
        }
    };
}

impl_outputs! {0; A}
impl_outputs! {0, 1; A, B}
impl_outputs! {0, 1, 2; A, B, C}
impl_outputs! {0, 1, 2, 3; A, B, C, D}
impl_outputs! {0, 1, 2, 3, 4; A, B, C, D, E}
impl_outputs! {0, 1, 2, 3, 4, 5; A, B, C, D, E, F}
