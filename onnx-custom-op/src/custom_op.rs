use std::ffi::CString;
use std::os::raw::{c_char, c_void};

use crate::api::{ElementType, KernelInfo};
use crate::bindings::{
    size_t, ONNXTensorElementDataType, OrtApi, OrtCustomOp, OrtCustomOpInputOutputCharacteristic,
    OrtKernelContext, OrtKernelInfo,
};

use ndarray::{ArrayD, ArrayViewD};

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

pub trait Inputs<'s> {
    const INPUT_TYPES: &'static [ElementType];
    fn from_ort(api: &OrtApi, ctx: &'s OrtKernelContext) -> Self;
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

pub trait Outputs {
    const OUTPUT_TYPES: &'static [ElementType];
    fn write_to_ort(self, api: &OrtApi, ctx: &mut OrtKernelContext);
}

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

pub trait CustomOp {
    const VERSION: u32;
    const NAME: &'static str;

    type OpInputs<'s>: Inputs<'s>;
    type OpOutputs: Outputs;

    fn kernel_create(info: &KernelInfo) -> Self;
    fn kernel_compute(&self, inputs: Self::OpInputs<'_>) -> Self::OpOutputs;
}

struct WrappedKernel<T> {
    user_kernel: T,
    api: &'static OrtApi,
}

pub const fn build<'s, T>() -> OrtCustomOp
where
    T: CustomOp,
    <T as CustomOp>::OpInputs<'s>: Inputs<'s>,
{
    extern "C" fn get_name<T: CustomOp>(_op: *const OrtCustomOp) -> *const c_char {
        CString::new(T::NAME).unwrap().into_raw()
    }

    extern "C" fn get_execution_provider_type<T: CustomOp>(
        _op: *const OrtCustomOp,
    ) -> *const c_char {
        b"CPUExecutionProvider\0".as_ptr() as *const _
    }

    extern "C" fn get_input_type<'s, T: CustomOp>(
        _op: *const OrtCustomOp,
        index: size_t,
    ) -> ONNXTensorElementDataType {
        <<T as CustomOp>::OpInputs<'s> as Inputs<'s>>::INPUT_TYPES[index as usize].to_ort_encoding()
    }

    extern "C" fn get_input_type_count<'s, T: CustomOp>(_op: *const OrtCustomOp) -> size_t {
        <<T as CustomOp>::OpInputs<'s> as Inputs<'s>>::INPUT_TYPES.len() as _
    }

    extern "C" fn get_output_type<'s, T: CustomOp>(
        _op: *const OrtCustomOp,
        index: size_t,
    ) -> ONNXTensorElementDataType {
        <<T as CustomOp>::OpOutputs as Outputs>::OUTPUT_TYPES[index as usize].to_ort_encoding()
    }

    extern "C" fn get_output_type_count<'s, T: CustomOp>(_op: *const OrtCustomOp) -> size_t {
        <<T as CustomOp>::OpOutputs as Outputs>::OUTPUT_TYPES.len() as _
    }

    unsafe extern "C" fn create_kernel<T: CustomOp>(
        _ort_op: *const OrtCustomOp,
        ort_api: *const OrtApi,
        ort_info: *const OrtKernelInfo,
    ) -> *mut c_void {
        let api = &*ort_api;
        let info = KernelInfo::from_ort(api, &*ort_info);
        let user_kernel = T::kernel_create(&info);
        let wrapped_kernel = WrappedKernel { user_kernel, api };
        Box::leak(Box::new(wrapped_kernel)) as *mut _ as *mut c_void
    }

    unsafe extern "C" fn kernel_compute<'s, T: CustomOp>(
        op_kernel: *mut c_void,
        context_ptr: *mut OrtKernelContext,
    ) where
        <T as CustomOp>::OpInputs<'s>: Inputs<'s>,
    {
        let wrapped_kernel: &mut WrappedKernel<T> = &mut *(op_kernel as *mut _);
        let kernel = &wrapped_kernel.user_kernel;
        let api = &wrapped_kernel.api;
        let context = context_ptr.as_ref::<'s>().unwrap();

        let outputs = {
            let inputs = <T::OpInputs<'s> as Inputs>::from_ort(api, context);
            kernel.kernel_compute(inputs)
        };

        // Bad hack to use a second context here! The first one is
        // still borrowed immutably at this point :/
        let out_context = context_ptr.as_mut::<'s>().unwrap();
        outputs.write_to_ort(api, out_context);
    }

    unsafe extern "C" fn kernel_destroy<T: CustomOp>(op_kernel: *mut c_void) {
        drop(Box::from_raw(op_kernel as *mut WrappedKernel<T>));
    }

    extern "C" fn get_input_characteristic(
        _op: *const OrtCustomOp,
        _index: size_t,
    ) -> OrtCustomOpInputOutputCharacteristic {
        dbg!("get_input_characteristic");
        unimplemented!()
    }

    extern "C" fn get_output_characteristic(
        _op: *const OrtCustomOp,
        _index: size_t,
    ) -> OrtCustomOpInputOutputCharacteristic {
        dbg!("get_out_characteristic");
        unimplemented!()
    }

    OrtCustomOp {
        version: T::VERSION,
        CreateKernel: Some(create_kernel::<T>),
        GetName: Some(get_name::<T>),
        GetExecutionProviderType: Some(get_execution_provider_type::<T>),
        GetInputType: Some(get_input_type::<T>),
        GetInputTypeCount: Some(get_input_type_count::<T>),
        GetOutputType: Some(get_output_type::<T>),
        GetOutputTypeCount: Some(get_output_type_count::<T>),
        KernelCompute: Some(kernel_compute::<T>),
        KernelDestroy: Some(kernel_destroy::<T>),
        GetInputCharacteristic: Some(get_input_characteristic),
        GetOutputCharacteristic: Some(get_output_characteristic),
    }
}
