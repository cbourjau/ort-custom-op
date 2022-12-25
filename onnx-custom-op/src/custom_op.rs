use std::ffi::CString;
use std::os::raw::{c_char, c_void};

use crate::api::{ElementType, ExecutionProviders, KernelContext, KernelInfo, OutputValue};
use crate::bindings::{
    size_t, ONNXTensorElementDataType, OrtApi, OrtCustomOp, OrtCustomOpInputOutputCharacteristic,
    OrtKernelContext, OrtKernelInfo,
};

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

#[derive(Debug)]
pub struct Outputs<'s> {
    outputs: Vec<OutputValue<'s>>,
}

impl<'s> Outputs<'s> {
    pub fn into_array<T1>(self, shape1: &'s [usize]) -> ArrayViewMutD<'s, T1> {
        let mut it = self.outputs.into_iter();
        it.next()
            .expect("Too few inputs.")
            .get_tensor_data_mut::<T1>(shape1)
            .expect("Loading input data of given type failed.")
    }
    pub fn into_2_arrays<T1, T2>(
        self,
        shape1: &'s [usize],
        shape2: &'s [usize],
    ) -> (ArrayViewMutD<'s, T1>, ArrayViewMutD<'s, T2>) {
        let mut it = self.outputs.into_iter();
        (
            it.next()
                .expect("Too few inputs.")
                .get_tensor_data_mut::<T1>(shape1)
                .expect("Loading input data of given type failed."),
            it.next()
                .expect("Too few inputs.")
                .get_tensor_data_mut::<T2>(shape2)
                .expect("Loading input data of given type failed."),
        )
    }
}

pub trait Inputs<'s> {
    const INPUT_TYPES: &'static [ElementType];
    fn from_ort(ctx: &KernelContext<'s>) -> Self;
}

trait Input<'s> {
    const INPUT_TYPE: ElementType;

    fn from_ort(ctx: &KernelContext<'s>, idx: u64) -> Self;
}

impl<'s> Input<'s> for ArrayViewD<'s, f32> {
    const INPUT_TYPE: ElementType = ElementType::F32;

    fn from_ort(ctx: &KernelContext<'s>, idx: u64) -> Self {
        ctx.get_input_value(idx)
            .unwrap()
            .get_tensor_data::<f32>()
            .expect("Loading input data of given type failed.")
    }
}

impl<'s> Input<'s> for ArrayD<String> {
    const INPUT_TYPE: ElementType = ElementType::String;

    fn from_ort(ctx: &KernelContext<'s>, idx: u64) -> Self {
        ctx.get_input_value(idx)
            .unwrap()
            .get_tensor_data_str()
            .expect("Loading input data of given type failed.")
    }
}

impl<'s, A> Inputs<'s> for (A,)
where
    A: Input<'s>,
{
    const INPUT_TYPES: &'static [ElementType] = &[<A as Input>::INPUT_TYPE];
    fn from_ort(ctx: &KernelContext<'s>) -> Self {
        (A::from_ort(ctx, 0),)
    }
}

impl<'s, A, B> Inputs<'s> for (A, B)
where
    A: Input<'s>,
    B: Input<'s>,
{
    const INPUT_TYPES: &'static [ElementType] =
        &[<A as Input>::INPUT_TYPE, <B as Input>::INPUT_TYPE];

    fn from_ort(ctx: &KernelContext<'s>) -> Self {
        (A::from_ort(ctx, 0), B::from_ort(ctx, 1))
    }
}

pub trait CustomOp {
    const VERSION: u32;
    const NAME: &'static str;

    const OUTPUT_TYPES: &'static [ElementType];

    type OpInputs<'s>: Inputs<'s>;

    const EXECUTION_PROVIDER: ExecutionProviders = ExecutionProviders::Cpu;

    fn kernel_create(info: &KernelInfo) -> Self;
    fn kernel_compute<'s>(
        &self,
        context: &KernelContext<'s>,
        inputs: Self::OpInputs<'s>,
        outputs: Outputs,
    );
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
        T::EXECUTION_PROVIDER.as_c_char_ptr()
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

    extern "C" fn get_output_type<T: CustomOp>(
        _op: *const OrtCustomOp,
        index: size_t,
    ) -> ONNXTensorElementDataType {
        T::OUTPUT_TYPES[index as usize].to_ort_encoding()
    }

    extern "C" fn get_output_type_count<T: CustomOp>(_op: *const OrtCustomOp) -> size_t {
        T::OUTPUT_TYPES.len() as _
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
        context: *mut OrtKernelContext,
    ) where
        <T as CustomOp>::OpInputs<'s>: Inputs<'s>,
    {
        let wrapped_kernel: &mut WrappedKernel<T> = &mut *(op_kernel as *mut _);
        let kernel = &wrapped_kernel.user_kernel;
        let api = &wrapped_kernel.api;
        let context = KernelContext::from_raw(api, context.as_mut::<'s>().unwrap());

        let n_outputs = context.get_output_count().unwrap();
        let outputs = {
            let mut outputs = vec![];
            for n in 0..n_outputs {
                outputs.push(context.get_output_value(n))
            }
            outputs
        };
        let my_inputs = <T::OpInputs<'s> as Inputs>::from_ort(&context);
        kernel.kernel_compute(&context, my_inputs, Outputs { outputs });
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
