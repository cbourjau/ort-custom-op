use std::ffi::CString;
use std::os::raw::{c_char, c_void};

use crate::api::{Api, ElementType, ExecutionProviders, KernelContext, OutputValue, Value};
use crate::bindings::{
    size_t, ONNXTensorElementDataType, OrtApi, OrtCustomOp, OrtCustomOpInputOutputCharacteristic,
    OrtKernelContext, OrtKernelInfo,
};

pub trait IntoArrays<T> {
    fn into_arrays(self) -> T;
}
use ndarray::{ArrayView, ArrayViewMut, IxDyn};

#[derive(Debug)]
pub struct Inputs<'s> {
    inputs: Vec<Value<'s>>,
}

#[derive(Debug)]
pub struct Outputs<'s> {
    outputs: Vec<OutputValue<'s>>,
}

impl<'s> Inputs<'s> {
    pub fn into_array<T1>(self) -> ArrayView<'s, T1, IxDyn> {
        self.inputs
            .into_iter()
            .next()
            .expect("Too few inputs.")
            .get_tensor_data::<T1>()
            .expect("Loading input data of given type failed.")
    }
    pub fn into_2_arrays<T1, T2>(self) -> (ArrayView<'s, T1, IxDyn>, ArrayView<'s, T2, IxDyn>) {
        let mut it = self.inputs.into_iter();
        (
            it.next()
                .expect("Too few inputs.")
                .get_tensor_data::<T1>()
                .expect("Loading input data of given type failed."),
            it.next()
                .expect("Too few inputs.")
                .get_tensor_data::<T2>()
                .expect("Loading input data of given type failed."),
        )
    }
}

impl<'s> Outputs<'s> {
    pub fn into_array<T1>(self, shape1: &'s [usize]) -> ArrayViewMut<'s, T1, IxDyn> {
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
    ) -> (ArrayViewMut<'s, T1, IxDyn>, ArrayViewMut<'s, T2, IxDyn>) {
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

pub trait CustomOp {
    const VERSION: u32;
    const NAME: &'static str;
    const INPUT_TYPES: &'static [ElementType];
    const OUTPUT_TYPES: &'static [ElementType];

    const EXECUTION_PROVIDER: ExecutionProviders = ExecutionProviders::Cpu;
    // fn get_input_type(op: &OrtCustomOp, index: usize);
    fn kernel_create(op: &OrtCustomOp, api: &Api, info: &OrtKernelInfo) -> Self;
    fn kernel_compute(&self, context: &KernelContext, inputs: Inputs, outputs: Outputs);
}

struct WrappedKernel<T> {
    user_kernel: T,
    api: Api,
}

pub const fn build<T: CustomOp>() -> OrtCustomOp {
    extern "C" fn get_name<T: CustomOp>(_op: *const OrtCustomOp) -> *const c_char {
        CString::new(T::NAME).unwrap().into_raw()
    }

    extern "C" fn get_execution_provider_type<T: CustomOp>(
        _op: *const OrtCustomOp,
    ) -> *const c_char {
        T::EXECUTION_PROVIDER.as_c_char_ptr()
    }

    extern "C" fn get_input_type<T: CustomOp>(
        _op: *const OrtCustomOp,
        index: size_t,
    ) -> ONNXTensorElementDataType {
        T::INPUT_TYPES[index as usize].to_ort_encoding()
    }

    extern "C" fn get_input_type_count<T: CustomOp>(_op: *const OrtCustomOp) -> size_t {
        T::INPUT_TYPES.len() as _
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
        op: *const OrtCustomOp,
        api: *const OrtApi,
        info: *const OrtKernelInfo,
    ) -> *mut c_void {
        let api = Api::from_raw(&*api);
        let user_kernel = T::kernel_create(&*op, &api, &*info);
        let wrapped_kernel = WrappedKernel { user_kernel, api };
        Box::leak(Box::new(wrapped_kernel)) as *mut _ as *mut c_void
    }

    unsafe extern "C" fn kernel_compute<T: CustomOp>(
        op_kernel: *mut c_void,
        context: *mut OrtKernelContext,
    ) {
        let wrapped_kernel: &mut WrappedKernel<T> = &mut *(op_kernel as *mut _);
        let kernel = &wrapped_kernel.user_kernel;
        let api = &wrapped_kernel.api;

        // Create to Context objects since we need to borrow it
        // mutably for the output. The second one could also access
        // the same output memory mutably, so this is not safe!
        let context_output = KernelContext::from_raw(&api, context);

        let n_inputs = context_output.get_input_count().unwrap();
        let inputs = {
            let mut inputs = vec![];
            for n in 0..n_inputs {
                inputs.push(
                    KernelContext::from_raw(&api, context)
                        .get_input_value(n)
                        .unwrap(),
                )
            }
            inputs
        };

        let n_outputs = context_output.get_output_count().unwrap();
        let outputs = {
            let mut outputs = vec![];
            for n in 0..n_outputs {
                outputs.push(KernelContext::from_raw(&api, context).get_safe_output(n))
            }
            outputs
        };
        kernel.kernel_compute(&context_output, Inputs { inputs }, Outputs { outputs });
    }

    unsafe extern "C" fn kernel_destroy<T: CustomOp>(op_kernel: *mut c_void) {
        Box::from_raw(op_kernel as *mut WrappedKernel<T>);
        dbg!("kernel_destroy");
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
