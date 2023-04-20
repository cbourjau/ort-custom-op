use std::ffi::CString;
use std::os::raw::{c_char, c_void};

use crate::api::KernelInfo;
use crate::bindings::{
    ONNXTensorElementDataType, OrtApi, OrtCustomOp, OrtCustomOpInputOutputCharacteristic,
    OrtKernelContext, OrtKernelInfo,
};
pub use crate::inputs_and_outputs::{Inputs, Outputs};

/// Trait defining the behavior of a custom operator.
pub trait CustomOp {
    const VERSION: u32;
    const NAME: &'static str;

    type OpInputs<'s>: Inputs<'s>;
    type OpOutputs: Outputs;

    fn kernel_create(info: &KernelInfo) -> Self;
    fn kernel_compute(&self, inputs: Self::OpInputs<'_>) -> Self::OpOutputs;
}

/// Function to build static instances of `OrtCustomOp`.
///
/// These static objects are registered using the
/// `create_custom_op_domain` function.
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

    extern "C" fn get_input_type<T: CustomOp>(
        _op: *const OrtCustomOp,
        index: usize,
    ) -> ONNXTensorElementDataType {
        <<T as CustomOp>::OpInputs<'_> as Inputs>::INPUT_TYPES[index as usize].to_ort_encoding()
    }

    extern "C" fn get_input_type_count<T: CustomOp>(_op: *const OrtCustomOp) -> usize {
        <<T as CustomOp>::OpInputs<'_> as Inputs>::INPUT_TYPES.len() as _
    }

    extern "C" fn get_output_type<T: CustomOp>(
        _op: *const OrtCustomOp,
        index: usize,
    ) -> ONNXTensorElementDataType {
        <<T as CustomOp>::OpOutputs as Outputs>::OUTPUT_TYPES[index as usize].to_ort_encoding()
    }

    extern "C" fn get_output_type_count<T: CustomOp>(_op: *const OrtCustomOp) -> usize {
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
        _index: usize,
    ) -> OrtCustomOpInputOutputCharacteristic {
        unimplemented!()
    }

    extern "C" fn get_output_characteristic(
        _op: *const OrtCustomOp,
        _index: usize,
    ) -> OrtCustomOpInputOutputCharacteristic {
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

struct WrappedKernel<T> {
    user_kernel: T,
    api: &'static OrtApi,
}
