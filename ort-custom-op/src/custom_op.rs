use std::ffi::CString;
use std::os::raw::{c_char, c_void};

use crate::api::{KernelInfo, API_VERSION};
use crate::bindings::{
    ONNXTensorElementDataType, OrtApi, OrtCustomOp, OrtCustomOpInputOutputCharacteristic,
    OrtErrorCode_ORT_RUNTIME_EXCEPTION, OrtKernelContext, OrtKernelInfo, OrtMemType,
    OrtMemType_OrtMemTypeDefault, OrtStatus,
};
pub use crate::inputs::Inputs;
pub use crate::outputs::Outputs;

/// Trait defining the behavior of a custom operator.
pub trait CustomOp {
    type KernelCreateError;
    type ComputeError;
    const NAME: &'static str;

    type OpInputs<'s>: Inputs<'s>;
    type OpOutputs: Outputs;

    fn kernel_create(info: &KernelInfo) -> Result<Self, Self::KernelCreateError>
    where
        Self: Sized;
    fn kernel_compute(
        &self,
        inputs: Self::OpInputs<'_>,
    ) -> Result<Self::OpOutputs, Self::ComputeError>;
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
        <<T as CustomOp>::OpInputs<'_> as Inputs>::INPUT_TYPES[index].to_ort_encoding()
    }

    extern "C" fn get_input_type_count<T: CustomOp>(_op: *const OrtCustomOp) -> usize {
        <<T as CustomOp>::OpInputs<'_> as Inputs>::CHARACTERISTICS.len()
    }

    extern "C" fn get_output_type<T: CustomOp>(
        _op: *const OrtCustomOp,
        index: usize,
    ) -> ONNXTensorElementDataType {
        <<T as CustomOp>::OpOutputs as Outputs>::OUTPUT_TYPES[index].to_ort_encoding()
    }

    extern "C" fn get_output_type_count<T: CustomOp>(_op: *const OrtCustomOp) -> usize {
        <<T as CustomOp>::OpOutputs as Outputs>::OUTPUT_TYPES.len()
    }

    unsafe extern "C" fn create_kernel_fallible<T: CustomOp>(
        _ort_op: *const OrtCustomOp,
        ort_api: *const OrtApi,
        ort_info: *const OrtKernelInfo,
        kernel: *mut *mut c_void,
    ) -> *mut OrtStatus {
        let api = &*ort_api;
        let info = KernelInfo::from_ort(api, &*ort_info);
        let user_kernel = T::kernel_create(&info);
        let wrapped_kernel = WrappedKernel { user_kernel, api };

        *kernel = Box::leak(Box::new(wrapped_kernel)) as *mut _ as *mut c_void;
        std::ptr::null_mut()
    }

    unsafe extern "C" fn kernel_compute_fallible<'s, T: CustomOp>(
        op_kernel: *mut c_void,
        context_ptr: *mut OrtKernelContext,
    ) -> *mut OrtStatus
    where
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

        match outputs {
            Ok(outputs) => {
                // Bad hack to use a second context here! The first one is
                // still borrowed immutably at this point :/
                let out_context = context_ptr.as_mut::<'s>().unwrap();
                outputs.write_to_ort(api, out_context);
                return std::ptr::null_mut();
            }
            Err(err) => api.CreateStatus.unwrap()(
                OrtErrorCode_ORT_RUNTIME_EXCEPTION,
                CString::new("").unwrap().into_raw(),
            ),
        }
    }

    unsafe extern "C" fn kernel_destroy<T: CustomOp>(op_kernel: *mut c_void) {
        drop(Box::from_raw(op_kernel as *mut WrappedKernel<T>));
    }

    extern "C" fn get_input_characteristic<'s, T>(
        _op: *const OrtCustomOp,
        index: usize,
    ) -> OrtCustomOpInputOutputCharacteristic
    where
        T: CustomOp,
        <T as CustomOp>::OpInputs<'s>: Inputs<'s>,
    {
        <<T as CustomOp>::OpInputs<'s> as Inputs>::CHARACTERISTICS[index]
    }

    extern "C" fn get_output_characteristic<T>(
        _op: *const OrtCustomOp,
        index: usize,
    ) -> OrtCustomOpInputOutputCharacteristic
    where
        T: CustomOp,
        <T as CustomOp>::OpOutputs: Outputs,
    {
        <<T as CustomOp>::OpOutputs as Outputs>::CHARACTERISTICS[index]
    }

    extern "C" fn get_mem_type_default(_op: *const OrtCustomOp, _index: usize) -> OrtMemType {
        OrtMemType_OrtMemTypeDefault
    }

    extern "C" fn get_variadic_input_homogeneity<'s, T>(
        _op: *const OrtCustomOp,
    ) -> ::std::os::raw::c_int
    where
        T: CustomOp,
        <T as CustomOp>::OpInputs<'s>: Inputs<'s>,
    {
        i32::from(<<T as CustomOp>::OpInputs<'s> as Inputs>::VARIADIC_IS_HOMOGENEOUS)
    }

    extern "C" fn get_variadic_input_min_arity<'s, T>(
        _op: *const OrtCustomOp,
    ) -> ::std::os::raw::c_int
    where
        T: CustomOp,
        <T as CustomOp>::OpInputs<'s>: Inputs<'s>,
    {
        <<T as CustomOp>::OpInputs<'s> as Inputs>::VARIADIC_MIN_ARITY as _
    }

    extern "C" fn get_variadic_output_homogeneity<T>(
        _op: *const OrtCustomOp,
    ) -> ::std::os::raw::c_int
    where
        T: CustomOp,
        <T as CustomOp>::OpOutputs: Outputs,
    {
        i32::from(<<T as CustomOp>::OpOutputs as Outputs>::VARIADIC_IS_HOMOGENEOUS)
    }

    extern "C" fn get_variadic_output_min_arity<T>(_op: *const OrtCustomOp) -> ::std::os::raw::c_int
    where
        T: CustomOp,
        <T as CustomOp>::OpOutputs: Outputs,
    {
        <<T as CustomOp>::OpOutputs as Outputs>::VARIADIC_MIN_ARITY as _
    }

    OrtCustomOp {
        // This is the API version, not the version of the
        // operator. It is currently not clear to me how one defines a
        // version for an operator.
        version: API_VERSION,
        CreateKernel: None, // Some(create_kernel::<T>),
        GetName: Some(get_name::<T>),
        GetExecutionProviderType: Some(get_execution_provider_type::<T>),
        GetInputType: Some(get_input_type::<T>),
        GetInputTypeCount: Some(get_input_type_count::<T>),
        GetOutputType: Some(get_output_type::<T>),
        GetOutputTypeCount: Some(get_output_type_count::<T>),
        KernelCompute: None, // Some(kernel_compute::<T>),,
        KernelDestroy: Some(kernel_destroy::<T>),
        GetInputCharacteristic: Some(get_input_characteristic::<T>),
        GetOutputCharacteristic: Some(get_output_characteristic::<T>),
        GetInputMemoryType: Some(get_mem_type_default),
        GetVariadicInputMinArity: Some(get_variadic_input_min_arity::<T>),
        GetVariadicInputHomogeneity: Some(get_variadic_input_homogeneity::<T>),
        GetVariadicOutputMinArity: Some(get_variadic_output_min_arity::<T>),
        GetVariadicOutputHomogeneity: Some(get_variadic_output_homogeneity::<T>),
        CreateKernelFallible: Some(create_kernel_fallible::<T>), // Some(kernel_compute_fallible::<T>),
        KernelComputeFallible: Some(kernel_compute_fallible::<T>),
    }
}

struct WrappedKernel<T> {
    user_kernel: T,
    api: &'static OrtApi,
}
