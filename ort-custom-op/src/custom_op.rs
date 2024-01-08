use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::unimplemented;

use crate::api::{KernelInfo, API_VERSION};
use crate::bindings::{
    ONNXTensorElementDataType, OrtApi, OrtCustomOp, OrtCustomOpInputOutputCharacteristic,
    OrtErrorCode_ORT_RUNTIME_EXCEPTION, OrtKernelContext, OrtKernelInfo, OrtMemType,
    OrtMemType_OrtMemTypeDefault, OrtStatus,
};
pub use crate::outputs::Outputs;
pub use crate::value::TryFromValues;
use crate::value::{TryIntoInputTuple, Value};

/// Trait defining the behavior of a custom operator.
pub trait CustomOp<'s> {
    type KernelCreateError;
    type ComputeError;
    const NAME: &'static str;

    type OpInputs: TryFromValues<'s>;
    type OpOutputs: Outputs;

    fn kernel_create(info: &KernelInfo) -> Result<Self, Self::KernelCreateError>
    where
        Self: Sized;
    fn kernel_compute(&self, inputs: Self::OpInputs)
        -> Result<Self::OpOutputs, Self::ComputeError>;
}

/// Function to build static instances of `OrtCustomOp`.
///
/// These static objects are registered using the
/// `create_custom_op_domain` function.
pub const fn build<'ctx, 'data, T>() -> OrtCustomOp
where
    'ctx: 'data,
    T: for<'s> CustomOp<'s>,
    <T as CustomOp<'data>>::KernelCreateError: std::fmt::Display,
    <T as CustomOp<'data>>::ComputeError: std::fmt::Display,
{
    extern "C" fn get_name<'s, T>(_op: *const OrtCustomOp) -> *const c_char
    where
        T: CustomOp<'s>,
    {
        CString::new(T::NAME).unwrap().into_raw()
    }

    extern "C" fn get_execution_provider_type(_op: *const OrtCustomOp) -> *const c_char {
        b"CPUExecutionProvider\0".as_ptr() as *const _
    }

    extern "C" fn get_input_type<'s, T>(
        _op: *const OrtCustomOp,
        index: usize,
    ) -> ONNXTensorElementDataType
    where
        T: CustomOp<'s>,
    {
        // <<T as CustomOp>::OpInputs as Inputs>::INPUT_TYPES[index].to_ort_encoding()
        unimplemented!()
    }

    extern "C" fn get_input_type_count<'s, T>(_op: *const OrtCustomOp) -> usize
    where
        T: CustomOp<'s>,
    {
        // <<T as CustomOp>::OpInputs as Inputs>::CHARACTERISTICS.len()
        unimplemented!()
    }

    extern "C" fn get_output_type<'s, T>(
        _op: *const OrtCustomOp,
        index: usize,
    ) -> ONNXTensorElementDataType
    where
        T: CustomOp<'s>,
    {
        // <<T as CustomOp>::OpOutputs as Outputs>::OUTPUT_TYPES[index].to_ort_encoding()
        unimplemented!()
    }

    extern "C" fn get_output_type_count<'s, T>(_op: *const OrtCustomOp) -> usize
    where
        T: CustomOp<'s>,
    {
        <<T as CustomOp>::OpOutputs as Outputs>::OUTPUT_TYPES.len()
    }

    unsafe extern "C" fn create_kernel_fallible<'s, T>(
        _ort_op: *const OrtCustomOp,
        ort_api: *const OrtApi,
        ort_info: *const OrtKernelInfo,
        kernel: *mut *mut c_void,
    ) -> *mut OrtStatus
    where
        T: CustomOp<'s>,
        <T as CustomOp<'s>>::KernelCreateError: std::fmt::Display,
    {
        let api = &*ort_api;
        let info = KernelInfo::from_ort(api, &*ort_info);
        let user_kernel = match T::kernel_create(&info) {
            Ok(kernel) => kernel,
            Err(err) => {
                *kernel = std::ptr::null_mut();
                // msg is copied inside `CreateStatus`
                let msg = CString::new(format!("{}: {}", T::NAME, err)).unwrap();
                return api.CreateStatus.unwrap()(OrtErrorCode_ORT_RUNTIME_EXCEPTION, msg.as_ptr());
            }
        };
        let wrapped_kernel = WrappedKernel { user_kernel, api };

        *kernel = Box::leak(Box::new(wrapped_kernel)) as *mut _ as *mut c_void;
        std::ptr::null_mut()
    }

    unsafe extern "C" fn kernel_compute_fallible<'ctx, T>(
        op_kernel: *mut c_void,
        context_ptr: *mut OrtKernelContext,
    ) -> *mut OrtStatus
    where
        T: for<'s> CustomOp<'s>,
        //for<'s> <T as CustomOp<'s>>::ComputeError: std::fmt::Display,
    {
        let wrapped_kernel: &mut WrappedKernel<T> = &mut *(op_kernel as *mut _);
        let kernel = &wrapped_kernel.user_kernel;
        let api = &wrapped_kernel.api;
        let context = context_ptr.as_ref::<'ctx>().unwrap();

        let input_values = vec![];
        // context.get_input_values(api).unwrap();
        let input_values = input_values.as_slice();
        let tuple = TryFromValues::try_from_values(input_values).unwrap();
        let outputs = kernel.kernel_compute(tuple);

        // let outputs = {
        //     let inputs = <T::OpInputs as Inputs>::from_ort(api, context);
        //     kernel.kernel_compute(inputs)
        // };

        match outputs {
            Ok(outputs) => {
                // Bad hack to use a second context here! The first one is
                // still borrowed immutably at this point :/
                let out_context = context_ptr.as_mut::<'ctx>().unwrap();
                outputs.write_to_ort(api, out_context);
                return std::ptr::null_mut();
            }
            Err(_err) => {
                let msg = CString::new(format!("{}: print error", T::NAME)).unwrap();
                return api.CreateStatus.unwrap()(OrtErrorCode_ORT_RUNTIME_EXCEPTION, msg.as_ptr());
            }
        }
    }

    unsafe extern "C" fn kernel_destroy<'s, T>(op_kernel: *mut c_void)
    where
        T: CustomOp<'s>,
    {
        drop(Box::from_raw(op_kernel as *mut WrappedKernel<T>));
    }

    extern "C" fn get_input_characteristic<'s, T>(
        _op: *const OrtCustomOp,
        index: usize,
    ) -> OrtCustomOpInputOutputCharacteristic
    where
        T: CustomOp<'s>,
        // <T as CustomOp>::OpInputs<'s>: Inputs<'s>,
    {
        // <<T as CustomOp>::OpInputs<'s> as Inputs>::CHARACTERISTICS[index]
        unimplemented!()
    }

    extern "C" fn get_output_characteristic<'s, T>(
        _op: *const OrtCustomOp,
        index: usize,
    ) -> OrtCustomOpInputOutputCharacteristic
    where
        T: CustomOp<'s>,
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
        T: CustomOp<'s>,
    {
        unimplemented!()
        // i32::from(<<T as CustomOp>::OpInputs<'s> as Inputs>::VARIADIC_IS_HOMOGENEOUS)
    }

    extern "C" fn get_variadic_input_min_arity<'s, T>(
        _op: *const OrtCustomOp,
    ) -> ::std::os::raw::c_int
    where
        T: CustomOp<'s>,
        // <T as CustomOp>::OpInputs<'s>: Inputs<'s>,
    {
        unimplemented!()
        // <<T as CustomOp>::OpInputs<'s> as Inputs>::VARIADIC_MIN_ARITY as _
    }

    extern "C" fn get_variadic_output_homogeneity<'s, T>(
        _op: *const OrtCustomOp,
    ) -> ::std::os::raw::c_int
    where
        T: CustomOp<'s>,
    {
        i32::from(<<T as CustomOp>::OpOutputs as Outputs>::VARIADIC_IS_HOMOGENEOUS)
    }

    extern "C" fn get_variadic_output_min_arity<'s, T>(
        _op: *const OrtCustomOp,
    ) -> ::std::os::raw::c_int
    where
        T: CustomOp<'s>,
        <T as CustomOp<'s>>::KernelCreateError: std::fmt::Display,
        <T as CustomOp<'s>>::ComputeError: std::fmt::Display,
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
        GetExecutionProviderType: Some(get_execution_provider_type),
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
        CreateKernelV2: Some(create_kernel_fallible::<T>),
        KernelComputeV2: Some(kernel_compute_fallible::<T>),
    }
}

struct WrappedKernel<T> {
    user_kernel: T,
    api: &'static OrtApi,
}
