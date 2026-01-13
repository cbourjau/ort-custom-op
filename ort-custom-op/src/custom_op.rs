use std::ffi::CString;
use std::os::raw::{c_char, c_void};

use crate::api::{API_VERSION, KernelInfo};
use crate::bindings::{
    ONNXTensorElementDataType, OrtApi, OrtCustomOp, OrtCustomOpInputOutputCharacteristic,
    OrtErrorCode_ORT_RUNTIME_EXCEPTION, OrtKernelContext, OrtKernelInfo, OrtMemType,
    OrtMemType_OrtMemTypeDefault, OrtStatus,
};
pub use crate::inputs::Inputs;
pub use crate::outputs::Outputs;

/// Trait defining the behavior of a custom operator.
pub trait CustomOp {
    /// Error type for the kernel creation
    type KernelCreateError;
    /// Error type of the compute operation
    type ComputeError;
    /// Name of the operator
    const NAME: &'static str;
    /// Minimum number of variadic inputs. Any non-zero value requires
    /// that the last input is variadic.
    const VARIADIC_MIN_ARITY: usize = 0;

    type OpInputs<'s>: Inputs<'s>;
    type OpOutputs: Outputs;

    /// Set up state later used in compute calls. Called once per session.
    fn kernel_create(info: &KernelInfo) -> Result<Self, Self::KernelCreateError>
    where
        Self: Sized;

    fn kernel_compute(
        &self,
        inputs: Self::OpInputs<'_>,
    ) -> Result<Self::OpOutputs, Self::ComputeError>;
}

/// Function to build static instances of [`OrtCustomOp`].
///
/// The produced static object can be registered using the
/// [`crate::prelude::create_custom_op_domain`] function.
pub const fn build<T>() -> OrtCustomOp
where
    T: CustomOp,
    T::KernelCreateError: std::fmt::Display,
    T::ComputeError: std::fmt::Display,
{
    // This `const` function is meant to be called at compile
    // time. Therefore, the below panic is a compilation error from
    // the user perspective.
    if T::VARIADIC_MIN_ARITY > 0 && <T::OpInputs<'_>>::VARIADIC_IS_HOMOGENEOUS.is_none() {
        panic!("Specified non-zero `MIN_VARIADIC_ARITY` but the operators inputs are not variadic.")
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

/// Conditionally return with a non-null `OrtStatus` pointer from a result.
///
/// Return if the provided result is the error variant. Otherwise,
/// unwrap the `Ok` value.
macro_rules! bail_on_error {
    ($api:expr, $res:expr) => {
        match $res {
            Ok(val) => val,
            Err(err) => {
                // msg is copied inside `CreateStatus`; no need to leak
                let msg = CString::new(format!("{}: {}", T::NAME, err)).unwrap();
                return unsafe {
                    $api.CreateStatus.unwrap()(OrtErrorCode_ORT_RUNTIME_EXCEPTION, msg.as_ptr())
                };
            }
        }
    };
}

/// Helper struct which contains a reference to the api object. We use
/// it to shuttle a reference to the Api object from the
/// kernel-creation time to the compute method which would otherwise
/// have no such reference.
struct WrappedKernel<T> {
    user_kernel: T,
    api: &'static OrtApi,
}

extern "C" fn get_name<T>(_op: *const OrtCustomOp) -> *const c_char
where
    T: CustomOp,
{
    CString::new(T::NAME).unwrap().into_raw()
}

extern "C" fn get_execution_provider_type(_op: *const OrtCustomOp) -> *const c_char {
    c"CPUExecutionProvider".as_ptr()
}

extern "C" fn get_input_type<T>(_op: *const OrtCustomOp, index: usize) -> ONNXTensorElementDataType
where
    T: CustomOp,
{
    <T::OpInputs<'_>>::tensor_data_type(index)
        .unwrap_or_else(|| panic!("Input '{}' is not a tensor", index))
}

extern "C" fn get_input_type_count<T>(_op: *const OrtCustomOp) -> usize
where
    T: CustomOp,
{
    // A possibly variadic input counts as a single input
    let is_variadic = <T::OpInputs<'_>>::VARIADIC_IS_HOMOGENEOUS.is_some();
    <T::OpInputs<'_>>::NUM_POSITIONAL + if is_variadic { 1 } else { 0 }
}

extern "C" fn get_output_type<T>(_op: *const OrtCustomOp, index: usize) -> ONNXTensorElementDataType
where
    T: CustomOp,
{
    T::OpOutputs::OUTPUT_TYPES[index].to_ort_encoding()
}

extern "C" fn get_output_type_count<T>(_op: *const OrtCustomOp) -> usize
where
    T: CustomOp,
{
    // A possibly variadic output counts as a single output
    <T::OpOutputs as Outputs>::OUTPUT_TYPES.len()
}

unsafe extern "C" fn create_kernel_fallible<T>(
    _ort_op: *const OrtCustomOp,
    ort_api: *const OrtApi,
    ort_info: *const OrtKernelInfo,
    kernel: *mut *mut c_void,
) -> *mut OrtStatus
where
    T: CustomOp,
    T::KernelCreateError: std::fmt::Display,
{
    let api = unsafe { &*ort_api };
    let info = KernelInfo::from_ort(api, unsafe { &*ort_info });
    let user_kernel = bail_on_error!(api, T::kernel_create(&info));
    let wrapped_kernel = WrappedKernel { user_kernel, api };

    // Kernel is later destroyed in `kernel_destroy`
    unsafe {
        *kernel = Box::leak(Box::new(wrapped_kernel)) as *mut _ as *mut c_void;
    }
    std::ptr::null_mut()
}

unsafe extern "C" fn kernel_compute_fallible<T>(
    op_kernel: *mut c_void,
    context_ptr: *mut OrtKernelContext,
) -> *mut OrtStatus
where
    T: CustomOp,
    T::ComputeError: std::fmt::Display,
{
    let WrappedKernel::<T> { user_kernel, api } = unsafe { &mut *(op_kernel as *mut _) };

    let context = unsafe { context_ptr.as_mut::<'_>() }.unwrap();
    let outputs = {
        let bufs = bail_on_error!(api, context.get_input_values(api));
        // Owned buffers
        let bufs: Vec<_> = bufs
            .iter()
            .map(|el| el.as_ref().map(|some_buf| some_buf.normalize_buffers()))
            .collect();
        // Create arrays borrowing from owned buffers
        let input_values: anyhow::Result<Vec<Option<_>>> = bufs
            .iter()
            .map(|el| el.as_ref().map(|some_buf| some_buf.as_value()).transpose())
            .collect();
        let input_values = bail_on_error!(api, input_values);
        let tuple = bail_on_error!(api, T::OpInputs::try_from_values(input_values));
        bail_on_error!(api, user_kernel.kernel_compute(tuple))
    };

    outputs.write_to_ort(api, context);
    std::ptr::null_mut()
}

unsafe extern "C" fn kernel_destroy<T>(op_kernel: *mut c_void)
where
    T: CustomOp,
{
    drop(unsafe { Box::from_raw(op_kernel as *mut WrappedKernel<T>) });
}

extern "C" fn get_input_characteristic<T>(
    _op: *const OrtCustomOp,
    index: usize,
) -> OrtCustomOpInputOutputCharacteristic
where
    T: CustomOp,
{
    <T::OpInputs<'_>>::characteristic(index)
}

extern "C" fn get_output_characteristic<T>(
    _op: *const OrtCustomOp,
    index: usize,
) -> OrtCustomOpInputOutputCharacteristic
where
    T: CustomOp,
{
    T::OpOutputs::CHARACTERISTICS[index]
}

extern "C" fn get_mem_type_default(_op: *const OrtCustomOp, _index: usize) -> OrtMemType {
    OrtMemType_OrtMemTypeDefault
}

extern "C" fn get_variadic_input_homogeneity<T>(_op: *const OrtCustomOp) -> ::std::os::raw::c_int
where
    T: CustomOp,
{
    i32::from(
        <T::OpInputs<'_>>::VARIADIC_IS_HOMOGENEOUS
            .expect("'get_variadic_input_homogeneity' was called for operator with fixed arity."),
    )
}

extern "C" fn get_variadic_input_min_arity<T>(_op: *const OrtCustomOp) -> ::std::os::raw::c_int
where
    T: CustomOp,
{
    T::VARIADIC_MIN_ARITY as _
}

extern "C" fn get_variadic_output_homogeneity<T>(_op: *const OrtCustomOp) -> ::std::os::raw::c_int
where
    T: CustomOp,
{
    i32::from(T::OpOutputs::VARIADIC_IS_HOMOGENEOUS)
}

extern "C" fn get_variadic_output_min_arity<T>(_op: *const OrtCustomOp) -> ::std::os::raw::c_int
where
    T: CustomOp,
{
    T::OpOutputs::VARIADIC_MIN_ARITY as _
}
