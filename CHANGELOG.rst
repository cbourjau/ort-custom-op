.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

0.7.2 (2024-09-03)
------------------

**Bug fix**

- Fix crash when custom operators received input with a zero-sized dimension.


0.7.1 (2024-07-04)
------------------

**Bug fix**

- Fix compilation error on ``linux_aarch64``


0.7.0 (2024-04-29)
------------------

**New feature**

- Add support for output tensors with `i8` and `i16` element types.


0.6.0 (2024-01-06)
------------------

**Breaking changes**

- String input tensors are now passed as `ArrayViewD<'_, &str>` objects rather than `ArrayD<String>` objects.
- The `ort_custom_ops::prelude::KernelInfo.get_attribute_tensors` function now returns owned array objects for numerical data. String tensors are currently not supported.


0.5.1 (2023-11-04)
------------------

**New feature**

- Error messages now include the name of the node that caused the error.


0.5.0 (2023-10-06)
------------------

**Breaking changes**

- Enable error propagation from kernel creation and compute functions.
- Require ``onnxruntime > 1.16``.


0.4.1 (2023-09-14)
------------------

**Bug fix**

- Fix double-free bug when attempting to access a non-existing attribute.

0.4.0 (2023-09-01)
------------------

**New feature**

- Add support for tensor attributes
