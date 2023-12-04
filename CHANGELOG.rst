.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

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
