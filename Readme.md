Proof of concept for writing custom operators for onnxruntime
=============================================================

Custom operators can be made available to the onnxruntime by creating a shared library with a standardized API.
Directly interfacing with the API and types exposed by onnxruntime is quite cumbersome and error prone, though.
This project provides abstractions that make this interfacing much easier and safe.

Each custom operator is an individual type which implements the `CustomOp` trait. Types which implement that trait can be `build` into static objects which are in turn exposed to the onnxruntime.

The `example` crate demonstrates how to implement various custom operators.
These operators are loaded and used in the Python test cases in `tests/python`.
Building and running these tests requires `cargo` (i.e. the standard rust tool chain), `onnxruntime` and `pytest`.

Execute the following at the root of this repository to build the shared
library and to run the python-defined tests:

```python
cargo b && pytest tests/python -s
```


