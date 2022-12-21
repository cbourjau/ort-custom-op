Proof of concept for writing custom operators for onnxruntime
=============================================================

The usage of the library is demonstrated in `lib.rs`. Everything else
is just making that machinery work but should not concern the user
further.

The core idea (at the moment) is that the user has to implement the
`CustomOp` trait for a struct. This should then make it trivial to
build the objects and APIs expected by onnxruntime.

The "test case" demonstrated in `lib.rs` is building two custom
operators. An ONNX file using these operators is also part of this
library. Testing the created shared library requires cargo (i.e. the
standard rust tool chain), `onnxruntime` and `pytest`.

Run the following at the root of this repository to build the shared
library and run the python-defined tests:

```python
cargo b && pytest tests/python -s
```


