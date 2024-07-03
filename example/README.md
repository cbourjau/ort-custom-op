Example crate
-------------

This crate demonstrates how to implement various custom operators.
These operators are loaded and used in the Python test cases in `tests/python`.
Building and running these tests requires `cargo` (i.e. the standard rust tool chain), `onnxruntime>=1.16` and `pytest`.

Execute the following at the root of this repository to build the shared
library and to run the python-defined tests:

```python
cargo b && pytest tests/python -s
```
