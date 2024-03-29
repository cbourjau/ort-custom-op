#! /bin/bash

# Create a new bindings file
~/.cargo/bin/bindgen \
    --allowlist-type "OrtApi.*" \
    --no-copy "Ort.*" \
    -o src/bindings.rs \
    ./onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h
