#! /bin/bash

# Create a new bindings file
bindgen \
    --allowlist-type "OrtApi.*" \
    --no-copy "Ort.*" \
    -o src/bindings.rs \
    ./c-api-headers/onnxruntime_c_api.h 
