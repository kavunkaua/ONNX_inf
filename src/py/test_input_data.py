#!/usr/bin/env python3
import onnx
import onnxruntime as ort

# Model path
model_path = '../../models/Llama-3.2-3B-Instruct-ONNX/cuda/cuda-fp16/model.onnx'

# Check the model file directly without loading it into memory
onnx.checker.check_model(model_path)

# Set the execution provider to CUDA (GPU)
providers = ['CUDAExecutionProvider']

# Initialize the ONNX Runtime session with the GPU provider
session = ort.InferenceSession(model_path, providers=providers)

# Print input details
for i, input in enumerate(session.get_inputs()):
    print(f"Input {i}: name={input.name}, shape={input.shape}, type={input.type}")
