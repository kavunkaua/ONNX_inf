#!/usr/bin/env python3

from huggingface_hub import snapshot_download

# Repository name
repo_id = "onnx-community/Llama-3.2-3B-Instruct-ONNX"

# Specify the path to save files
local_dir = "../../models/Llama-3.2-3B-Instruct-ONNX"

# Download all files from the repository
snapshot_download(repo_id=repo_id, local_dir=local_dir)

print(f"All model files are saved in the folder: {local_dir}")