#!/usr/bin/env python3

from transformers import AutoTokenizer
import time
import os

# Enable multithreading for the tokenizer
# os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Turn off multithreading
os.environ["TOKENIZERS_PARALLELISM"] = "false"


tokenizer = AutoTokenizer.from_pretrained("../../models/Llama-3.2-3B-Instruct-ONNX/cuda/cuda-fp16/")
print("Start")
start_time = time.perf_counter()
for _ in range(60000):
    tokens = []
    tokens.extend(tokenizer.encode("Hello World! String for tokenizer test :-)"))
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("Time: {:.1f} s".format(elapsed_time))

print(tokens)
print(tokenizer.decode(tokens))

# Decode two specific IDs into a string
# decoded_string = tokenizer.decode([235])
# print(decoded_string)




