#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np
import sys
from transformers import AutoTokenizer
import time
import os

print("Available providers before session:", ort.get_available_providers())

MODEL_PATH = "../../models/Llama-3.2-3B-Instruct-ONNX/cuda/cuda-fp16/model.onnx"

# --- 1. Create a session ---
providers = ["CUDAExecutionProvider"]
session = ort.InferenceSession(MODEL_PATH, providers=providers)

print("Providers:", session.get_providers())
if "CUDAExecutionProvider" not in session.get_providers():
    print("ERROR: CUDAExecutionProvider not active. Model is running on CPU.")
    sys.exit(1)

# --- 2. Get a list of inputs ---
inputs_info = session.get_inputs()
input_names = [i.name for i in inputs_info]
#print("Model inputs:", input_names)

# --- 2.1. Converting Prompts to Tokens ---
# Turn off multithreading
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained("../../models/Llama-3.2-3B-Instruct-ONNX/cuda/cuda-fp16/")

#prompt="Hello, world"
prompt="The Enigma machine is a cipher device developed and used in the"

for _ in range(60000):
    tokens = []
    tokens.extend(tokenizer.encode(prompt))

# --- 3. Basic mandatory inputs ---
input_ids = np.array([tokens], dtype=np.int64)

attention_mask = np.ones_like(input_ids, dtype=np.int64)

inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
}

print(inputs)

# --- 4. Automatically determine past_key_values ​​---
pkv_inputs = [x for x in input_names if x.startswith("past_key_values.")]

if pkv_inputs:
    #print("Model uses past_key_values")

    # Find the number of layers
    layer_ids = set(int(name.split(".")[1]) for name in pkv_inputs)
    num_layers = max(layer_ids) + 1

    # Let's take an example of a tensor to find out the dimensions
    example_name = "past_key_values.0.key"
    example_input = next(i for i in inputs_info if i.name == example_name)
    shape = example_input.shape 

    batch_size = 1             
    num_heads = shape[1]       
    seq_len = 0                
    head_dim = shape[3]

    #print(f"Detected PKV shapes: layers={num_layers}, heads={num_heads}, head_dim={head_dim}, seq_len={seq_len}")

    # Create empty past_key_values ​​for all layers
    for layer in range(num_layers):
        inputs[f"past_key_values.{layer}.key"] = np.zeros(
            (batch_size, num_heads, seq_len, head_dim), dtype=np.float16
        )
        inputs[f"past_key_values.{layer}.value"] = np.zeros(
            (batch_size, num_heads, seq_len, head_dim), dtype=np.float16
        )
else:
    print("Model does NOT require past_key_values")


"""
# Inference (one token)
outputs = session.run(None, inputs)
print("Outputs received:")
print(outputs)
"""

"""
logits = outputs[0]  # principal tensor, logits for all tokens
# Take the last token
last_logits = logits[0, -1, :]
max_logit = np.max(last_logits)
shifted_logits = last_logits - max_logit
probs = np.exp(shifted_logits) / np.sum(np.exp(shifted_logits))

# Top-5 tokens
top5_idx = np.argsort(probs)[-5:][::-1]  
print("Top-5 tokens and probabilities:")
for idx in top5_idx:
    print(idx, probs[idx], tokenizer.decode([idx]))
"""


# Inference (N tokens)
def sample_top_k_threshold(probs, top_k=9, threshold=0.09):
    """
    probs: Numpy array of probabilities (softmax)
    top_k: how many tokens to take from the top
    threshold: minimum fraction of the maximum probability of top-k
    """
    # 1. Top-k indices
    topk_idx = np.argsort(probs)[-top_k:][::-1]  # от max к min
    topk_probs = probs[topk_idx]
    
    # 2. Select tokens with prob >= threshold * max_prob_in_topk
    max_topk_prob = np.max(topk_probs)
    mask = topk_probs >= threshold * max_topk_prob
    candidates = topk_idx[mask]
    candidate_probs = topk_probs[mask]
    
    # normalization of probabilities among candidates
    candidate_probs = candidate_probs / np.sum(candidate_probs)
    
    #3. Random selection
    return int(np.random.choice(candidates, p=candidate_probs))

generated = []  # we will add tokens here

print("\n\n==============================================================")
print(prompt, end="")

N = 6 # every N-th step will destabilize the inference

for step in range(199):  

    outputs = session.run(None, inputs)
    logits = outputs[0]   # [batch, seq_len, vocab]
    
    # we take logits for the last token
    last_logits = logits[0, -1, :]

    # stabilized softmax
    max_logit = np.max(last_logits)
    shifted = last_logits - max_logit
    probs = np.exp(shifted) / np.sum(np.exp(shifted))

    # select a random token from the top-k
    next_token = sample_top_k_threshold(probs)


    # ---- Warning: condition for "destabilization" ----
    if (step + 1) % N == 0:
        # select the token with the MINIMUM probability
        next_token = int(np.argmin(probs))
    else:
        # normal sampling
        next_token = sample_top_k_threshold(probs)
    # ------------------------------------------------

    generated.append(next_token)

    #print(f"[{step}] picked token:", next_token, tokenizer.decode([next_token]))
    print(tokenizer.decode([next_token]), end="", flush=True)

    # add the token to input_ids (for the next step)
    inputs["input_ids"] = np.concatenate([
        inputs["input_ids"],
        np.array([[next_token]], dtype=np.int64)
    ], axis=1)

    # extend attention mask 
    inputs["attention_mask"] = np.concatenate([
        inputs["attention_mask"],
        np.array([[1]], dtype=np.int64)
    ], axis=1)

    # if the end token is reached, exit
    if next_token == tokenizer.eos_token_id:
        break
