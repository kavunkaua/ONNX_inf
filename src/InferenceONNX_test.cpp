#include "InferenceONNX.h"
#include <unistd.h>

int main() {

CInferenceONNX inf_onnx("../models/Llama-3.2-3B-Instruct-ONNX/cuda/cuda-fp16/");

GenerationOptions options;
options.temperature = 0.8f;
options.top_k = 50;
options.top_p = 0.9f;
options.repetition_penalty = 1.2f;
options.frequency_penalty = 0.4f;
options.presence_penalty = 0.6f;
options.seed = 12345;

inf_onnx.GetResponse("Hello, LLM!", options);

sleep(9);
return 0;
}
