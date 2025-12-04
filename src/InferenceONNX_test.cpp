#include "InferenceONNX.h"
#include <unistd.h>

int main() {

CInferenceONNX inf_onnx("../models/Llama-3.2-3B-Instruct-ONNX/cuda/cuda-fp16/");

GenerationOptions options;
options.temperature = .7f;
options.top_k = 19;
options.top_p = 0.96f;
options.repetition_penalty = 1.2f;
options.frequency_penalty = 0.4f;
options.presence_penalty = 0.6f;
options.max_tokens = 180;
options.seed = 123459;
options.softmax = true;
options.top_fraction = 0.01;

inf_onnx.GetResponse("The Enigma machine is a cipher device developed and used ", options);


return 0;
}
