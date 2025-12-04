#ifndef INFERENCE_H
#define INFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <iterator>
#include "tokenizer.h"

// ---------------------------
// Structure to hold generation parameters
struct GenerationOptions {
    float temperature = 1.0f;         // Controls randomness; higher -> more random
    int top_k = 50;                   // Keep only top-k tokens
    float top_p = 0.9f;               // Nucleus sampling: keep minimal set with cumulative probability >= top_p
    float repetition_penalty = 1.1f;  // Base penalty for repeating tokens
    float frequency_penalty = 0.5f;   // Additional penalty proportional to frequency
    float presence_penalty = 0.8f;    // Additional penalty if token is present at least once
    int max_tokens = -1;              // If less than 0, then token generation will occur before receiving the eos token.
    unsigned int seed = 0;            // Seed for reproducible random sampling; 0 -> random
    bool softmax = true;              // If !softmax, then the most probable token is taken without applying penalties
    float top_fraction = 0.3f;        // What percentage of sorted raw tokens will be sampled?
};

class CInferenceONNX {
    
    private:

    // Входные размеры
    std::unique_ptr<CTokenizer> tokenizer;
    std::unique_ptr<Ort::Session> pSession;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    int num_layers = 0;
    int num_heads = 0;
    int head_dim = 0;
    int past_seq_len = 0;

    void CreateInputOutputNames();
    void ExtractModelConfig();
    ///////////////

    public:
    CInferenceONNX(const std::string& model_path);


    std::string GetResponse_(const std::string& Request, GenerationOptions options);
    std::string GetResponse(const std::string& Request, GenerationOptions options);

    ~CInferenceONNX();

    private:


};

#endif  // INFERENCE_H