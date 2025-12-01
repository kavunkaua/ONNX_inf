#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>

int main() {
    const std::string model_path =
        "../models/Llama-3.2-3B-Instruct-ONNX/cuda/cuda-fp16/model.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "llama_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // Print inputs
    std::cout << "Inputs:\n";
    std::vector<std::string> input_names;
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        auto name_alloc = session.GetInputNameAllocated(i, allocator);
        std::string name(name_alloc.get());
        input_names.push_back(name);

        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::cout << i << ": " << name << "  Type: " << tensor_info.GetElementType()
                  << "  Shape: ";
        for (auto dim : tensor_info.GetShape()) std::cout << dim << " ";
        std::cout << "\n";
    }

    // Print outputs
    std::cout << "\nOutputs:\n";
    std::vector<std::string> output_names;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        auto name_alloc = session.GetOutputNameAllocated(i, allocator);
        std::string name(name_alloc.get());
        output_names.push_back(name);

        auto type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::cout << i << ": " << name << "  Type: " << tensor_info.GetElementType()
                  << "  Shape: ";
        for (auto dim : tensor_info.GetShape()) std::cout << dim << " ";
        std::cout << "\n";
    }

    // Example inputs
    std::vector<int64_t> input_ids = {1, 2};
    std::vector<int64_t> attention_mask(input_ids.size(), 1);
    std::vector<int64_t> input_shape = {1, (int64_t)input_ids.size()};

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_ids_tensor =
        Ort::Value::CreateTensor<int64_t>(mem_info, input_ids.data(),
                                          input_ids.size(),
                                          input_shape.data(), input_shape.size());

    Ort::Value attention_mask_tensor =
        Ort::Value::CreateTensor<int64_t>(mem_info, attention_mask.data(),
                                          attention_mask.size(),
                                          input_shape.data(), input_shape.size());

    // --- Create past_key_values (float16) ---
    const int num_layers = 28;
    const int num_heads = 8;
    const int head_dim = 128;
    const int past_seq_len = 1;

    std::vector<int64_t> pkv_shape = {1, num_heads, past_seq_len, head_dim};
    size_t pkv_elems = 1LL * num_heads * past_seq_len * head_dim;

    std::vector<Ort::Float16_t> pkv_buffer(num_layers * 2 * pkv_elems, Ort::Float16_t(0.0f));

    std::vector<Ort::Value> past_key_values;
    past_key_values.reserve(num_layers * 2);

    for (int i = 0; i < num_layers * 2; ++i) {
        past_key_values.push_back(
            Ort::Value::CreateTensor<Ort::Float16_t>(
                mem_info,
                pkv_buffer.data() + i * pkv_elems,
                pkv_elems,
                pkv_shape.data(),
                pkv_shape.size()));
    }

    std::cout << "\nInput tensors prepared successfully.\n";

    // -------- Prepare input name pointers ----------
    std::vector<const char*> input_names_ptrs;
    input_names_ptrs.reserve(input_names.size());

    for (auto& s : input_names)
        input_names_ptrs.push_back(s.c_str());

    // --- actual input tensors in proper order ---
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_ids_tensor));
    inputs.push_back(std::move(attention_mask_tensor));

    // add PKV
    for (auto &v : past_key_values)
        inputs.push_back(std::move(v));

    // -------- Prepare output name pointers ----------
    std::vector<const char*> output_names_ptrs;
    for (auto& s : output_names)
        output_names_ptrs.push_back(s.c_str());

    // ============== RUN ================
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names_ptrs.data(),
        inputs.data(),
        inputs.size(),
        output_names_ptrs.data(),
        output_names_ptrs.size()
    );

    std::cout << "Inference OK\n";

    // Output processing
    const float* logits = output_tensors[0].GetTensorMutableData<float>();
    const float* logits2 = output_tensors[1].GetTensorMutableData<float>();
    const float *logits3 = output_tensors[2].GetTensorMutableData<float>();

    // Extracting the shape of a tensor
    auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor_info.GetShape();

    // Output the form
    std::cout << "Output tensor shape: ";
    for (int64_t dim : shape)
    {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // Calculate the size (total number of elements)
    size_t total_size = 1;
    for (int64_t dim : shape)
    {
        total_size *= dim;
    }

    std::cout << "Total size of logits: " << total_size << std::endl;

    // Output results
    std::cout << "Logits (first 9 values):" << std::endl;
    for (int i = 0; i < 9; i++) {
        std::cout << logits[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < 100; i++) {
        std::cout << logits2[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < 100; i++) {
        std::cout << logits3[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}