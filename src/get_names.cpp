#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>

int main() {

    const std::string model_path = "../models/Llama-3.2-3B-Instruct-ONNX/cuda/cuda-fp16/model.onnx";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "llama_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    //session_options.SetLogSeverityLevel(0);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // Print the inputs
    std::cout << "Inputs:\n";
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        auto name_alloc = session.GetInputNameAllocated(i, allocator);
        std::string name(name_alloc.get());
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::cout << i << ": " << name << "  Type: " << tensor_info.GetElementType() << "  Shape: ";
        for (auto dim : tensor_info.GetShape()) std::cout << dim << " ";
        std::cout << "\n";
    }

    // Print the outputs
    std::cout << "\nOutputs:\n";
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        auto name_alloc = session.GetOutputNameAllocated(i, allocator);
        std::string name(name_alloc.get());
        auto type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::cout << i << ": " << name << "  Type: " << tensor_info.GetElementType() << "  Shape: ";
        for (auto dim : tensor_info.GetShape()) std::cout << dim << " ";
        std::cout << "\n";
    }

    // Example of creating input_ids and attention_mask
    std::vector<int64_t> input_ids = {1, 2};
    std::vector<int64_t> attention_mask(input_ids.size(), 1);
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()
    );
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size()
    );

    // Create empty past_key_values ​​(float16)
    int num_layers = 28;
    int num_heads = 8;
    int head_dim = 128;
    int past_seq_len = 1;
    size_t pkv_elems = 1 * num_heads * past_seq_len * head_dim;
    std::vector<Ort::Float16_t> pkv_buffer(num_layers * 2 * pkv_elems, Ort::Float16_t(0.0f));
    std::vector<Ort::Value> past_key_values;
    past_key_values.reserve(num_layers * 2);
    for (int i = 0; i < num_layers * 2; ++i) {
        past_key_values.push_back(
            Ort::Value::CreateTensor<Ort::Float16_t>(
                mem_info,
                pkv_buffer.data() + i * pkv_elems,
                pkv_elems,
                std::vector<int64_t>{1, num_heads, past_seq_len, head_dim}.data(),
                4
            )
        );
    }

    std::cout << "\nInput tensors prepared successfully.\n";

    return 0;
}



