#include "InferenceONNX.h"
#include <unistd.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <cmath>
#include <iostream>



// ---------------------------
// Apply temperature to logits
std::vector<float> apply_temperature(const std::vector<float>& logits, float temperature) {
    std::vector<float> adjusted(logits.size());
    for (size_t i = 0; i < logits.size(); i++)
        adjusted[i] = logits[i] / temperature;
    return adjusted;
}

// ---------------------------
// Apply repetition penalty
void apply_repetition_penalty(std::vector<float>& logits, const std::unordered_set<int>& history, float penalty) {
    for (int token_id : history) {
        if (token_id < logits.size()) {
            if (logits[token_id] > 0)
                logits[token_id] /= penalty;
            else
                logits[token_id] *= penalty;
        }
    }
}

// Apply repetition, frequency and presence penalties
void apply_complex_penalty(std::vector<float>& logits,
                           const std::unordered_map<int, int>& token_history,
                           const GenerationOptions& options)
{
    for (const auto& [token_id, count] : token_history) {
        if (token_id >= logits.size()) continue;

        // Base repetition penalty
        if (logits[token_id] > 0)
            logits[token_id] /= options.repetition_penalty;
        else
            logits[token_id] *= options.repetition_penalty;

        // Frequency penalty: subtract proportional to how many times token appeared
        logits[token_id] -= options.frequency_penalty * count;

        // Presence penalty: subtract fixed amount if token appeared at least once
        if (count > 0)
            logits[token_id] -= options.presence_penalty;
    }
}

// ---------------------------
// Top-k filtering
void top_k_filter(std::vector<float>& logits, int k) {
    if (k <= 0 || k >= logits.size()) return;

    std::vector<size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                     [&](size_t a, size_t b) { return logits[a] > logits[b]; });

    float min_topk = logits[indices[k-1]];
    for (float &logit : logits)
        if (logit < min_topk) logit = -1e9f;
}

// ---------------------------
// Top-p (nucleus) filtering
void top_p_filter(std::vector<float>& logits, float p) {
    if (p <= 0.0f || p >= 1.0f) return;

    std::vector<size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return logits[a] > logits[b]; });

    std::vector<float> probs(logits.size());
    float max_logit = logits[indices[0]];
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (float &pval : probs) pval /= sum;

    float cumulative = 0.0f;
    for (size_t idx : indices) {
        cumulative += probs[idx];
        if (cumulative > p)
            logits[idx] = -1e9f;
    }
}

// ---------------------------
// Sample a token from logits using softmax
int sample_token(const std::vector<float>& logits, unsigned int seed = 0) {
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;

    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (float &p : probs) p /= sum;

    std::mt19937 gen(seed ? seed : std::random_device{}());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return dist(gen);
}

// ---------------------------
// Main function to get the next token
int get_next_token_complex(const std::vector<float>& logits,
                           const std::unordered_map<int,int>& token_history,
                           const GenerationOptions& options)
{
    std::vector<float> local_logits = logits;

    // 1. Temperature
    local_logits = apply_temperature(local_logits, options.temperature);

    // 2. Complex penalties
    apply_complex_penalty(local_logits, token_history, options);

    // 3. Top-k
    top_k_filter(local_logits, options.top_k);

    // 4. Top-p
    top_p_filter(local_logits, options.top_p);

    // 5. Sample token
    return sample_token(local_logits, options.seed);
}

////////////////////////////////////////////////////////////////////////////////////

CTokenizer* CreateTokenizer(const std::string& model_path) {
    return new CTokenizer(model_path);
}

// Глобальный, создаётся один раз на всё приложение
static Ort::Env g_env(ORT_LOGGING_LEVEL_WARNING, "inference_onnx");

std::unique_ptr<Ort::Session> CreateSession(const std::string& model_path) {

    std::string model_path_name =
        (model_path.back() == '/')
        ? model_path + "model.onnx"
        : model_path + "/model.onnx";

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ✔ создаём session в unique_ptr
    auto session = std::make_unique<Ort::Session>(
        g_env,
        model_path_name.c_str(),
        session_options
    );

    Ort::AllocatorWithDefaultOptions allocator;

    ////////////////////////////////////////////////////////////////////////////
    // Inputs

    for (size_t i = 0; i < session->GetInputCount(); ++i) {
        auto name_alloc = session->GetInputNameAllocated(i, allocator);
        std::string name(name_alloc.get());
        auto type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Outputs

    for (size_t i = 0; i < session->GetOutputCount(); ++i) {
        auto name_alloc = session->GetOutputNameAllocated(i, allocator);
        std::string name(name_alloc.get());
        auto type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    }

    return session;
}

void CInferenceONNX::CreateInputOutputNames()
{
     // Print inputs
    std::cout << "Inputs:\n";
    //std::vector<std::string> input_names;
    input_names.clear();
    for (size_t i = 0; i < pSession->GetInputCount(); ++i) {
        auto name_alloc = pSession->GetInputNameAllocated(i, allocator);
        std::string name(name_alloc.get());
        input_names.push_back(name);

        auto type_info = pSession->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::cout << i << ": " << name << "  Type: " << tensor_info.GetElementType()
                  << "  Shape: ";
        for (auto dim : tensor_info.GetShape()) std::cout << dim << " ";
        std::cout << "\n";
    }

    // Print outputs
    std::cout << "\nOutputs:\n";
    //std::vector<std::string> output_names;
    output_names.clear();
    for (size_t i = 0; i < pSession->GetOutputCount(); ++i) {
        auto name_alloc = pSession->GetOutputNameAllocated(i, allocator);
        std::string name(name_alloc.get());
        output_names.push_back(name);

        auto type_info = pSession->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::cout << i << ": " << name << "  Type: " << tensor_info.GetElementType()
                  << "  Shape: ";
        for (auto dim : tensor_info.GetShape()) std::cout << dim << " ";
        std::cout << "\n";
    }   
}


void CInferenceONNX::ExtractModelConfig() {

    Ort::AllocatorWithDefaultOptions allocator;

    num_layers   = 0;
    num_heads    = 0;
    head_dim     = 0;
    past_seq_len = 0;

    // -----------------------------
    // 1. Определяем num_layers
    // Считаем все past_key_values.*.key и past_key_values.*.value
    // Каждый слой задаётся двумя входами: key + value
    // -----------------------------
    size_t past_count = 0;

    for (size_t i = 0; i < pSession->GetInputCount(); ++i) {
        std::string name = pSession->GetInputNameAllocated(i, allocator).get();
        if (name.find("past_key_values") != std::string::npos)
            past_count++;
    }

    if (past_count > 0)
        num_layers = past_count / 2;

    // -----------------------------
    // 2. Берём первый past_key_values.*.key для извлечения остальных параметров
    // Формат: [batch, num_heads, past_seq_len, head_dim]
    // -----------------------------
    for (size_t i = 0; i < pSession->GetInputCount(); ++i) {

        std::string name = pSession->GetInputNameAllocated(i, allocator).get();

        // Ищем ключи слоёв
        if (name.find("past_key_values") == std::string::npos ||
            name.find(".key") == std::string::npos)
            continue;

        auto type_info = pSession->GetInputTypeInfo(i);
        if (type_info.GetONNXType() != ONNX_TYPE_TENSOR)
            continue;

        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        // Ожидаем 4D: [batch, num_heads, past_seq_len, head_dim]
        if (shape.size() != 4)
            continue;

        num_heads    = (shape[1] > 0) ? (int)shape[1] : 0;
        past_seq_len = (shape[2] > 0) ? (int)shape[2] : 0;
        head_dim     = (shape[3] > 0) ? (int)shape[3] : 0;

        break; // хватит одного слоя
    }

    // Отладка
    std::cout << "Model config: "
              << "num_layers=" << num_layers
              << ", num_heads=" << num_heads
              << ", head_dim=" << head_dim
              << ", past_seq_len=" << past_seq_len
              << std::endl;
}

CInferenceONNX::CInferenceONNX(const std::string& model_path) : tokenizer(CreateTokenizer(model_path)), pSession(CreateSession(model_path))
{
    CreateInputOutputNames();
    ExtractModelConfig();
}

template <typename T>
void PrintTensor(const Ort::Value& tensor) {
    // Extract information about the tensor type and shape
    auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor_info.GetShape();

    // Extract data
    const T* data = tensor.GetTensorData<T>();

    // Print the tensor shape
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    // Print data in Python style
    size_t total_size = tensor_info.GetElementCount();
    std::cout << "Data: [";
    for (size_t i = 0; i < total_size; ++i) {
        std::cout << static_cast<float>(data[i]) << (i < total_size - 1 ? ", " : "");
    }
    std::cout << "]\n";
}

std::string CInferenceONNX::GetResponse(const std::string& Request)
{
    std::string Response;

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
    auto output_tensors = pSession->Run(
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

    return Response;
}

CInferenceONNX::~CInferenceONNX()
{

}