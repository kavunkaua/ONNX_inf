#include "InferenceONNX.h"
#include <unistd.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <cmath>
#include <iostream>

struct LogitToken {
    float logit;
    int token_id;
    float prob; 
};

struct PCG32
{
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc   = 0xda3e39cb94b95bdbULL;

    void seed(uint64_t s)
    {
        state = s + 0x853c49e6748fea9bULL;
        inc   = 0xda3e39cb94b95bdbULL;
        random_uint(); // прогоняем
    }

    uint32_t random_uint()
    {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + (inc | 1);
        uint32_t xorshifted = ((old >> 18u) ^ old) >> 27u;
        uint32_t rot = old >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    float random_float()  // [0..1)
    {
        return (random_uint() >> 8) * (1.0f / 16777216.0f);
    }
};

void compute_softmax(std::vector<LogitToken> &v)
{
    float max_log = -1e30f;
    for (auto &lt : v)
        max_log = std::max(max_log, lt.logit);

    float sum = 0.0f;
    for (auto &lt : v)
    {
        lt.prob = std::exp(lt.logit - max_log);
        sum += lt.prob;
    }

    for (auto &lt : v)
        lt.prob /= sum;
}

// Apply repetition + frequency + presence penalties
void apply_penalties(std::vector<LogitToken> &v,
                     const std::unordered_map<int,int> &history,
                     float repetition_penalty,
                     float frequency_penalty,
                     float presence_penalty)
{
    for (auto &lt : v)
    {
        auto it = history.find(lt.token_id);
        if (it != history.end())
        {
            if (repetition_penalty != 1.0f)
            {
                if (lt.logit > 0) lt.logit /= repetition_penalty;
                else lt.logit *= repetition_penalty;
            }

            lt.logit -= frequency_penalty * it->second;

            if (presence_penalty > 0.0f)
                lt.logit -= presence_penalty;
        }
    }
}

// Top-k filtering
void top_k(std::vector<LogitToken>& logits, int k)
{
    int n = logits.size();
    k = std::min(k, n);

    /*std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::nth_element(idx.begin(), idx.begin() + k - 1, idx.end(),
                     [&](int a, int b){ return logits[a].logit > logits[b].logit; });

    float threshold = logits[idx[k - 1]].logit;*/

    float threshold = logits[k-1].logit;

    std::vector<LogitToken> topk;
    for (int i = 0; i < n; i++)
    {
        if (logits[i].logit >= threshold)
            topk.push_back({logits[i].logit, logits[i].token_id, logits[i].prob});
    }

    logits = topk;
}

// Top-p (nucleus sampling)
void top_p(std::vector<LogitToken> &logits, float p)
{
    int n = logits.size();
    if (p <= 0.0f || p >= 1.0f || n == 0)
        return;

    // Сортируем логиты по убыванию
    std::sort(logits.begin(), logits.end(),
              [](const LogitToken &a, const LogitToken &b){ return a.logit > b.logit; });

    // Вычисляем softmax для нормализации
    float max_log = logits[0].logit;
    float sum = 0.0f;
    std::vector<float> probs(n);
    for (int i = 0; i < n; i++)
    {
        probs[i] = std::exp(logits[i].logit - max_log);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++)
        probs[i] /= sum;

    // Кумулятивная сумма и отсечение
    float cumulative = 0.0f;
    int cutoff_index = n;  // индекс, с которого будем отсекать токены
    for (int i = 0; i < n; i++)
    {
        cumulative += probs[i];
        if (cumulative > p && i != 0)
        {
            cutoff_index = i;
            break;
        }
    }

    // Обрезаем все токены после cutoff_index
    logits.resize(cutoff_index);
}

// Apply Temperature 
void apply_temperature(std::vector<LogitToken> &v, float T)
{
    if (T <= 1e-6f || fabs(T - 1.0f) < 1e-6f)
        return;

    for (auto &lt : v)
        lt.logit /= T;
}

// Sample from softmax
int sample_from_softmax(std::vector<LogitToken> &v, PCG32 &rng)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = rng.random_float();

    float cumulative = 0.0f;
    for (auto &lt : v)
    {
        cumulative += lt.prob;
        if (r <= cumulative)
            return lt.token_id;
    }

    return v.back().token_id; // fallback
}

// Get the highest probability token
int get_next_token_max(const std::vector<float> &logits)
{
    if (logits.empty())
        return -1; // безопасный вариант на случай пустого вектора

    int max_index = 0;
    float max_value = logits[0];

    for (size_t i = 1; i < logits.size(); ++i)
    {
        if (logits[i] > max_value)
        {
            max_value = logits[i];
            max_index = i;
        }
    }

    return max_index;
}


std::vector<LogitToken> top_fraction_logits(const std::vector<float> &raw_logits,
                                            float fraction)
{
    int n = raw_logits.size();

    int top_n = n * fraction;

    if(top_n <= 0 || top_n > raw_logits.size())
        top_n = n;

    // 1) Индексы всех элементов
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    // 2) Частичное разделение — топ top_n
    std::nth_element(idx.begin(), idx.begin() + top_n - 1, idx.end(),
                     [&](int a, int b){ return raw_logits[a] > raw_logits[b]; });

    // 3) Сортируем только top_n по убыванию
    std::sort(idx.begin(), idx.begin() + top_n,
              [&](int a, int b){ return raw_logits[a] > raw_logits[b]; });

    // 4) Создаём вектор LogitToken только для топ-X%
    std::vector<LogitToken> v;
    v.reserve(top_n);
    for (int i = 0; i < top_n; i++)
    {
        int id = idx[i];
        v.push_back({raw_logits[id], id, 0.0f});
    }

    return v;
}

int get_next_token_complex(std::vector<float> logits,
                           const std::unordered_map<int, int> &history,
                           const GenerationOptions &opt)
{

    if(!opt.softmax)
        return get_next_token_max(logits);

    std::vector<LogitToken> v = top_fraction_logits(logits, opt.top_fraction);

    // --- 1) Penalties ---
    apply_penalties(v, history, opt.repetition_penalty, opt.frequency_penalty, opt.presence_penalty);

    // --- 2) Temperature ---
    apply_temperature(v, opt.temperature);

    // --- 3) Top-K ---
    top_k(v, opt.top_k);

    // --- 4) Top-P ---
    top_p(v, opt.top_p);

    // --- 5) Softmax ---
    compute_softmax(v);

    // --- 6) Sampling (only here does the seed influence seed) ---
    PCG32 rng;
    rng.seed(opt.seed);
    return sample_from_softmax(v, rng);
}

static inline float fp16_to_fp32(uint16_t h)
{
    uint16_t sign = (h & 0x8000u) << 16;
    uint16_t exp = (h & 0x7C00u) >> 10;
    uint16_t mant = h & 0x03FFu;

    uint32_t f;

    if (exp == 0)
    {
        // Subnormal or zero
        f = (uint32_t)sign | (mant << 13);
    }
    else if (exp == 0x1F)
    {
        // Inf or NaN
        f = (uint32_t)sign | 0x7F800000u | (mant << 13);
    }
    else
    {
        // Normalized
        f = (uint32_t)sign | ((exp + 112) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}



////////////////////////////////////////////////////////////////////////////////////

CTokenizer *CreateTokenizer(const std::string &model_path)
{
    return new CTokenizer(model_path);
}

// Глобальный, создаётся один раз на всё приложение
static Ort::Env g_env(ORT_LOGGING_LEVEL_WARNING, "inference_onnx");

std::unique_ptr<Ort::Session> CreateSession(const std::string &model_path)
{

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
        session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    ////////////////////////////////////////////////////////////////////////////
    // Inputs

    for (size_t i = 0; i < session->GetInputCount(); ++i)
    {
        auto name_alloc = session->GetInputNameAllocated(i, allocator);
        std::string name(name_alloc.get());
        auto type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Outputs

    for (size_t i = 0; i < session->GetOutputCount(); ++i)
    {
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
    // std::vector<std::string> input_names;
    input_names.clear();
    for (size_t i = 0; i < pSession->GetInputCount(); ++i)
    {
        auto name_alloc = pSession->GetInputNameAllocated(i, allocator);
        std::string name(name_alloc.get());
        input_names.push_back(name);

        auto type_info = pSession->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::cout << i << ": " << name << "  Type: " << tensor_info.GetElementType()
                  << "  Shape: ";
        for (auto dim : tensor_info.GetShape())
            std::cout << dim << " ";
        std::cout << "\n";
    }

    // Print outputs
    std::cout << "\nOutputs:\n";
    // std::vector<std::string> output_names;
    output_names.clear();
    for (size_t i = 0; i < pSession->GetOutputCount(); ++i)
    {
        auto name_alloc = pSession->GetOutputNameAllocated(i, allocator);
        std::string name(name_alloc.get());
        output_names.push_back(name);

        auto type_info = pSession->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        std::cout << i << ": " << name << "  Type: " << tensor_info.GetElementType()
                  << "  Shape: ";
        for (auto dim : tensor_info.GetShape())
            std::cout << dim << " ";
        std::cout << "\n";
    }
}

void CInferenceONNX::ExtractModelConfig()
{

    Ort::AllocatorWithDefaultOptions allocator;

    num_layers = 0;
    num_heads = 0;
    head_dim = 0;
    past_seq_len = 0;

    // -----------------------------
    // 1. Определяем num_layers
    // Считаем все past_key_values.*.key и past_key_values.*.value
    // Каждый слой задаётся двумя входами: key + value
    // -----------------------------
    size_t past_count = 0;

    for (size_t i = 0; i < pSession->GetInputCount(); ++i)
    {
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
    for (size_t i = 0; i < pSession->GetInputCount(); ++i)
    {

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

        num_heads = (shape[1] > 0) ? (int)shape[1] : 0;
        past_seq_len = (shape[2] > 0) ? (int)shape[2] : 0;
        head_dim = (shape[3] > 0) ? (int)shape[3] : 0;

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

CInferenceONNX::CInferenceONNX(const std::string &model_path) : tokenizer(CreateTokenizer(model_path)), pSession(CreateSession(model_path))
{
    CreateInputOutputNames();
    ExtractModelConfig();
}

template <typename T>
void PrintTensor(const Ort::Value &tensor)
{
    // Extract information about the tensor type and shape
    auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor_info.GetShape();

    // Extract data
    const T *data = tensor.GetTensorData<T>();

    // Print the tensor shape
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    // Print data in Python style
    size_t total_size = tensor_info.GetElementCount();
    std::cout << "Data: [";
    for (size_t i = 0; i < total_size; ++i)
    {
        std::cout << static_cast<float>(data[i]) << (i < total_size - 1 ? ", " : "");
    }
    std::cout << "]\n";
}

std::string CInferenceONNX::GetResponse_(const std::string &Request, GenerationOptions options)
{
    std::string Response;
    std::unordered_map<int, int> token_history;

    // Example inputs
    std::vector<int64_t> input_ids = tokenizer->encode(Request);

    int MAX_TOKENS = 20;
    int tokens = 0;
    while ((tokens++) < MAX_TOKENS)
    {

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

        for (int i = 0; i < num_layers * 2; ++i)
        {
            past_key_values.push_back(
                Ort::Value::CreateTensor<Ort::Float16_t>(
                    mem_info,
                    pkv_buffer.data() + i * pkv_elems,
                    pkv_elems,
                    pkv_shape.data(),
                    pkv_shape.size()));
        }

        // std::cout << "\nInput tensors prepared successfully.\n";

        // -------- Prepare input name pointers ----------
        std::vector<const char *> input_names_ptrs;
        input_names_ptrs.reserve(input_names.size());

        for (auto &s : input_names)
            input_names_ptrs.push_back(s.c_str());

        // --- actual input tensors in proper order ---
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(input_ids_tensor));
        inputs.push_back(std::move(attention_mask_tensor));

        // add PKV
        for (auto &v : past_key_values)
            inputs.push_back(std::move(v));

        // -------- Prepare output name pointers ----------
        std::vector<const char *> output_names_ptrs;
        for (auto &s : output_names)
            output_names_ptrs.push_back(s.c_str());

        // ============== RUN ================
        auto output_tensors = pSession->Run(
            Ort::RunOptions{nullptr},
            input_names_ptrs.data(),
            inputs.data(),
            inputs.size(),
            output_names_ptrs.data(),
            output_names_ptrs.size());

        // std::cout << "Inference OK\n";

        // The output is a vector of Ort::Value, we take the first tensor
        const Ort::Value &out = output_tensors[0];

        // Get tensor info
        Ort::TensorTypeAndShapeInfo info = out.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = info.GetShape(); // e.g. [1, 128, 32000]

        if (shape.size() != 3)
            throw std::runtime_error("Unexpected logits shape. Expected [batch, seq, vocab].");

        int64_t batch = shape[0];
        int64_t seq = shape[1];
        int64_t vocab = shape[2];

        if (batch != 1)
            throw std::runtime_error("Batch > 1 is not supported for sampling.");

        const uint16_t *fp16_data = output_tensors[0].GetTensorData<uint16_t>();

        // pointer to last token
        const uint16_t *last_fp16_logits = fp16_data + (seq - 1) * vocab;

        // convert to fp32 vector
        std::vector<float> logits(vocab);
        for (int i = 0; i < vocab; ++i)
            logits[i] = fp16_to_fp32(last_fp16_logits[i]);

        int next_token = get_next_token_complex(logits, token_history, options);
        // Update token history AFTER sampling
        token_history[next_token]++;

        input_ids.push_back(next_token);

        std::string token_str = tokenizer->decode({next_token});

        std::cout << token_str << std::endl;

        Response += token_str;
    }

    return Response;
}

std::string CInferenceONNX::GetResponse(const std::string &Request, GenerationOptions options)
{

    std::cout << Request << std::flush;
    std::string Response;
    std::unordered_map<int, int> token_history;

    std::vector<int64_t> input_ids = tokenizer->encode(Request);

    std::mt19937 rng(options.seed);

    int MAX_TOKENS = options.max_tokens;
    int tokens = 0;

    // --- перед циклом: однажды подготовим буфер PKV и указатели имён ----
    std::vector<int64_t> pkv_shape = {1, num_heads, past_seq_len, head_dim};
    size_t pkv_elems = 1LL * num_heads * past_seq_len * head_dim;

    // выделяем буфер один раз вне цикла
    std::vector<Ort::Float16_t> pkv_buffer(num_layers * 2 * pkv_elems, Ort::Float16_t(0.0f));

    // подготовим input/output name ptrs один раз
    std::vector<const char *> input_names_ptrs;
    input_names_ptrs.reserve(input_names.size());
    for (auto &s : input_names)
        input_names_ptrs.push_back(s.c_str());

    std::vector<const char *> output_names_ptrs;
    output_names_ptrs.reserve(output_names.size());
    for (auto &s : output_names)
        output_names_ptrs.push_back(s.c_str());

    // ----------------- сам цикл (из вашего кода, с изменениями) -----------------
    while ((tokens++) < MAX_TOKENS)
    {
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
        // pkv_buffer уже выделен вне цикла; здесь только создаём Ort::Value-обёртки
        std::vector<Ort::Value> past_key_values;
        past_key_values.reserve(num_layers * 2);

        for (int i = 0; i < num_layers * 2; ++i)
        {
            past_key_values.push_back(
                Ort::Value::CreateTensor<Ort::Float16_t>(
                    mem_info,
                    pkv_buffer.data() + i * pkv_elems,
                    pkv_elems,
                    pkv_shape.data(),
                    pkv_shape.size()));
        }

        // -------- Prepare input tensors in proper order ----------
        std::vector<Ort::Value> inputs;
        inputs.reserve(2 + past_key_values.size());
        inputs.push_back(std::move(input_ids_tensor));
        inputs.push_back(std::move(attention_mask_tensor));

        // add PKV (move Ort::Value into inputs)
        for (auto &v : past_key_values)
            inputs.push_back(std::move(v));

        // ============== RUN ================
        auto output_tensors = pSession->Run(
            Ort::RunOptions{nullptr},
            input_names_ptrs.data(),
            inputs.data(),
            inputs.size(),
            output_names_ptrs.data(),
            output_names_ptrs.size());

        // ... остальная логика обработки output_tensors как и до этого ...
        const Ort::Value &out = output_tensors[0];
        Ort::TensorTypeAndShapeInfo info = out.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = info.GetShape();

        if (shape.size() != 3)
            throw std::runtime_error("Unexpected logits shape. Expected [batch, seq, vocab].");

        int64_t batch = shape[0];
        int64_t seq = shape[1];
        int64_t vocab = shape[2];

        if (batch != 1)
            throw std::runtime_error("Batch > 1 is not supported for sampling.");

        const uint16_t *fp16_data = output_tensors[0].GetTensorData<uint16_t>();
        const uint16_t *last_fp16_logits = fp16_data + (seq - 1) * vocab;

        std::vector<float> logits(vocab);
        for (int i = 0; i < vocab; ++i)
            logits[i] = fp16_to_fp32(last_fp16_logits[i]);

        int next_token = get_next_token_complex(logits, token_history, options);

        //int next_token = get_next_token_max(logits);
        
        token_history[next_token]++;

        input_ids.push_back(next_token);

        std::string token_str = tokenizer->decode({next_token});
        std::cout << token_str << std::flush;
        Response += token_str;

        // --- Обновляем pkv_buffer из выходов модели ---
        std::vector<Ort::Value> new_pkv;
        new_pkv.reserve(num_layers * 2);

        for (int i = 0; i < num_layers * 2; i++)
        {
            new_pkv.push_back(std::move(output_tensors[1 + i]));
        }

        // заменить старые PKV
        past_key_values = std::move(new_pkv);
    }

    return Response;
}

CInferenceONNX::~CInferenceONNX()
{
}