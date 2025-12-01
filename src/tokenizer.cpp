#include "tokenizer.h"

// Convert a UTF-8 string to Latin-1, ignoring unsupported characters
std::string utf8ToLatin1(const std::string& utf8Str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> utf8Converter;
    std::wstring wideStr = utf8Converter.from_bytes(utf8Str);

    std::string latin1Str;
    short SymbolCount = 0;
    for (wchar_t wc : wideStr) {
        latin1Str += static_cast<unsigned char>(wc);
        SymbolCount++;
    }

    //if(SymbolCount < 2)
    //    return utf8Str;

    return latin1Str;
}

// Converts a vector of bytes to a string
std::string bytes_to_string(const std::vector<unsigned char>& bytes) {
    return std::string(bytes.begin(), bytes.end());
}

// Function for loading a JSON file
nlohmann::json load_json(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
    nlohmann::json j;
    file >> j;
    return j;
}

// Function for splitting a string using PCRE2 (pre-tokenization)
std::vector<std::string> splitWithRegex(const std::string& input, const std::string& pattern) {
    std::vector<std::string> tokens;

    int errornumber;
    PCRE2_SIZE erroroffset;
    pcre2_code *re = pcre2_compile(
        reinterpret_cast<PCRE2_SPTR>(pattern.c_str()), // Pattern
        PCRE2_ZERO_TERMINATED,                         // Pattern length
        PCRE2_UTF | PCRE2_UCP,                         // Flags for UTF-8 support
        &errornumber,                                  // Error code
        &erroroffset,                                  // Error offset
        NULL                                           // Options
    );

    if (!re) {
        std::cerr << "PCRE2 compilation failed at offset " << erroroffset
                  << ": error code " << errornumber << std::endl;
        return tokens;
    }

    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(re, NULL);
    PCRE2_SIZE* ovector;

    size_t offset = 0;
    while (offset < input.size()) {
        int rc = pcre2_match(
            re,                                          // Compiled pattern
            reinterpret_cast<PCRE2_SPTR>(input.c_str()), // Input string
            input.size(),                                // Length of the string
            offset,                                      // Offset to start searching from
            0,                                           // Options
            match_data,                                  // Data structure to store the result
            NULL                                         // Match context (optional)
        );

        if (rc <= 0) break;

        ovector = pcre2_get_ovector_pointer(match_data);
        size_t start = ovector[0];
        size_t end = ovector[1];
        tokens.emplace_back(input.substr(start, end - start));

        offset = end;
    }

    pcre2_match_data_free(match_data);
    pcre2_code_free(re);

    return tokens;
}

CTokenizer::CTokenizer(const std::string& model_path)
{

    bool slash = false;
    if (model_path.back() == '/')
        slash = true;

    try
    {
        std::string tokenizer_json;

        if (slash)
            tokenizer_json = "tokenizer.json";
        else
            tokenizer_json = "/tokenizer.json";

        tokenizer = load_json(model_path + tokenizer_json);
    }
    catch (const std::runtime_error &e)
    {
        LOGS << "[ERROR] Cannot open file: " << e.what() << std::endl;
        exit(1);
    }

    try
    {
        regexPattern = tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"];
    }
    catch (const nlohmann::detail::type_error& e)
    {
        LOGS << "[WARNING] " << e.what() << std::endl;
    }

    if(regexPattern.length() == 0)
        LOGS << "[WARNING] regexPattern == 0 (there will be no pre-tokenization)" << std::endl;

    try
    {
        begin_of_text_token.first = tokenizer["post_processor"]["processors"][1]["single"][0]["SpecialToken"]["id"];
        begin_of_text_token.second = tokenizer["post_processor"]["processors"][1]["special_tokens"][begin_of_text_token.first]["ids"][0];
    }
    catch (const nlohmann::detail::type_error& e)
    {
        LOGS << "[WARNING] " << e.what() << std::endl;
    }

    // Load the dictionary and merge rules
    // Get the "vocab" dictionary from JSON
    auto originalVocab = tokenizer["model"]["vocab"].get<std::unordered_map<std::string, int>>();

    // We go through all the elements and change the keys
    for (const auto& [key, value] : originalVocab) {
        std::string modifiedKey = utf8ToLatin1(key);

        // Insert the modified key (if it is unique or) and the corresponding value into vocab
        if (vocab.find(modifiedKey) == vocab.end() || vocab[modifiedKey] > value) 
            vocab[modifiedKey] = value;

    }

    // filling the map for decoding
    for (const auto& [key, value] : vocab) 
        bacov[value] = key;

    // filling the map with additional tokens
    size_t added_tokens_count = tokenizer["added_tokens"].size();
    for(size_t i = 0; i < added_tokens_count; i++)
    {
        added_tokens[tokenizer["added_tokens"][i]["id"]] = tokenizer["added_tokens"][i]["content"];
    }
    
    std::vector<std::pair<std::string, std::string>> merges;
    for (const auto &merge : tokenizer["model"]["merges"])
    {
        if (merge.is_array() && merge.size() == 2)
        {
            merges.emplace_back(utf8ToLatin1(merge[0].get<std::string>()), utf8ToLatin1((merge[1].get<std::string>())));
        }
        else if (!merge.is_array())
        {
            size_t pos = merge.get<std::string>().find(' ');
            if (pos != std::string::npos)
            {
                std::string firstPart = utf8ToLatin1(merge.get<std::string>().substr(0, pos));
                std::string secondPart = utf8ToLatin1(merge.get<std::string>().substr(pos + 1));
                merges.emplace_back(firstPart, secondPart);
            }
        }
    }

   // Sort the merges by their order in the "merges" array
    for (size_t i = 0; i < merges.size(); ++i) {
        merge_rank[merges[i]] = i;
    }

    LOGS << "[INFO] Vocab: " << vocab.size() << std::endl;
    LOGS << "[INFO] Merges: " << merges.size() << std::endl;
}

std::vector<std::string> TextToLetters(const std::string &text)
{
    std::vector<std::string> letters;
    for (size_t i = 0; i < text.size();)
    {
        // Determine the length of the current UTF-8 character
        unsigned char byte = static_cast<unsigned char>(text[i]);
        size_t char_len = 0;

        if ((byte & 0x80) == 0)
            char_len = 1; // 1 byte (ASCII)
        else if ((byte & 0xE0) == 0xC0)
            char_len = 2; // 2 bytes
        else if ((byte & 0xF0) == 0xE0)
            char_len = 3; // 3 bytes
        else if ((byte & 0xF8) == 0xF0)
            char_len = 4; // 4 bytes (emoji, etc.)

        // Extract the character as a substring
        std::string sub_str = text.substr(i, char_len);

        letters.emplace_back(sub_str);
        i += char_len; // Move on to the next character
    }

    return letters;
}

std::vector<std::string> LettersToBytes(const std::vector<std::string> &letters)
{
    std::vector<std::string> tokens;

    // Split the text into bytes, including spaces
    for (auto let : letters)
    {
        std::vector<unsigned char> letter;
        short k = 0;
        for (unsigned char ch : let)
        {
            if (k > 0)
            {
                if (ch <= 160)
                    ch -= 94;
                else if (ch == 170) // the - Ъ sign consists of 2 tokenss
                {
                    tokens.push_back(bytes_to_string(letter));
                    letter.clear();
                }
                else if (ch <= 173)
                    ch -= 106;
            }
            else
                k++;
            letter.push_back(ch);
        }

        tokens.push_back(bytes_to_string(letter));
    }

    return tokens;
}

std::string BytesToText(const std::vector<std::string> &letters)
{
    std::string Text = "";
    std::vector<std::string> tokens;

    // Split the text into bytes, including spaces
    for (auto let : letters)
    {
        std::vector<unsigned char> letter;
        short k = 0;
        for (unsigned char ch : let)
        {
            if (k > 0)
            {
                if (ch <= (160 - 94))
                    ch += 94;
                else if (ch == 170) // the - Ъ sign consists of 2 tokens
                {
                    tokens.push_back(bytes_to_string(letter));
                    letter.clear();
                }
                else if (ch <= (173 - 106))
                    ch += 106;
            }
            else
                k++;
            letter.push_back(ch);
        }

        tokens.push_back(bytes_to_string(letter));
    }

    for(size_t i = 0; i < tokens.size(); i++)
        Text += tokens[i];
    
    return Text;
}
// Get key by value
std::optional<std::string> getKeyByValue(const std::unordered_map<std::string, int64_t>& map, int64_t value) {
    for (const auto& [key, val] : map) {
        if (val == value) {
            return key; // Key found
        }
    }
    return std::nullopt; // Key not found
}

void CTokenizer::MergeTokens(std::vector<std::string> &tokens)
{
        while (true) {
        std::pair<int, std::pair<size_t, size_t>> best_merge = {INT_MAX, {0, 0}};
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            std::pair<std::string, std::string> bigram = {tokens[i], tokens[i + 1]};
            if (merge_rank.find(bigram) != merge_rank.end()) {
                int rank = merge_rank[bigram];
                if (rank < best_merge.first) {
                    best_merge = {rank, {i, i + 1}};
                }
            }
        }

        // If there is nothing more to merge, we exit
        if (best_merge.first == INT_MAX) {
            break;
        }

        // Perform the merge
        size_t start = best_merge.second.first;
        size_t end = best_merge.second.second;
        tokens[start] = tokens[start] + tokens[end];
        tokens.erase(tokens.begin() + end);
    }
}

std::vector<int64_t> CTokenizer::encode(const std::string& text)
{
    std::vector<std::string> tokens;
    m_UnrecognizedTokens.clear();
    m_UnrecognizedTokensFirst.UnrecognizedStrs.clear();
    m_UnrecognizedTokensFirst.Poses.clear();
    m_UnrecognizedTokensFirst.RecognizedIDs.clear();


    if(regexPattern.length() > 0)
        tokens = splitWithRegex(text, regexPattern);

    if (tokens.size() == 0) // if there is no pre-tokenization
    {
        // Break the text into characters
        std::vector<std::string> letters = TextToLetters(text);

        //get tokens
        tokens = LettersToBytes(letters);
    }

    // Apply merges

    this->MergeTokens(tokens);

    // Convert to token indices
    std::vector<int64_t> token_ids;
    token_ids.push_back(begin_of_text_token.second);
    int pos = 0;
    for (const std::string& token : tokens) {
        if (vocab.find(token) != vocab.end()) {
            token_ids.push_back(vocab[token]);
        } else {
            //LOGS << "[WARNING] Unrecognized token: " << token << std::endl;
            m_UnrecognizedTokensFirst.UnrecognizedStrs.push_back(token);
            m_UnrecognizedTokensFirst.Poses.push_back(pos);
        }
        pos++;
    }

    // split tokens character by character, then merge, re-recognize and insert into the appropriate positions
    for (size_t i = 0; i < m_UnrecognizedTokensFirst.UnrecognizedStrs.size(); i++)
    {
        std::vector<std::string> letters = TextToLetters(m_UnrecognizedTokensFirst.UnrecognizedStrs[i]);
        std::vector<std::string> buf_tokens = LettersToBytes(letters); // tokens cut into letters
        this->MergeTokens(buf_tokens);

        std::vector<int64_t> token_ids_buf;
        for (const std::string &token : buf_tokens)
        {
            if (vocab.find(token) != vocab.end())
            {
                token_ids_buf.push_back(vocab[token]);
            }
            else
            {
                LOGS << "[WARNING] Unrecognized token: " << token << std::endl;
                m_UnrecognizedTokens.push_back(token);
            }
        }

        m_UnrecognizedTokensFirst.RecognizedIDs.push_back(token_ids_buf);
    }

    pos = 0;
    for (size_t i = 0; i < m_UnrecognizedTokensFirst.UnrecognizedStrs.size(); i++)
    {
        token_ids.insert(token_ids.begin() + m_UnrecognizedTokensFirst.Poses[i] + pos + 1, m_UnrecognizedTokensFirst.RecognizedIDs[i].begin(), m_UnrecognizedTokensFirst.RecognizedIDs[i].end());
        pos += m_UnrecognizedTokensFirst.RecognizedIDs[i].size() - 1;
    }

    return token_ids;
}

std::string CTokenizer::decode(const std::vector<int64_t>& token_ids)
{
    std::string RawText;

    for(size_t i = 0; i < token_ids.size(); i++)
    {
        if (bacov.find(token_ids[i]) != bacov.end())
            RawText += bacov[token_ids[i]];
        else if (added_tokens.find(token_ids[i]) != added_tokens.end())
             RawText += added_tokens[token_ids[i]];
        else
        {
            LOGS << "[WARNING] Unknown token ID: " << token_ids[i] << std::endl;
            m_UnknownTokenIDs.push_back(token_ids[i]);
        }
    }

    // Break the text into characters
    std::vector<std::string> letters = TextToLetters(RawText);
    std::string response;
    response = BytesToText(letters);
    
        
    return response;
}

std::string CTokenizer::GetTokenByID(size_t id)
{
    auto key = getKeyByValue(vocab, id);

    return *key;
}

void CTokenizer::WriteLogs()
{
    std::ofstream file("logs.txt");
    file << LOGS.str();
    file.close();
}

void CTokenizer::PrintLogs()
{
    std::cout << LOGS.str() << std::endl;
}