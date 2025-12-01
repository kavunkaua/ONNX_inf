#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <climits>
#include <nlohmann/json.hpp>


#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

struct UnrecognizedTokens {
    std::vector<std::string>  UnrecognizedStrs;
    std::vector<int> Poses;
    std::vector<std::vector<int64_t>>  RecognizedIDs;     // распознаные IDs токены
}; 

class CTokenizer {
public:

    // unrecognized tokens
    std::vector<std::string> m_UnrecognizedTokens;
    std::vector<int> m_UnknownTokenIDs;
    

    CTokenizer(const std::string& model_path);

    std::vector<int64_t> encode(const std::string& text);

    std::string decode(const std::vector<int64_t>& token_ids);

    std::string GetTokenByID(size_t id);

    void WriteLogs();
    void PrintLogs();

private:

// unrecognized tokens after the first pass
UnrecognizedTokens m_UnrecognizedTokensFirst;

std::string regexPattern = "";
std::stringstream LOGS;
nlohmann::json tokenizer;
std::unordered_map<std::string, int64_t> vocab;
std::unordered_map<int64_t, std::string> bacov;
std::unordered_map<int64_t, std::string> added_tokens;
std::map<std::pair<std::string, std::string>, int64_t> merge_rank;

std::pair<std::string, int64_t> begin_of_text_token;


void MergeTokens(std::vector<std::string> &tokens);

};

// Convert a UTF-8 string to Latin-1, ignoring unsupported characters
std::string utf8ToLatin1(const std::string& utf8Str);

// Converts a vector of bytes to a string
std::string bytes_to_string(const std::vector<unsigned char>& bytes);

// Loading the json file
nlohmann::json load_json(const std::string& file_path);

// Function for splitting a string using PCRE2
std::vector<std::string> splitWithRegex(const std::string& input, const std::string& pattern);


#endif  // TOKENIZER_H