#include "tokenizer.h"
#include <ctime>

int main() {

    CTokenizer tokenizer("../models/Llama-3.2-3B-Instruct-ONNX/cuda/cuda-int4-rtn-block-32/");

    std::vector<int64_t> token_ids;

    std::string teststr = "Lllama-Hello, Мирююбббр!-ДААА3п.2-до3B-Instruct-ONNX/cuda/cuda-int4-rtn-block-32/ЪЪЪм мм32 Hello world - Hello, мир}Ъ!";
    token_ids = tokenizer.encode(teststr);
    

    if(tokenizer.m_UnrecognizedTokens.size() > 0)
        std::cout << "Unrecognized tokens: ";

    for(size_t i = 0; i < tokenizer.m_UnrecognizedTokens.size(); i++)
    {
        std::cout << tokenizer.m_UnrecognizedTokens[i];

        if(i < tokenizer.m_UnrecognizedTokens.size() - 1)
            std::cout << ", ";
        else
            std::cout << std::endl;
    }

    std::cout << "Tokens: ";
    for(size_t i = 0; i < token_ids.size(); i++)
    {
        std::cout << token_ids[i];

        if(i < token_ids.size() - 1)
            std::cout << ", ";
        else
            std::cout << std::endl;
    }


    
    
    std::string decodestr = tokenizer.decode(token_ids);

    std::cout << decodestr << std::endl;

    if(decodestr != teststr)
    std::cout << "[ERROR] Strings do not match" << std::endl;

     if(tokenizer.m_UnknownTokenIDs.size() > 0)
        std::cout << "Unknown token IDs: ";
    
    for(size_t i = 0; i < tokenizer.m_UnknownTokenIDs.size(); i++)
    {
        std::cout << tokenizer.m_UnknownTokenIDs[i];

        if(i < tokenizer.m_UnrecognizedTokens.size() - 1)
            std::cout << ", ";
        else
            std::cout << std::endl;
    }

    tokenizer.WriteLogs();
        
return 0;
}