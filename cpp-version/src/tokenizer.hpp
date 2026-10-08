#pragma once
#include <string>
#include <vector>
#include <memory>
#include <sentencepiece_processor.h>

/**
 * @brief SentencePiece tokenizer wrapper supporting English and Bengali.
 * 
 * Special tokens are handled by SentencePiece:
 *   - <PAD>: 0
 *   - <SOS>: 1
 *   - <EOS>: 2
 *   - <UNK>: 3
 */
class SPTokenizer {
public:
    explicit SPTokenizer(int vocab_size);
    
    void train(const std::vector<std::string>& texts);
    std::vector<int64_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int64_t>& ids) const;
    
    int vocab_size() const;

    // Serialization support (matches Python __getstate__/__setstate__)
    bool load(const std::string& filename);
    bool save(const std::string& filename) const;

private:
    int _vocab_size;
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp;
    std::string model_content;  // Store model binary for serialization
};
