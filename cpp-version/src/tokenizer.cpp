#include "tokenizer.hpp"
#include <sentencepiece_trainer.h>
#include <fstream>
#include <stdexcept>
#include <cstdio>

SPTokenizer::SPTokenizer(int vocab_size) : _vocab_size(vocab_size)
{
    sp = std::make_unique<sentencepiece::SentencePieceProcessor>();
}

void SPTokenizer::train(const std::vector<std::string> &texts)
{
    // Create temp file for training
    std::ofstream out("temp_train.txt");
    for (const auto &t : texts)
    {
        out << t << "\n";
    }
    out.close();

    // Train model
    std::string model_prefix = "sp_model";
    std::string args = "--input=temp_train.txt"
                       " --model_prefix=" +
                       model_prefix +
                       " --vocab_size=" + std::to_string(_vocab_size) +
                       " --model_type=unigram"
                       " --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3"
                       " --pad_piece=<PAD> --bos_piece=<SOS> --eos_piece=<EOS> --unk_piece=<UNK>"
                       " --character_coverage=1.0"
                       " --hard_vocab_limit=false";

    auto status = sentencepiece::SentencePieceTrainer::Train(args);
    if (!status.ok())
    {
        throw std::runtime_error("SentencePiece training failed: " + status.ToString());
    }

    // Load model content
    std::ifstream model_file(model_prefix + ".model", std::ios::binary);
    model_content = std::string(
        (std::istreambuf_iterator<char>(model_file)),
        std::istreambuf_iterator<char>());
    model_file.close();

    // Load into processor
    status = sp->LoadFromSerializedProto(model_content);
    if (!status.ok())
    {
        throw std::runtime_error("Failed to load trained model: " + status.ToString());
    }

    // Cleanup
    std::remove("temp_train.txt");
    std::remove((model_prefix + ".model").c_str());
    std::remove((model_prefix + ".vocab").c_str());
}

std::vector<int64_t> SPTokenizer::encode(const std::string &text) const
{
    std::vector<int> ids;
    sp->Encode(text, &ids);
    // Convert int to int64_t
    std::vector<int64_t> result;
    result.reserve(ids.size());
    for (int id : ids)
        result.push_back(static_cast<int64_t>(id));
    return result;
}

std::string SPTokenizer::decode(const std::vector<int64_t> &ids) const
{
    std::vector<int> int_ids;
    for (int64_t id : ids)
        int_ids.push_back(static_cast<int>(id));

    std::string text;
    sp->Decode(int_ids, &text);
    return text;
}

int SPTokenizer::vocab_size() const
{
    return sp->GetPieceSize();
}

bool SPTokenizer::load(const std::string &filename)
{
    std::ifstream model_file(filename, std::ios::binary);
    if (!model_file.is_open())
        return false;

    model_content = std::string(
        (std::istreambuf_iterator<char>(model_file)),
        std::istreambuf_iterator<char>());
    model_file.close();

    if (model_content.empty())
        return false;
    return sp->LoadFromSerializedProto(model_content).ok();
}

bool SPTokenizer::save(const std::string &filename) const
{
    if (model_content.empty())
        return false;
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open())
        return false;
    out.write(model_content.data(), model_content.size());
    return out.good();
}
