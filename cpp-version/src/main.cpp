#include <torch/torch.h>
#include <iostream>
#include <string>
#include <algorithm>

#include "config.hpp"
#include "tokenizer.hpp"
#include "model/translator.hpp"
#include "data.hpp"
#include "training.hpp"
#include "inference.hpp"
#include "e2b.hpp"

int main(int argc, char **argv)
{
    Config cfg;
    std::string cmd = (argc > 1) ? argv[1] : "";

    // Device selection
    auto device = torch::kCPU;
    if (torch::cuda::is_available())
    {
        device = torch::kCUDA;
    }

    if (cmd == "train")
    {
        auto data = get_raw_data(50000);
        run_training(cfg, data, device);
    }
    else if (cmd == "chat")
    {
        SPTokenizer en_tok(cfg.vocab_size);
        SPTokenizer bn_tok(cfg.vocab_size);

        if (!en_tok.load("en_tok.model") || !bn_tok.load("bn_tok.model"))
        {
            std::cout << "Model file not found. Run 'train' first." << std::endl;
            return 1;
        }

        // Load model
        Translator m(cfg, en_tok.vocab_size(), bn_tok.vocab_size());
        try
        {
            torch::load(m, "translator.pth");
        }
        catch (...)
        {
            std::cout << "Model file not found. Run 'train' first." << std::endl;
            return 1;
        }
        m->to(device);

        std::cout << "Model Loaded! (Vocab: " << en_tok.vocab_size() << " words)" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;

        std::string t;
        while (true)
        {
            std::cout << "\nEN (type 'exit'): ";
            std::getline(std::cin, t);

            // Case-insensitive exit check (matches Python t.lower() == 'exit')
            std::string t_lower = t;
            std::transform(t_lower.begin(), t_lower.end(), t_lower.begin(), ::tolower);
            // Trim whitespace
            t_lower.erase(0, t_lower.find_first_not_of(" \t\r\n"));
            t_lower.erase(t_lower.find_last_not_of(" \t\r\n") + 1);

            if (t_lower == "exit")
                break;

            // Trim original input
            t.erase(0, t.find_first_not_of(" \t\r\n"));
            t.erase(t.find_last_not_of(" \t\r\n") + 1);
            if (t.empty())
                continue;

            std::string pred = translate_sentence(m, en_tok, bn_tok, t);
            std::cout << "BN: " << pred << std::endl;
        }
    }
    else if (cmd == "e2b")
    {
        // E2b numeric conformance (thesis §8.8). Needs no trained checkpoint:
        // the weights come from the shared neutral file.
        std::string dir = (argc > 2) ? argv[2] : "../../analysis/e2b";
        return run_e2b(dir, "cpp");
    }
    else if (cmd == "e2c")
    {
        // E2c gradient conformance (thesis 8.9). Like e2b, needs no trained
        // checkpoint: weights come from the shared neutral file.
        std::string dir = (argc > 2) ? argv[2] : "../../analysis/e2b";
        return run_e2c(dir, "cpp");
    }
    else if (cmd == "validate")
    {
        // Equivalence-protocol validation mode (E1/E2/E3), language-agnostic
        // machine-diffable output schema.
        SPTokenizer en_tok(cfg.vocab_size);
        SPTokenizer bn_tok(cfg.vocab_size);

        if (!en_tok.load("en_tok.model") || !bn_tok.load("bn_tok.model"))
        {
            std::cout << "Model file not found. Run 'train' first." << std::endl;
            return 1;
        }

        Translator m(cfg, en_tok.vocab_size(), bn_tok.vocab_size());
        try
        {
            torch::load(m, "translator.pth");
        }
        catch (...)
        {
            std::cout << "Model file not found. Run 'train' first." << std::endl;
            return 1;
        }
        m->to(device);
        m->eval();

        std::cout << "VALIDATE START lang=cpp" << std::endl;
        std::cout << "VOCAB en=" << en_tok.vocab_size() << " bn=" << bn_tok.vocab_size() << std::endl;

        // E1: canonical strings -> encoder input IDs ([SOS] + ids + [EOS])
        std::vector<std::pair<std::string, std::string>> e1_cases = {
            {"T1", "Hello, world!"},
            {"T2", "I love programming."},
            {"T3", "Neural machine translation is challenging."},
            {"T4", "Data-driven systems require careful evaluation."},
        };
        for (const auto &[tid, text] : e1_cases)
        {
            auto enc = en_tok.encode(text);
            std::cout << "E1 " << tid << " ids=[1";
            for (auto id : enc)
                std::cout << ", " << id;
            std::cout << ", 2]" << std::endl;
        }

        // E2: shape checkpoints for a fixed 2-sentence batch
        {
            torch::NoGradGuard no_grad;
            auto shape_str = [](const torch::Tensor &t)
            {
                std::string s = "[";
                for (int64_t i = 0; i < t.dim(); ++i)
                {
                    if (i)
                        s += ", ";
                    s += std::to_string(t.size(i));
                }
                return s + "]";
            };
            std::vector<torch::Tensor> seqs;
            for (int c = 0; c < 2; ++c)
            {
                std::vector<int64_t> ids = {1};
                auto enc = en_tok.encode(e1_cases[c].second);
                ids.insert(ids.end(), enc.begin(), enc.end());
                ids.push_back(2);
                seqs.push_back(torch::tensor(ids));
            }
            int64_t max_len = std::max(seqs[0].size(0), seqs[1].size(0));
            auto src = torch::zeros({2, max_len}, torch::kLong);
            for (int c = 0; c < 2; ++c)
            {
                src[c].slice(0, 0, seqs[c].size(0)).copy_(seqs[c]);
            }
            src = src.to(device);
            auto trg = torch::tensor({{1L, 4L, 5L, 2L}, {1L, 6L, 7L, 2L}}).to(device);

            std::cout << "E2 enc_input_ids shape=" << shape_str(src) << std::endl;
            auto emb = m->se(src) * std::sqrt(static_cast<double>(m->dim));
            std::cout << "E2 enc_embed shape=" << shape_str(emb) << std::endl;
            auto sm = (src != 0).unsqueeze(1).unsqueeze(2);
            auto e = emb;
            for (size_t i = 0; i < m->enc->size(); ++i)
            {
                e = m->enc->at<BlockImpl>(i).forward(e, {}, sm, {}, m->freqs);
            }
            std::cout << "E2 enc_out shape=" << shape_str(e) << std::endl;
            std::cout << "E2 dec_input_ids shape=" << shape_str(trg) << std::endl;
            auto logits = m->forward(src, trg);
            std::cout << "E2 dec_logits shape=" << shape_str(logits) << std::endl;
        }

        // E3: beam-search decision trace for a fixed input
        const DecodingPolicy &dp = decoding_policy();
        std::cout << "E3 params beam=" << dp.beam
                  << " lp=" << dp.lp << " rp=" << dp.rep_penalty
                  << " max_steps=" << dp.max_steps
                  << " temp=" << dp.temperature << std::endl;
        std::string result = translate_sentence(m, en_tok, bn_tok, "Hello.",
                                                dp.beam, dp.lp, dp.rep_penalty, true);
        std::cout << "E3 result=" << result << std::endl;
        std::cout << "VALIDATE END" << std::endl;
    }
    else
    {
        std::cout << "Usage: translator-ai-cpp [train|chat|validate|e2b <dir>|e2c <dir>]" << std::endl;
    }

    return 0;
}
