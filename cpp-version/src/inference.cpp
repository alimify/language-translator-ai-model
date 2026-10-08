#include "inference.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <unordered_map>

const DecodingPolicy &decoding_policy()
{
    static const DecodingPolicy p;
    return p;
}

std::string translate_sentence(Translator &m,
                               SPTokenizer &en_tok,
                               SPTokenizer &bn_tok,
                               const std::string &sentence,
                               int beam_size,
                               float lp,
                               float rep_penalty,
                               bool trace)
{
    m->eval();

    // Encode source sentence
    auto tokens = en_tok.encode(sentence);
    if (tokens.empty())
        return "";

    // Create source tensor with SOS and EOS
    std::vector<int64_t> src_vec = {1};
    src_vec.insert(src_vec.end(), tokens.begin(), tokens.end());
    src_vec.push_back(2);

    auto device = m->parameters()[0].device();
    auto src = torch::tensor(src_vec).unsqueeze(0).to(device);

    // Initialize beams: (normalized_score, log_sum, sequence)
    struct Beam
    {
        float norm_score;
        float log_sum;
        std::vector<int64_t> seq;
        bool operator<(const Beam &other) const { return norm_score < other.norm_score; }
    };

    std::vector<Beam> beams = {{0.0f, 0.0f, {1}}};

    for (int step = 0; step < decoding_policy().max_steps; ++step)
    {
        std::vector<Beam> new_beams;

        for (const auto &b : beams)
        {
            // If sequence ended, keep it as-is
            if (b.seq.back() == 2)
            {
                float ns = b.log_sum / std::pow(b.seq.size(), lp);
                new_beams.push_back({ns, b.log_sum, b.seq});
                continue;
            }

            torch::NoGradGuard no_grad;
            auto trg = torch::tensor(b.seq).unsqueeze(0).to(device);
            auto logits = m->forward(src, trg);

            // Get last step logits with temperature scaling
            auto curr_logits = logits.index({0, -1, torch::indexing::Slice()}) / decoding_policy().temperature;

            // Apply repetition penalty
            std::unordered_map<int64_t, int> token_counts;
            for (auto t : b.seq)
            {
                if (t > 2)
                    token_counts[t]++; // Skip special tokens
            }

            for (const auto &[token, count] : token_counts)
            {
                // Stronger penalty for repeated tokens
                float penalty = (count == 1) ? rep_penalty : rep_penalty * 1.5f;

                if (curr_logits[token].item<float>() > 0)
                {
                    curr_logits[token] /= penalty;
                }
                else
                {
                    curr_logits[token] *= penalty;
                }
            }

            // Get top-k candidates
            auto probs = torch::log_softmax(curr_logits, -1);
            auto top_k = probs.topk(beam_size);
            auto top_v = std::get<0>(top_k);
            auto top_i = std::get<1>(top_k);

            for (int i = 0; i < beam_size; ++i)
            {
                int64_t t_id = top_i[i].item<int64_t>();
                float new_sum = b.log_sum + top_v[i].item<float>();
                auto new_seq = b.seq;
                new_seq.push_back(t_id);
                float normalized_score = new_sum / std::pow(new_seq.size(), lp);
                new_beams.push_back({normalized_score, new_sum, new_seq});
            }
        }

        // Keep top beams (stable sort matches Python's sorted(), preserving
        // insertion order on ties)
        std::stable_sort(new_beams.begin(), new_beams.end(),
                         [](const Beam &a, const Beam &b)
                         { return a.norm_score > b.norm_score; });

        if (new_beams.size() > static_cast<size_t>(beam_size))
        {
            new_beams.resize(beam_size);
        }
        beams = new_beams;

        // Structured beam trace (equivalence protocol E3)
        if (trace)
        {
            for (size_t rank = 0; rank < beams.size(); ++rank)
            {
                printf("E3 step=%d rank=%zu tok=%lld score=%.6f\n",
                       step + 1, rank,
                       static_cast<long long>(beams[rank].seq.back()),
                       beams[rank].norm_score);
            }
        }

        // Early stopping if all beams ended
        bool all_eos = true;
        for (const auto &b : beams)
        {
            if (b.seq.back() != 2)
                all_eos = false;
        }
        if (all_eos)
            break;
    }

    return bn_tok.decode(beams[0].seq);
}
