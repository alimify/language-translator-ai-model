#include "training.hpp"
#include "data.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void run_training(const Config &cfg,
                  std::vector<std::pair<std::string, std::string>> &data,
                  torch::Device device)
{
    if (data.empty())
        return;

    // Shuffle and split data (95% train, 5% val)
    // Hold out the final 200 pairs; seed the shuffle for cross-
    // implementation identity (see analysis/patch_disjoint_split.py).
    if (data.size() > 200)
        data.resize(data.size() - 200);
    std::mt19937 g(20260815);
    std::shuffle(data.begin(), data.end(), g);
    std::cout << "Split: pool=" << data.size() << " seed=20260815" << std::endl;

    size_t split = static_cast<size_t>(data.size() * 0.95);
    std::vector<std::pair<std::string, std::string>> train_data(data.begin(), data.begin() + split);

    // Remove existing checkpoint
    std::remove("translator.pth");

    // Tokenizers: load the shared frozen artifacts when present (thesis
    // equivalence protocol E1 requires one tokenizer artifact across all
    // language implementations); otherwise train from the local split.
    SPTokenizer en_tok(cfg.vocab_size);
    SPTokenizer bn_tok(cfg.vocab_size);

    if (en_tok.load("en_tok.model") && bn_tok.load("bn_tok.model"))
    {
        std::cout << "Loaded shared tokenizer artifacts (en_tok.model, bn_tok.model)" << std::endl;
    }
    else
    {
        std::vector<std::string> en_texts, bn_texts;
        for (const auto &[en, bn] : train_data)
        {
            en_texts.push_back(en);
            bn_texts.push_back(bn);
        }
        en_tok.train(en_texts);
        bn_tok.train(bn_texts);
    }

    // Initialize model
    Translator m(cfg, en_tok.vocab_size(), bn_tok.vocab_size());
    m->to(device);

    // Optimizer with weight decay
    torch::optim::AdamW optimizer(m->parameters(),
                                  torch::optim::AdamWOptions(cfg.lr).weight_decay(0.01));

    // OneCycleLR scheduler, matching torch.optim.lr_scheduler.OneCycleLR with
    // anneal_strategy='cos', pct_start=0.3, div_factor=25 (default),
    // final_div_factor=1e4 (default), cycle_momentum=True (default; for AdamW
    // this cycles betas[0] between max_momentum=0.95 and base_momentum=0.85).
    int total_steps = cfg.epochs * (static_cast<int>(train_data.size()) / cfg.batch_size + 1);

    const double max_lr = cfg.lr;
    const double initial_lr = max_lr / 25.0;
    const double min_lr = initial_lr / 10000.0;
    const double phase1_end = 0.3 * total_steps - 1.0;
    const double phase2_end = static_cast<double>(total_steps) - 1.0;

    auto anneal_cos = [](double start, double end, double pct)
    {
        return end + (start - end) / 2.0 * (std::cos(M_PI * pct) + 1.0);
    };
    auto one_cycle = [&](int step_num, double &lr_out, double &beta1_out)
    {
        if (step_num <= phase1_end)
        {
            double pct = step_num / phase1_end;
            lr_out = anneal_cos(initial_lr, max_lr, pct);
            beta1_out = anneal_cos(0.95, 0.85, pct);
        }
        else
        {
            double pct = (step_num - phase1_end) / (phase2_end - phase1_end);
            lr_out = anneal_cos(max_lr, min_lr, pct);
            beta1_out = anneal_cos(0.85, 0.95, pct);
        }
    };
    auto apply_lr = [&](double lr, double beta1)
    {
        for (auto &group : optimizer.param_groups())
        {
            auto &opts = static_cast<torch::optim::AdamWOptions &>(group.options());
            opts.lr(lr);
            opts.betas(std::make_tuple(beta1, std::get<1>(opts.betas())));
        }
    };

    // PyTorch sets the LR/momentum for step 0 at scheduler construction, so
    // the very first optimizer.step() runs at initial_lr with beta1=0.95.
    double current_lr = 0.0, current_beta1 = 0.0;
    one_cycle(0, current_lr, current_beta1);
    apply_lr(current_lr, current_beta1);

    std::cout << "Training (Data: " << data.size() << " pairs | Steps: " << total_steps << ")" << std::endl;

    int global_step = 0;

    for (int ep = 0; ep < cfg.epochs; ++ep)
    {
        std::shuffle(train_data.begin(), train_data.end(), g);
        m->train();
        float total_loss = 0;
        int batch_count = 0;

        for (size_t i = 0; i < train_data.size(); i += cfg.batch_size)
        {
            // Build batch
            std::vector<std::pair<std::string, std::string>> batch;
            for (size_t j = i; j < i + cfg.batch_size && j < train_data.size(); ++j)
            {
                batch.push_back(train_data[j]);
            }
            if (batch.size() < 2)
                continue;

            // Prepare batch tensors with SOS and EOS tokens
            std::vector<torch::Tensor> src_list, trg_list;
            for (const auto &[e, b] : batch)
            {
                std::vector<int64_t> src_ids = {1};
                auto en_enc = en_tok.encode(e);
                src_ids.insert(src_ids.end(), en_enc.begin(), en_enc.end());
                src_ids.push_back(2);
                src_list.push_back(torch::tensor(src_ids));

                std::vector<int64_t> trg_ids = {1};
                auto bn_enc = bn_tok.encode(b);
                trg_ids.insert(trg_ids.end(), bn_enc.begin(), bn_enc.end());
                trg_ids.push_back(2);
                trg_list.push_back(torch::tensor(trg_ids));
            }

            // pad_sequence equivalent (batch_first=True, pad with 0)
            auto pad_sequence = [](const std::vector<torch::Tensor> &seqs)
            {
                int64_t max_len = 0;
                for (const auto &s : seqs)
                    max_len = std::max(max_len, s.size(0));
                std::vector<torch::Tensor> padded;
                for (const auto &s : seqs)
                {
                    auto p = torch::zeros(max_len, torch::kLong);
                    p.slice(0, 0, s.size(0)).copy_(s);
                    padded.push_back(p);
                }
                return torch::stack(padded);
            };

            auto src = pad_sequence(src_list).to(device);
            auto trg = pad_sequence(trg_list).to(device);

            // Forward pass
            optimizer.zero_grad();
            auto logits = m->forward(src, trg.slice(1, 0, -1));

            // Cross-entropy loss with label smoothing
            auto loss = torch::nn::functional::cross_entropy(
                logits.reshape({-1, logits.size(-1)}),
                trg.slice(1, 1).reshape({-1}),
                torch::nn::functional::CrossEntropyFuncOptions()
                    .ignore_index(0)
                    .label_smoothing(cfg.label_smoothing));

            // Backward pass
            loss.backward();

            // Gradient clipping
            torch::nn::utils::clip_grad_norm_(m->parameters(), 1.0);

            // Optimizer step, then scheduler step (same order as Python:
            // optimizer.step(); scheduler.step())
            optimizer.step();
            global_step++;
            one_cycle(global_step, current_lr, current_beta1);
            apply_lr(current_lr, current_beta1);

            total_loss += loss.item<float>();
            batch_count++;
        }

        // Epoch logging (current_lr mirrors scheduler.get_last_lr()[0])
        float avg_loss = batch_count > 0 ? total_loss / batch_count : 0;
        std::cout << "Epoch " << ep + 1 << "/" << cfg.epochs
                  << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | LR: " << std::setprecision(7) << current_lr << std::endl;
    }

    // Save model and tokenizers
    torch::save(m, "translator.pth");
    en_tok.save("en_tok.model");
    bn_tok.save("bn_tok.model");
    std::cout << "Saved to 'translator.pth'" << std::endl;
}
