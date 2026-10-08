#include "e2b.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "config.hpp"
#include "model/translator.hpp"

namespace
{

    // The sidecar formats exist so that no implementation needs a JSON parser to
    // participate; parsing them requires nothing beyond the standard library.
    std::map<std::string, std::string> read_arch(const std::string &path)
    {
        std::map<std::string, std::string> kv;
        std::ifstream fh(path);
        std::string line;
        while (std::getline(fh, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
            auto eq = line.find('=');
            if (eq == std::string::npos)
                continue;
            std::string v = line.substr(eq + 1);
            while (!v.empty() && (v.back() == '\r' || v.back() == '\n'))
                v.pop_back();
            kv[line.substr(0, eq)] = v;
        }
        return kv;
    }

    struct Entry
    {
        long long offset = 0;
        long long count = 0;
        std::vector<int64_t> dims;
    };

    std::map<std::string, Entry> read_manifest(const std::string &path)
    {
        std::map<std::string, Entry> out;
        std::ifstream fh(path);
        std::string line;
        while (std::getline(fh, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
            std::vector<std::string> f;
            std::string cell;
            std::istringstream ss(line);
            while (std::getline(ss, cell, '\t'))
                f.push_back(cell);
            if (f.size() < 4)
                continue;
            Entry e;
            e.offset = std::stoll(f[1]);
            e.count = std::stoll(f[2]);
            std::istringstream ds(f[3]);
            while (std::getline(ds, cell, ','))
            {
                while (!cell.empty() && (cell.back() == '\r' || cell.back() == '\n'))
                    cell.pop_back();
                if (!cell.empty())
                    e.dims.push_back(std::stoll(cell));
            }
            out[f[0]] = e;
        }
        return out;
    }

    std::vector<std::vector<int64_t>> read_probe_block(std::ifstream &fh, int rows)
    {
        std::vector<std::vector<int64_t>> m;
        std::string line;
        for (int r = 0; r < rows && std::getline(fh, line); ++r)
        {
            std::vector<int64_t> row;
            std::string cell;
            std::istringstream ss(line);
            while (std::getline(ss, cell, ','))
            {
                while (!cell.empty() && (cell.back() == '\r' || cell.back() == '\n'))
                    cell.pop_back();
                if (!cell.empty())
                    row.push_back(std::stoll(cell));
            }
            m.push_back(row);
        }
        return m;
    }

} // namespace

int run_e2b(const std::string &dir, const std::string &tag)
{
    const std::string sep = "/";
    auto arch = read_arch(dir + sep + "arch.txt");
    if (arch.empty())
    {
        std::cerr << "E2b: cannot read " << dir << "/arch.txt" << std::endl;
        return 1;
    }
    auto man = read_manifest(dir + sep + "manifest.tsv");
    if (man.empty())
    {
        std::cerr << "E2b: cannot read " << dir << "/manifest.tsv" << std::endl;
        return 1;
    }

    Config cfg;
    cfg.dim = std::stoi(arch["dim"]);
    cfg.heads = std::stoi(arch["heads"]);
    cfg.layers = std::stoi(arch["layers"]);
    cfg.ff_mult = std::stoi(arch["ff_mult"]);
    cfg.max_len = std::stoi(arch["max_len"]);
    cfg.dropout = 0.0;
    const int sv = std::stoi(arch["vocab_src"]);
    const int tv = std::stoi(arch["vocab_trg"]);

    // LibTorch stores nn::Linear weights as [out, in], the same convention the
    // file declares, so no transpose is required here. An implementation whose
    // linear layers expect [in, out] must transpose -- see arch.txt's
    // linear_layout and the JavaScript runner.
    if (arch["linear_layout"] != "out_in")
    {
        std::cerr << "E2b: unexpected linear_layout " << arch["linear_layout"]
                  << std::endl;
        return 1;
    }

    std::ifstream bin(dir + sep + "model.bin", std::ios::binary);
    if (!bin)
    {
        std::cerr << "E2b: cannot open model.bin" << std::endl;
        return 1;
    }

    Translator m(cfg, sv, tv);
    m->to(torch::kCPU);
    m->eval();

    // Populate every registered parameter by name. `freqs` is a buffer and is
    // absent from the file by design: it is recomputed by the constructor from
    // the architecture, so an incorrect rotation cannot borrow the reference's
    // table (thesis §8.8).
    int loaded = 0;
    torch::NoGradGuard no_grad;
    for (auto &p : m->named_parameters())
    {
        const std::string name = p.key();
        auto it = man.find(name);
        if (it == man.end())
        {
            std::cerr << "E2b: no tensor named " << name << " in manifest"
                      << std::endl;
            return 1;
        }
        const Entry &e = it->second;
        std::vector<float> buf(static_cast<size_t>(e.count));
        bin.seekg(e.offset);
        bin.read(reinterpret_cast<char *>(buf.data()),
                 static_cast<std::streamsize>(e.count * sizeof(float)));
        if (!bin)
        {
            std::cerr << "E2b: short read for " << name << std::endl;
            return 1;
        }
        auto t = torch::from_blob(buf.data(), e.dims, torch::kFloat32).clone();
        if (t.sizes() != p.value().sizes())
        {
            std::cerr << "E2b: shape mismatch for " << name << std::endl;
            return 1;
        }
        p.value().copy_(t);
        ++loaded;
    }

    // Weight tying: the output projection is computed functionally from
    // te->weight in this implementation, so assigning te.weight is sufficient.

    std::ifstream pf(dir + sep + "probe.txt");
    if (!pf)
    {
        std::cerr << "E2b: cannot read probe.txt" << std::endl;
        return 1;
    }
    std::string line;
    int src_rows = 0, trg_rows = 0;
    std::vector<std::vector<int64_t>> src_ids, trg_ids;
    while (std::getline(pf, line))
    {
        if (line.rfind("src_rows=", 0) == 0)
            src_rows = std::stoi(line.substr(9));
        else if (line.rfind("src_cols=", 0) == 0)
            src_ids = read_probe_block(pf, src_rows);
        else if (line.rfind("trg_rows=", 0) == 0)
            trg_rows = std::stoi(line.substr(9));
        else if (line.rfind("trg_cols=", 0) == 0)
            trg_ids = read_probe_block(pf, trg_rows);
    }
    if (src_ids.empty() || trg_ids.empty())
    {
        std::cerr << "E2b: probe.txt did not parse" << std::endl;
        return 1;
    }

    auto to_tensor = [](const std::vector<std::vector<int64_t>> &v)
    {
        const int64_t R = static_cast<int64_t>(v.size());
        const int64_t C = static_cast<int64_t>(v[0].size());
        std::vector<int64_t> flat;
        flat.reserve(static_cast<size_t>(R * C));
        for (const auto &row : v)
            flat.insert(flat.end(), row.begin(), row.end());
        return torch::from_blob(flat.data(), {R, C}, torch::kInt64).clone();
    };
    auto src = to_tensor(src_ids);
    auto trg = to_tensor(trg_ids);

    torch::Tensor logits = m->forward(src, trg);
    auto out = logits.contiguous().to(torch::kFloat32);

    const std::string outpath = dir + sep + "logits_" + tag + ".bin";
    std::ofstream of(outpath, std::ios::binary);
    of.write(reinterpret_cast<const char *>(out.data_ptr<float>()),
             static_cast<std::streamsize>(out.numel() * sizeof(float)));
    of.close();

    double sum = out.to(torch::kDouble).sum().item<double>();
    std::cout << tag << ": shape=[" << out.size(0) << "," << out.size(1) << ","
              << out.size(2) << "] params=" << loaded
              << " sum=" << std::fixed << sum
              << " min=" << out.min().item<float>()
              << " max=" << out.max().item<float>() << std::endl;
    std::cout << "  wrote " << outpath << std::endl;
    return 0;
}

int run_e2c(const std::string &dir, const std::string &tag)
{
    const std::string sep = "/";
    auto arch = read_arch(dir + sep + "arch.txt");
    if (arch.empty())
    {
        std::cerr << "E2c: cannot read " << dir << "/arch.txt" << std::endl;
        return 1;
    }
    auto man = read_manifest(dir + sep + "manifest.tsv");
    if (man.empty())
    {
        std::cerr << "E2c: cannot read " << dir << "/manifest.tsv" << std::endl;
        return 1;
    }
    if (arch["linear_layout"] != "out_in")
    {
        std::cerr << "E2c: unexpected linear_layout" << std::endl;
        return 1;
    }

    Config cfg;
    cfg.dim = std::stoi(arch["dim"]);
    cfg.heads = std::stoi(arch["heads"]);
    cfg.layers = std::stoi(arch["layers"]);
    cfg.ff_mult = std::stoi(arch["ff_mult"]);
    cfg.max_len = std::stoi(arch["max_len"]);
    // Dropout off: a stochastic mask would make the gradient a random variable
    // and the cross-implementation comparison meaningless.
    cfg.dropout = 0.0;
    const int sv = std::stoi(arch["vocab_src"]);
    const int tv = std::stoi(arch["vocab_trg"]);

    std::ifstream bin(dir + sep + "model.bin", std::ios::binary);
    if (!bin)
    {
        std::cerr << "E2c: cannot open model.bin" << std::endl;
        return 1;
    }

    Translator m(cfg, sv, tv);
    m->to(torch::kCPU);
    // train() matches the training loop's mode; dropout is 0.0 so this stays
    // deterministic.
    m->train();

    int loaded = 0;
    {
        torch::NoGradGuard no_grad;
        for (auto &p : m->named_parameters())
        {
            const std::string name = p.key();
            auto it = man.find(name);
            if (it == man.end())
            {
                std::cerr << "E2c: no tensor named " << name << std::endl;
                return 1;
            }
            const Entry &e = it->second;
            std::vector<float> buf(static_cast<size_t>(e.count));
            bin.seekg(e.offset);
            bin.read(reinterpret_cast<char *>(buf.data()),
                     static_cast<std::streamsize>(e.count * sizeof(float)));
            if (!bin)
            {
                std::cerr << "E2c: short read for " << name << std::endl;
                return 1;
            }
            auto t = torch::from_blob(buf.data(), e.dims, torch::kFloat32).clone();
            if (t.sizes() != p.value().sizes())
            {
                std::cerr << "E2c: shape mismatch for " << name << std::endl;
                return 1;
            }
            p.value().copy_(t);
            ++loaded;
        }
    }

    std::ifstream pf(dir + sep + "probe.txt");
    if (!pf)
    {
        std::cerr << "E2c: cannot read probe.txt" << std::endl;
        return 1;
    }
    std::string line;
    int src_rows = 0, trg_rows = 0;
    std::vector<std::vector<int64_t>> src_ids, trg_ids;
    while (std::getline(pf, line))
    {
        if (line.rfind("src_rows=", 0) == 0)
            src_rows = std::stoi(line.substr(9));
        else if (line.rfind("src_cols=", 0) == 0)
            src_ids = read_probe_block(pf, src_rows);
        else if (line.rfind("trg_rows=", 0) == 0)
            trg_rows = std::stoi(line.substr(9));
        else if (line.rfind("trg_cols=", 0) == 0)
            trg_ids = read_probe_block(pf, trg_rows);
    }
    if (src_ids.empty() || trg_ids.empty())
    {
        std::cerr << "E2c: probe.txt did not parse" << std::endl;
        return 1;
    }
    auto to_tensor = [](const std::vector<std::vector<int64_t>> &v)
    {
        const int64_t R = static_cast<int64_t>(v.size());
        const int64_t C = static_cast<int64_t>(v[0].size());
        std::vector<int64_t> flat;
        flat.reserve(static_cast<size_t>(R * C));
        for (const auto &row : v)
            flat.insert(flat.end(), row.begin(), row.end());
        return torch::from_blob(flat.data(), {R, C}, torch::kInt64).clone();
    };
    auto src = to_tensor(src_ids);
    auto trg = to_tensor(trg_ids);

    // Teacher forcing and loss exactly as src/training.cpp computes them.
    auto trg_in = trg.slice(1, 0, trg.size(1) - 1);
    auto trg_out = trg.slice(1, 1);
    auto logits = m->forward(src, trg_in);
    auto loss = torch::nn::functional::cross_entropy(
        logits.reshape({-1, logits.size(-1)}),
        trg_out.reshape({-1}),
        torch::nn::functional::CrossEntropyFuncOptions()
            .ignore_index(0)
            .label_smoothing(cfg.label_smoothing));

    m->zero_grad();
    loss.backward();

    // Emit in sorted parameter-name order, matching the reference runner. The
    // order is recorded in the sidecar so the comparison reports a
    // misalignment as a misalignment rather than as a numeric divergence.
    std::vector<std::string> names;
    for (const auto &p : m->named_parameters())
        names.push_back(p.key());
    std::sort(names.begin(), names.end());

    const std::string outpath = dir + sep + "grads_" + tag + ".bin";
    std::ofstream of(outpath, std::ios::binary);
    double total_sq = 0.0;
    int with_grad = 0;
    int64_t values = 0;
    std::string order_csv;
    for (const auto &name : names)
    {
        auto p = m->named_parameters().find(name);
        if (p == nullptr || !p->grad().defined())
            continue;
        auto g = p->grad().contiguous().to(torch::kFloat32);
        total_sq += g.to(torch::kDouble).pow(2).sum().item<double>();
        of.write(reinterpret_cast<const char *>(g.data_ptr<float>()),
                 static_cast<std::streamsize>(g.numel() * sizeof(float)));
        values += g.numel();
        if (!order_csv.empty())
            order_csv += ",";
        order_csv += name;
        ++with_grad;
    }
    of.close();
    const double grad_norm = std::sqrt(total_sq);
    const double loss_val = loss.item<double>();

    std::ofstream sc(dir + sep + "grads_" + tag + ".txt");
    sc.precision(10);
    sc << std::fixed;
    sc << "tag=" << tag << "\n";
    sc << "loss=" << loss_val << "\n";
    sc << "grad_norm=" << grad_norm << "\n";
    sc << "grad_values=" << values << "\n";
    sc << "tensors_with_grad=" << with_grad << "\n";
    sc << "params_loaded=" << loaded << "\n";
    sc << "order=" << order_csv << "\n";
    sc.close();

    std::cout.precision(10);
    std::cout << std::fixed << tag << ": loss=" << loss_val
              << " grad_norm=" << grad_norm << " values=" << values
              << " tensors=" << with_grad << std::endl;
    std::cout << "  wrote " << outpath << std::endl;
    return 0;
}
