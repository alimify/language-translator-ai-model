//! CLI entry point for training and chat modes.
//!
//! Matches Python: main.py

use std::io::{ self, Write };
use tch::{ nn, Device };

use translator_ai_rust::config::Config;
use translator_ai_rust::tokenizer::SPTokenizer;
use translator_ai_rust::model::Translator;
use translator_ai_rust::data::get_raw_data;
use translator_ai_rust::training::run_training;
use translator_ai_rust::inference::translate_sentence;
use translator_ai_rust::e2b::run_e2b;
use translator_ai_rust::e2c::run_e2c;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("");

    let cfg = Config::default();

    // Device selection
    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };

    match cmd {
        "train" => {
            let mut data = get_raw_data(50000, cfg.max_len as usize);
            run_training(&cfg, &mut data, device);
        }

        "e2b" => {
            // E2b numeric conformance (thesis 8.8). Needs no trained
            // checkpoint: weights come from the shared neutral file.
            let dir = args
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("../../analysis/e2b");
            std::process::exit(run_e2b(dir, "rust"));
        }
        "e2c" => {
            let dir = args
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("../../analysis/e2b");
            std::process::exit(run_e2c(dir, "rust"));
        }

        "chat" => {
            // Load tokenizers
            let mut en_tok = SPTokenizer::new(cfg.vocab_size);
            let mut bn_tok = SPTokenizer::new(cfg.vocab_size);

            if en_tok.load("en_tok.model").is_err() || bn_tok.load("bn_tok.model").is_err() {
                println!("Model file not found. Run 'train' first.");
                return;
            }

            // Load model
            let mut vs = nn::VarStore::new(device);
            let model = Translator::new(&vs.root(), &cfg, en_tok.vocab_size(), bn_tok.vocab_size());

            if vs.load("translator.pth").is_err() {
                println!("Model file not found. Run 'train' first.");
                return;
            }

            println!("Model Loaded! (Vocab: {} words)", en_tok.vocab_size());
            println!("--------------------------------------------------");

            loop {
                print!("\nEN (type 'exit'): ");
                io::stdout().flush().unwrap();

                let mut t = String::new();
                io::stdin().read_line(&mut t).unwrap();
                let t = t.trim();

                if t.eq_ignore_ascii_case("exit") {
                    break;
                }
                if t.is_empty() {
                    continue;
                }

                let pred = translate_sentence(
                    &model,
                    &en_tok,
                    &bn_tok,
                    t,
                    3, // beam_size
                    0.6, // lp
                    1.4, // rep_penalty
                    device,
                    false
                );
                println!("BN: {}", pred);
            }
        }

        "validate" => {
            // Equivalence-protocol validation mode (E1/E2/E3), language-
            // agnostic machine-diffable output schema.
            let mut en_tok = SPTokenizer::new(cfg.vocab_size);
            let mut bn_tok = SPTokenizer::new(cfg.vocab_size);

            if en_tok.load("en_tok.model").is_err() || bn_tok.load("bn_tok.model").is_err() {
                println!("Model file not found. Run 'train' first.");
                return;
            }

            let mut vs = nn::VarStore::new(device);
            let model = Translator::new(&vs.root(), &cfg, en_tok.vocab_size(), bn_tok.vocab_size());
            if vs.load("translator.pth").is_err() {
                println!("Model file not found. Run 'train' first.");
                return;
            }

            println!("VALIDATE START lang=rust");
            println!("VOCAB en={} bn={}", en_tok.vocab_size(), bn_tok.vocab_size());

            // E1: canonical strings -> encoder input IDs ([SOS] + ids + [EOS])
            let e1_cases = [
                ("T1", "Hello, world!"),
                ("T2", "I love programming."),
                ("T3", "Neural machine translation is challenging."),
                ("T4", "Data-driven systems require careful evaluation."),
            ];
            for (tid, text) in &e1_cases {
                let mut ids: Vec<i64> = vec![1];
                ids.extend(en_tok.encode(text));
                ids.push(2);
                let strs: Vec<String> = ids
                    .iter()
                    .map(|i| i.to_string())
                    .collect();
                println!("E1 {} ids=[{}]", tid, strs.join(", "));
            }

            // E2: shape checkpoints for a fixed 2-sentence batch
            tch::no_grad(|| {
                let mut seqs: Vec<Vec<i64>> = Vec::new();
                for (_, text) in e1_cases.iter().take(2) {
                    let mut ids: Vec<i64> = vec![1];
                    ids.extend(en_tok.encode(text));
                    ids.push(2);
                    seqs.push(ids);
                }
                let max_len = seqs
                    .iter()
                    .map(|s| s.len())
                    .max()
                    .unwrap();
                for s in &mut seqs {
                    s.resize(max_len, 0);
                }
                let flat: Vec<i64> = seqs.concat();
                let src = tch::Tensor
                    ::from_slice(&flat)
                    .view([2, max_len as i64])
                    .to_device(device);
                let trg = tch::Tensor
                    ::from_slice(&[1i64, 4, 5, 2, 1, 6, 7, 2])
                    .view([2, 4])
                    .to_device(device);
                model.validate_shapes(&src, &trg);
            });

            // E3: beam-search decision trace for a fixed input
            let dp = translator_ai_rust::inference::DECODING_POLICY;
            println!(
                "E3 params beam={} lp={} rp={} max_steps={} temp={}",
                dp.beam,
                dp.lp,
                dp.rep_penalty,
                dp.max_steps,
                dp.temperature
            );
            let result = translate_sentence(
                &model,
                &en_tok,
                &bn_tok,
                "Hello.",
                dp.beam,
                dp.lp,
                dp.rep_penalty,
                device,
                true
            );
            println!("E3 result={}", result);
            println!("VALIDATE END");
        }

        _ => {
            println!("Usage: translator-ai-rust [train|chat|validate|e2b <dir>|e2c <dir>]");
        }
    }
}
