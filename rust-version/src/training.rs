//! Training loop implementation.
//!
//! Matches Python: training.py

use std::fs;
use tch::{ nn, nn::OptimizerConfig, Device, Kind, Tensor };
use rand::seq::SliceRandom;
use crate::config::Config;
use crate::tokenizer::SPTokenizer;
use crate::model::Translator;

/// Run full training loop.
///
/// Matches Python: def run_training(cfg, data)
///
/// Features:
/// - 95/5 train/val split
/// - AdamW optimizer with weight_decay=0.01
/// - OneCycleLR scheduler with pct_start=0.3, cosine annealing
/// - Gradient clipping with max_norm=1.0
/// - Label smoothing
/// - Model checkpoint saving
pub fn run_training(cfg: &Config, data: &mut Vec<(String, String)>, device: Device) {
    if data.is_empty() {
        return;
    }

    // Shuffle and split data (95% train, 5% val)
    // Hold out the final 200 pairs; seed the shuffle for cross-
    // implementation identity (see analysis/patch_disjoint_split.py).
    let keep = data.len().saturating_sub(200);
    data.truncate(keep);
    println!("Split: pool={} seed=20260815", data.len());
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(20260815);
    data.shuffle(&mut rng);
    let split = ((data.len() as f64) * 0.95) as usize;
    let train_data: Vec<(String, String)> = data[..split].to_vec();

    // Remove existing checkpoint
    let _ = fs::remove_file("translator.pth");

    // Tokenizers: load the shared frozen artifacts when present (thesis
    // equivalence protocol E1 requires one tokenizer artifact across all
    // language implementations); otherwise train via the spm_train CLI.
    let mut en_tok = SPTokenizer::new(cfg.vocab_size);
    let mut bn_tok = SPTokenizer::new(cfg.vocab_size);

    if en_tok.load("en_tok.model").is_ok() && bn_tok.load("bn_tok.model").is_ok() {
        println!("Loaded shared tokenizer artifacts (en_tok.model, bn_tok.model)");
    } else {
        let en_texts: Vec<String> = train_data
            .iter()
            .map(|(e, _)| e.clone())
            .collect();
        let bn_texts: Vec<String> = train_data
            .iter()
            .map(|(_, b)| b.clone())
            .collect();

        en_tok.train(&en_texts).expect("Failed to train EN tokenizer");
        bn_tok.train(&bn_texts).expect("Failed to train BN tokenizer");
    }

    // Initialize model (Xavier init over the VarStore matches Python's
    // post-construction init loop)
    let vs = nn::VarStore::new(device);
    let model = Translator::new(&vs.root(), cfg, en_tok.vocab_size(), bn_tok.vocab_size());
    Translator::xavier_init_varstore(&vs);

    // Optimizer: AdamW with explicit weight_decay=0.01 (matches Python)
    let adamw = nn::AdamW { wd: 0.01, ..Default::default() };
    let mut opt = adamw.build(&vs, cfg.lr).unwrap();

    // OneCycleLR scheduler, matching torch.optim.lr_scheduler.OneCycleLR with
    // anneal_strategy='cos', pct_start=0.3, div_factor=25 (default),
    // final_div_factor=1e4 (default), cycle_momentum=True (default; for AdamW
    // this cycles betas[0] between max_momentum=0.95 and base_momentum=0.85).
    let total_steps = cfg.epochs * ((train_data.len() as i64) / cfg.batch_size + 1);
    let max_lr = cfg.lr;
    let initial_lr = max_lr / 25.0;
    let min_lr = initial_lr / 10000.0;
    let phase1_end = 0.3 * (total_steps as f64) - 1.0;
    let phase2_end = (total_steps as f64) - 1.0;

    let anneal_cos = |start: f64, end: f64, pct: f64| -> f64 {
        end + ((start - end) / 2.0) * ((std::f64::consts::PI * pct).cos() + 1.0)
    };
    let one_cycle = |step_num: i64| -> (f64, f64) {
        let s = step_num as f64;
        if s <= phase1_end {
            let pct = s / phase1_end;
            (anneal_cos(initial_lr, max_lr, pct), anneal_cos(0.95, 0.85, pct))
        } else {
            let pct = (s - phase1_end) / (phase2_end - phase1_end);
            (anneal_cos(max_lr, min_lr, pct), anneal_cos(0.85, 0.95, pct))
        }
    };

    // PyTorch sets the LR/momentum for step 0 at scheduler construction, so
    // the very first opt.step() runs at initial_lr with beta1=0.95.
    let (mut current_lr, mut current_beta1) = one_cycle(0);
    opt.set_lr(current_lr);
    opt.set_momentum(current_beta1);

    println!("Training (Data: {} pairs | Steps: {})", data.len(), total_steps);

    let mut global_step: i64 = 0;

    for ep in 0..cfg.epochs {
        // Shuffle each epoch
        let mut indices: Vec<usize> = (0..train_data.len()).collect();
        indices.shuffle(&mut rng);

        let mut total_loss = 0.0;
        let mut batch_count = 0;

        let mut i = 0;
        while i < indices.len() {
            let batch_end = (i + (cfg.batch_size as usize)).min(indices.len());
            let batch_indices = &indices[i..batch_end];

            if batch_indices.len() < 2 {
                i = batch_end;
                continue;
            }

            // Prepare batch tensors with SOS and EOS tokens
            let mut max_src = 0;
            let mut max_trg = 0;
            let mut src_vecs = Vec::new();
            let mut trg_vecs = Vec::new();

            for &idx in batch_indices {
                let (en, bn) = &train_data[idx];

                let mut src: Vec<i64> = vec![1]; // SOS
                src.extend(en_tok.encode(en));
                src.push(2); // EOS

                let mut trg: Vec<i64> = vec![1];
                trg.extend(bn_tok.encode(bn));
                trg.push(2);

                max_src = max_src.max(src.len());
                max_trg = max_trg.max(trg.len());

                src_vecs.push(src);
                trg_vecs.push(trg);
            }

            // Pad sequences
            for src in &mut src_vecs {
                src.resize(max_src, 0);
            }
            for trg in &mut trg_vecs {
                trg.resize(max_trg, 0);
            }

            let src_flat: Vec<i64> = src_vecs.into_iter().flatten().collect();
            let trg_flat: Vec<i64> = trg_vecs.into_iter().flatten().collect();

            let batch_size = batch_indices.len() as i64;
            let src = Tensor::from_slice(&src_flat)
                .view([batch_size, max_src as i64])
                .to_device(device);
            let trg = Tensor::from_slice(&trg_flat)
                .view([batch_size, max_trg as i64])
                .to_device(device);

            let trg_in = trg.slice(1, 0, -1, 1);
            let trg_out = trg.slice(1, 1, max_trg as i64, 1);

            // Forward pass
            opt.zero_grad();
            let logits = model.forward(&src, &trg_in, true);
            let vocab_size = logits.size()[2];

            // Cross-entropy loss with label smoothing
            let loss = logits
                .reshape(&[-1, vocab_size])
                .cross_entropy_loss::<Tensor>(
                    &trg_out.reshape(&[-1]),
                    None,
                    tch::Reduction::Mean,
                    0,
                    cfg.label_smoothing
                );

            // Backward pass with gradient clipping (max_norm=1.0)
            loss.backward();
            opt.clip_grad_norm(1.0);

            // Optimizer step, then scheduler step (same order as Python:
            // optimizer.step(); scheduler.step())
            opt.step();
            global_step += 1;
            let (lr, beta1) = one_cycle(global_step);
            current_lr = lr;
            current_beta1 = beta1;
            opt.set_lr(current_lr);
            opt.set_momentum(current_beta1);

            total_loss += loss.double_value(&[]);
            batch_count += 1;
            i = batch_end;
        }

        let avg_loss = if batch_count > 0 { total_loss / (batch_count as f64) } else { 0.0 };
        // current_lr mirrors scheduler.get_last_lr()[0]
        println!("Epoch {}/{} | Loss: {:.4} | LR: {:.7}", ep + 1, cfg.epochs, avg_loss, current_lr);
    }

    // Save model and tokenizers
    vs.save("translator.pth").unwrap();
    en_tok.save("en_tok.model").expect("Failed to save EN tokenizer");
    bn_tok.save("bn_tok.model").expect("Failed to save BN tokenizer");
    println!("Saved to 'translator.pth'");
}
