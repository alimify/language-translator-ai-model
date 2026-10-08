//! E2c gradient conformance runner (thesis §8.9, §9.7).
//!
//! E2b checks that this implementation computes the same decoder logits as the
//! reference when both load the same weights. That verifies the forward pass.
//! This runner verifies the backward one: it loads the same shared weights,
//! runs the same fixed probe through the *training* step this implementation
//! actually uses -- teacher forcing, cross-entropy with label smoothing 0.1 and
//! padding index 0 ignored -- calls `backward()`, and writes every parameter
//! gradient as little-endian `f32` in manifest order.
//!
//! Why it matters here: §9.2 reports that the five implementations differ in
//! held-out quality and that the difference cannot be attributed, because each
//! was trained once from its own initialisation. That treats training as opaque.
//! If the five agree on the gradient at step one, they were descending the same
//! surface and the divergence is attributable to initialisation and data order
//! alone. If they disagree, the divergence has a mechanism.
//!
//! Gradients are captured before `clip_grad_norm`, which projects onto a ball of
//! radius 1.0 and would mask a disagreement in magnitude.
//!
//! The loss is computed with the same call the training loop uses, rather than
//! restated here, so that this runner measures the implementation's own training
//! step and not a second opinion about what that step should be.

use std::fs::File;
use std::io::{ Read, Seek, SeekFrom, Write };
use std::path::Path;

use tch::{ nn::VarStore, Device, Kind, Tensor };

use crate::config::Config;
use crate::e2b::{ read_arch, read_manifest, read_probe };
use crate::model::Translator;

/// Run the E2c gradient conformance probe. Returns 0 on success.
pub fn run_e2c(dir: &str, tag: &str) -> i32 {
    let base = Path::new(dir);
    let arch = read_arch(&base.join("arch.txt"));
    if arch.is_empty() {
        eprintln!("E2c: cannot read {}/arch.txt", dir);
        return 1;
    }
    let man = read_manifest(&base.join("manifest.tsv"));
    if man.is_empty() {
        eprintln!("E2c: cannot read {}/manifest.tsv", dir);
        return 1;
    }
    if arch.get("linear_layout").map(|s| s.as_str()) != Some("out_in") {
        eprintln!("E2c: unexpected linear_layout");
        return 1;
    }

    let geti = |k: &str| -> i64 {
        arch.get(k)
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    let mut cfg = Config::default();
    cfg.dim = geti("dim");
    cfg.heads = geti("heads");
    cfg.layers = geti("layers");
    cfg.ff_mult = geti("ff_mult");
    cfg.max_len = geti("max_len");
    // Dropout off: a stochastic mask makes the gradient a random variable and
    // the cross-implementation comparison meaningless.
    cfg.dropout = 0.0;
    let (sv, tv) = (geti("vocab_src"), geti("vocab_trg"));

    let vs = VarStore::new(Device::Cpu);
    let model = Translator::new(&vs.root(), &cfg, sv, tv);

    let mut bin = match File::open(base.join("model.bin")) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("E2c: cannot open model.bin: {}", e);
            return 1;
        }
    };

    // Load under no_grad so the copy itself does not enter the graph; the
    // variables remain trainable afterwards, which is what backward() needs.
    let mut loaded = 0usize;
    let mut missing: Vec<String> = Vec::new();
    let mut order: Vec<String> = Vec::new();
    tch::no_grad(|| {
        let vars = vs.variables();
        let mut names: Vec<&String> = vars.keys().collect();
        names.sort();
        for name in names {
            let entry = match man.get(name) {
                Some(e) => e,
                None => {
                    missing.push(name.clone());
                    continue;
                }
            };
            let mut buf = vec![0u8; (entry.count as usize) * 4];
            if
                bin.seek(SeekFrom::Start(entry.offset)).is_err() ||
                bin.read_exact(&mut buf).is_err()
            {
                missing.push(format!("{} (short read)", name));
                continue;
            }
            let floats: Vec<f32> = buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let t = Tensor::from_slice(&floats).to_kind(Kind::Float).reshape(&entry.dims);
            let mut var = vars.get(name).unwrap().shallow_clone();
            if var.size() != t.size() {
                missing.push(format!("{} (shape)", name));
                continue;
            }
            var.copy_(&t);
            order.push(name.clone());
            loaded += 1;
        }
    });
    if !missing.is_empty() {
        eprintln!("E2c: {} variable(s) not loaded:", missing.len());
        for m in missing.iter().take(12) {
            eprintln!("   {}", m);
        }
        return 1;
    }

    let (src_ids, trg_ids) = read_probe(&base.join("probe.txt"));
    if src_ids.is_empty() || trg_ids.is_empty() {
        eprintln!("E2c: probe.txt did not parse");
        return 1;
    }
    let to_t = |v: &Vec<Vec<i64>>| -> Tensor {
        let rows = v.len() as i64;
        let cols = v[0].len() as i64;
        let flat: Vec<i64> = v.iter().flatten().copied().collect();
        Tensor::from_slice(&flat).reshape(&[rows, cols])
    };
    let src = to_t(&src_ids);
    let trg = to_t(&trg_ids);

    // Teacher forcing exactly as src/training.rs does it: input trg[:, :-1],
    // targets trg[:, 1:].
    let cols = trg.size()[1];
    let trg_in = trg.narrow(1, 0, cols - 1);
    let trg_out = trg.narrow(1, 1, cols - 1);

    // train=true matches the training loop's call; dropout is 0.0 so this stays
    // deterministic.
    let logits = model.forward(&src, &trg_in, true);
    let vocab_size = logits.size()[2];
    let loss = logits
        .reshape(&[-1, vocab_size])
        .cross_entropy_loss::<Tensor>(
            &trg_out.reshape(&[-1]),
            None,
            tch::Reduction::Mean,
            0,
            cfg.label_smoothing
        );

    // No zero_grad needed: this VarStore was constructed in this function and
    // has had exactly one forward pass, so every gradient is still undefined
    // and nothing can accumulate into it. (tch 0.14's VarStore exposes no
    // zero_grad; the optimizer owns that operation in the training loop.)
    loss.backward();

    // Emit gradients in manifest order so the byte layout matches the
    // reference's grads_python.bin exactly.
    let vars = vs.variables();
    let mut bytes: Vec<u8> = Vec::new();
    let mut total_sq = 0f64;
    let mut with_grad = 0usize;
    let mut names: Vec<&String> = vars.keys().collect();
    names.sort();
    // Record the emission order. The comparison script checks this against the
    // reference's order before comparing any values, so a misalignment is
    // reported as a misalignment instead of appearing as a numeric divergence
    // -- which is exactly how it first presented.
    let mut emitted: Vec<String> = Vec::new();
    for name in &names {
        let v = vars.get(*name).unwrap();
        let g = v.grad();
        if !g.defined() {
            continue;
        }
        emitted.push((*name).clone());
        let g = g.contiguous().to_kind(Kind::Float);
        let n: i64 = g.size().iter().product();
        let mut data = vec![0f32; n as usize];
        g.copy_data(&mut data, n as usize);
        for val in &data {
            total_sq += (*val as f64) * (*val as f64);
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        with_grad += 1;
    }
    let grad_norm = total_sq.sqrt();

    let outpath = base.join(format!("grads_{}.bin", tag));
    match File::create(&outpath) {
        Ok(mut f) => {
            if f.write_all(&bytes).is_err() {
                eprintln!("E2c: write failed");
                return 1;
            }
        }
        Err(e) => {
            eprintln!("E2c: cannot create {:?}: {}", outpath, e);
            return 1;
        }
    }

    let loss_val = f64::try_from(&loss).unwrap_or(f64::NAN);
    let sidecar = base.join(format!("grads_{}.txt", tag));
    if let Ok(mut f) = File::create(&sidecar) {
        let _ = writeln!(f, "tag={}", tag);
        let _ = writeln!(f, "loss={:.10}", loss_val);
        let _ = writeln!(f, "grad_norm={:.10}", grad_norm);
        let _ = writeln!(f, "grad_values={}", bytes.len() / 4);
        let _ = writeln!(f, "tensors_with_grad={}", with_grad);
        let _ = writeln!(f, "params_loaded={}", loaded);
        let _ = writeln!(f, "order={}", emitted.join(","));
    }

    println!(
        "{}: loss={:.10} grad_norm={:.10} values={} tensors={}",
        tag,
        loss_val,
        grad_norm,
        bytes.len() / 4,
        with_grad
    );
    println!("  wrote {:?}", outpath);
    0
}
