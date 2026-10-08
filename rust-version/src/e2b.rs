//! E2b numeric conformance runner (thesis §8.8, §9.6).
//!
//! Loads the framework-neutral weight file shared by all five implementations,
//! runs the fixed probe forward pass, and writes decoder logits as
//! little-endian `f32` so the numeric paths can be compared directly.
//!
//! Deliberately does not read `translator.pth` and needs no trained checkpoint
//! of its own: the point of the shared file is that an implementation without
//! trained weights can still participate in the comparison.
//!
//! The sidecar formats (`arch.txt`, `manifest.tsv`, `probe.txt`) are parsed
//! with the standard library alone — requiring a JSON crate here would
//! contradict the reason the format exists.

use std::collections::HashMap;
use std::fs::File;
use std::io::{ BufRead, BufReader, Read, Seek, SeekFrom, Write };
use std::path::Path;

use tch::{ nn, nn::VarStore, Device, Kind, Tensor };

use crate::config::Config;
use crate::model::Translator;

pub(crate) struct Entry {
    pub(crate) offset: u64,
    pub(crate) count: i64,
    pub(crate) dims: Vec<i64>,
}

pub(crate) fn read_arch(path: &Path) -> HashMap<String, String> {
    let mut kv = HashMap::new();
    if let Ok(fh) = File::open(path) {
        for line in BufReader::new(fh).lines().flatten() {
            let line = line.trim().to_string();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(eq) = line.find('=') {
                kv.insert(line[..eq].to_string(), line[eq + 1..].to_string());
            }
        }
    }
    kv
}

pub(crate) fn read_manifest(path: &Path) -> HashMap<String, Entry> {
    let mut out = HashMap::new();
    if let Ok(fh) = File::open(path) {
        for line in BufReader::new(fh).lines().flatten() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 4 {
                continue;
            }
            let dims: Vec<i64> = f[3]
                .trim()
                .split(',')
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok())
                .collect();
            out.insert(f[0].to_string(), Entry {
                offset: f[1].parse().unwrap_or(0),
                count: f[2].parse().unwrap_or(0),
                dims,
            });
        }
    }
    out
}

pub(crate) fn read_probe(path: &Path) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
    let mut src = Vec::new();
    let mut trg = Vec::new();
    let mut mode = 0;
    let mut remaining = 0;
    if let Ok(fh) = File::open(path) {
        for line in BufReader::new(fh).lines().flatten() {
            let l = line.trim().to_string();
            if let Some(v) = l.strip_prefix("src_rows=") {
                remaining = v.parse().unwrap_or(0);
                mode = 0;
            } else if l.starts_with("src_cols=") {
                mode = 1;
            } else if let Some(v) = l.strip_prefix("trg_rows=") {
                remaining = v.parse().unwrap_or(0);
                mode = 0;
            } else if l.starts_with("trg_cols=") {
                mode = 2;
            } else if mode != 0 && remaining > 0 && !l.is_empty() {
                let row: Vec<i64> = l
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if mode == 1 {
                    src.push(row);
                } else {
                    trg.push(row);
                }
                remaining -= 1;
            }
        }
    }
    (src, trg)
}

/// Run the E2b conformance probe. Returns 0 on success.
pub fn run_e2b(dir: &str, tag: &str) -> i32 {
    let base = Path::new(dir);
    let arch = read_arch(&base.join("arch.txt"));
    if arch.is_empty() {
        eprintln!("E2b: cannot read {}/arch.txt", dir);
        return 1;
    }
    let man = read_manifest(&base.join("manifest.tsv"));
    if man.is_empty() {
        eprintln!("E2b: cannot read {}/manifest.tsv", dir);
        return 1;
    }
    // tch binds LibTorch, which stores linear weights [out, in] -- the same
    // convention the file declares, so no transpose is needed here.
    if arch.get("linear_layout").map(|s| s.as_str()) != Some("out_in") {
        eprintln!("E2b: unexpected linear_layout");
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
    cfg.dropout = 0.0;
    let (sv, tv) = (geti("vocab_src"), geti("vocab_trg"));

    let vs = VarStore::new(Device::Cpu);
    let model = Translator::new(&vs.root(), &cfg, sv, tv);

    let mut bin = match File::open(base.join("model.bin")) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("E2b: cannot open model.bin: {}", e);
            return 1;
        }
    };

    // Populate every variable by name. `freqs` is not a VarStore variable in
    // this implementation -- it is recomputed by the constructor -- which is
    // the intended behaviour: an incorrect rotation cannot borrow the
    // reference's table.
    let mut loaded = 0usize;
    let mut missing: Vec<String> = Vec::new();
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
            loaded += 1;
        }
    });
    if !missing.is_empty() {
        eprintln!("E2b: {} variable(s) not loaded:", missing.len());
        for m in missing.iter().take(12) {
            eprintln!("   {}", m);
        }
        return 1;
    }

    let (src_ids, trg_ids) = read_probe(&base.join("probe.txt"));
    if src_ids.is_empty() || trg_ids.is_empty() {
        eprintln!("E2b: probe.txt did not parse");
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

    let logits = tch::no_grad(|| model.forward(&src, &trg, false));
    let logits = logits.contiguous().to_kind(Kind::Float);
    let sizes = logits.size();
    let n: i64 = sizes.iter().product();
    let mut data = vec![0f32; n as usize];
    logits.copy_data(&mut data, n as usize);

    let outpath = base.join(format!("logits_{}.bin", tag));
    match File::create(&outpath) {
        Ok(mut f) => {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in &data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            if f.write_all(&bytes).is_err() {
                eprintln!("E2b: write failed");
                return 1;
            }
        }
        Err(e) => {
            eprintln!("E2b: cannot create {:?}: {}", outpath, e);
            return 1;
        }
    }

    let sum: f64 = data
        .iter()
        .map(|v| *v as f64)
        .sum();
    let mn = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "{}: shape={:?} params={} sum={:.6} min={:.6} max={:.6}",
        tag,
        sizes,
        loaded,
        sum,
        mn,
        mx
    );
    println!("  wrote {:?}", outpath);
    0
}
