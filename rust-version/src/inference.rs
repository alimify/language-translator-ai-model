//! Beam search inference with length and repetition penalties.
//!
//! Matches Python: inference.py

use std::collections::HashMap;
use tch::{ Device, Kind, Tensor, no_grad };
use crate::tokenizer::SPTokenizer;
use crate::model::Translator;

/// The decoding policy actually in force.
///
/// One definition, used by the decoder and read by the E3 conformance check.
/// Previously that check printed these five values as a string literal while
/// the decoder used its own arguments, so the two could disagree without
/// anything failing -- the check could not detect a divergence in the policy it
/// claimed to verify (thesis 9.5). Changing a value here now changes both what
/// runs and what is reported.
#[derive(Debug, Clone, Copy)]
pub struct DecodingPolicy {
    pub beam: usize,
    pub lp: f64,
    pub rep_penalty: f64,
    pub max_steps: i64,
    pub temperature: f64,
}

pub const DECODING_POLICY: DecodingPolicy = DecodingPolicy {
    beam: 3,
    lp: 0.6,
    rep_penalty: 1.4,
    max_steps: 30,
    temperature: 0.8,
};

/// Translate an English sentence to Bengali using beam search.
///
/// Matches Python: def translate_sentence(m, en_tok, bn_tok, sentence,
///                                         beam_size=3, lp=0.6, rep_penalty=1.4)
///
/// # Arguments
/// * `m` - Trained Translator model
/// * `en_tok` - English tokenizer
/// * `bn_tok` - Bengali tokenizer
/// * `sentence` - Input English sentence
/// * `beam_size` - Number of beams (default 3)
/// * `lp` - Length penalty exponent (default 0.6)
/// * `rep_penalty` - Repetition penalty factor (default 1.4)
/// * `device` - Computation device
pub fn translate_sentence(
    m: &Translator,
    en_tok: &SPTokenizer,
    bn_tok: &SPTokenizer,
    sentence: &str,
    beam_size: usize,
    lp: f64,
    rep_penalty: f64,
    device: Device,
    trace: bool
) -> String {
    let tokens = en_tok.encode(sentence);
    if tokens.is_empty() {
        return String::new();
    }

    // Create source tensor with SOS and EOS
    let mut src_vec: Vec<i64> = vec![1];
    src_vec.extend(tokens);
    src_vec.push(2);

    let src = Tensor::from_slice(&src_vec).unsqueeze(0).to_device(device);

    // Beam structure: (normalized_score, log_sum, sequence)
    #[derive(Clone)]
    struct Beam {
        normalized_score: f64,
        log_sum: f64,
        seq: Vec<i64>,
    }

    let mut beams = vec![Beam { normalized_score: 0.0, log_sum: 0.0, seq: vec![1] }];

    for step in 0..DECODING_POLICY.max_steps {
        let mut new_beams: Vec<Beam> = Vec::new();

        for beam in &beams {
            // If sequence ended with EOS, keep as-is
            if *beam.seq.last().unwrap() == 2 {
                let ns = beam.log_sum / (beam.seq.len() as f64).powf(lp);
                new_beams.push(Beam {
                    normalized_score: ns,
                    log_sum: beam.log_sum,
                    seq: beam.seq.clone(),
                });
                continue;
            }

            let result = no_grad(|| {
                let trg = Tensor::from_slice(&beam.seq).unsqueeze(0).to_device(device);

                let logits = m.forward(&src, &trg, false);

                // Get last step logits with temperature 0.8
                let last_idx = (beam.seq.len() as i64) - 1;
                let mut curr_logits =
                    logits.select(0, 0).select(0, last_idx) / DECODING_POLICY.temperature;

                // Apply repetition penalty
                let mut token_counts: HashMap<i64, i64> = HashMap::new();
                for &t in &beam.seq {
                    if t > 2 {
                        *token_counts.entry(t).or_insert(0) += 1;
                    }
                }

                for (&token, &count) in &token_counts {
                    let penalty = if count == 1 { rep_penalty } else { rep_penalty * 1.5 };
                    let val = f64::from(curr_logits.double_value(&[token]));
                    // Source dtype must match the f32 logits tensor
                    // (index_put_ requires matching dtypes).
                    let new_val = if val > 0.0 { val / penalty } else { val * penalty };
                    curr_logits = curr_logits.index_put_(
                        &[Some(Tensor::from_slice(&[token]))],
                        &Tensor::from_slice(&[new_val as f32]),
                        false
                    );
                }

                let probs = curr_logits.log_softmax(-1, Kind::Float);
                probs.topk(beam_size as i64, -1, true, true)
            });

            let (top_v, top_i) = result;

            for i in 0..beam_size {
                let t_id = top_i.int64_value(&[i as i64]);
                let score = f64::from(top_v.double_value(&[i as i64]));

                let new_sum = beam.log_sum + score;
                let mut new_seq = beam.seq.clone();
                new_seq.push(t_id);

                let normalized_score = new_sum / (new_seq.len() as f64).powf(lp);
                new_beams.push(Beam { normalized_score, log_sum: new_sum, seq: new_seq });
            }
        }

        // Sort by normalized score (descending)
        new_beams.sort_by(|a, b| b.normalized_score.partial_cmp(&a.normalized_score).unwrap());
        new_beams.truncate(beam_size);
        beams = new_beams;

        // Structured beam trace (equivalence protocol E3)
        if trace {
            for (rank, b) in beams.iter().enumerate() {
                println!(
                    "E3 step={} rank={} tok={} score={:.6}",
                    step + 1,
                    rank,
                    b.seq.last().unwrap(),
                    b.normalized_score
                );
            }
        }

        // Early stopping if all beams end with EOS
        if beams.iter().all(|b| *b.seq.last().unwrap() == 2) {
            break;
        }
    }

    bn_tok.decode(&beams[0].seq)
}
