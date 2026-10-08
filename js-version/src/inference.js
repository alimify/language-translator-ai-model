/**Beam search inference with length and repetition penalties.*/
import { tf } from "./tf.js";

/**Translate an English sentence to Bengali using beam search.
 * 
 * Args:
 *     m: Trained Translator model
 *     en_tok: English tokenizer
 *     bn_tok: Bengali tokenizer
 *     sentence: Input English sentence
 *     beam_size: Number of beams to maintain
 *     lp: Length penalty exponent (default 0.6)
 *     rep_penalty: Repetition penalty factor (default 1.4)
 *         - First occurrence: divided/multiplied by 1.4
 *         - Subsequent occurrences: divided/multiplied by 1.4 * 1.5
 *         
 * Returns:
 *     Translated Bengali sentence
 */
/**
 * The decoding policy actually in force.
 *
 * One definition, used by the decoder and read by the E3 conformance check.
 * Previously that check logged these five values as a string literal while the
 * decoder used its own defaults, so the two could disagree without anything
 * failing -- the check could not detect a divergence in the policy it claimed
 * to verify (thesis 9.5). Changing a value here now changes both what runs and
 * what is reported.
 */
const DECODING_POLICY = Object.freeze({
    beam: 3,
    lp: 0.6,
    rep_penalty: 1.4,
    max_steps: 30,
    temperature: 0.8,
});

function translate_sentence(
    m,
    en_tok,
    bn_tok,
    sentence,
    beam_size = DECODING_POLICY.beam,
    lp = DECODING_POLICY.lp,
    rep_penalty = DECODING_POLICY.rep_penalty,
    trace = false
) {
    // Encode source sentence
    const tokens = en_tok.encode(sentence);
    if (!tokens || tokens.length === 0) return "";

    const src = tf.tensor2d([[1, ...tokens, 2]], undefined, "int32");

    // Initialize beams: (normalized_score, log_sum, sequence)
    let beams = [[0.0, 0.0, [1]]];

    for (let step = 0; step < DECODING_POLICY.max_steps; step++) {
        // Maximum generation steps
        const new_beams = [];

        for (const [_, log_sum, seq] of beams) {
            // If sequence ended, keep it as-is
            if (seq[seq.length - 1] === 2) {
                new_beams.push([
                    log_sum / Math.pow(seq.length, lp),
                    log_sum,
                    seq,
                ]);
                continue;
            }

            // tf.tidy disposes every intermediate tensor created by the
            // forward pass and top-k selection (the model forward allocates
            // many internal activations that must not leak per beam step).
            const { top_values, top_indices } = tf.tidy(() => {
                const trg_tensor = tf.tensor2d([seq], undefined, "int32");
                const logits = m.forward(src, trg_tensor, false);

                // Get last position logits: logits[0, -1, :], temperature /0.8
                const seq_len = seq.length;
                const curr_logits = logits
                    .slice([0, seq_len - 1, 0], [1, 1, -1])
                    .reshape([-1])
                    .div(DECODING_POLICY.temperature)
                    .arraySync();

                // Apply repetition penalty
                const token_set = new Set(seq);
                for (const token of token_set) {
                    if (token > 2) {
                        // Skip special tokens
                        const count = seq.filter((t) => t === token).length;
                        // Stronger penalty for repeated tokens
                        const penalty =
                            count === 1 ? rep_penalty : rep_penalty * 1.5;

                        if (curr_logits[token] > 0) {
                            curr_logits[token] /= penalty;
                        } else {
                            curr_logits[token] *= penalty;
                        }
                    }
                }

                // Get top-k candidates
                const probs = tf.logSoftmax(tf.tensor1d(curr_logits));
                const { values, indices } = tf.topk(probs, beam_size);
                return {
                    top_values: values.arraySync(),
                    top_indices: indices.arraySync(),
                };
            });

            for (let i = 0; i < beam_size; i++) {
                const t_id = top_indices[i];
                const new_sum = log_sum + top_values[i];
                const new_seq = [...seq, t_id];
                const normalized_score = new_sum / Math.pow(new_seq.length, lp);
                new_beams.push([normalized_score, new_sum, new_seq]);
            }
        }

        // Keep top beams
        new_beams.sort((a, b) => b[0] - a[0]);
        beams = new_beams.slice(0, beam_size);

        // Structured beam trace (equivalence protocol E3)
        if (trace) {
            beams.forEach(([ns, _, s], rank) => {
                console.log(
                    `E3 step=${step + 1} rank=${rank} tok=${s[s.length - 1]} score=${ns.toFixed(6)}`
                );
            });
        }

        // Early stopping if all beams ended
        if (beams.every(([_, __, s]) => s[s.length - 1] === 2)) {
            break;
        }
    }

    // Dispose source tensor
    src.dispose();

    return bn_tok.decode(beams[0][2]);
}

export { translate_sentence, DECODING_POLICY };
