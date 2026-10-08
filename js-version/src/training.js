/**Training loop with OneCycleLR scheduler and gradient clipping.*/
import fs from "fs";
import { tf, backendName } from "./tf.js";
import { SPTokenizer } from "./tokenizer.js";
import { Translator } from "./model/translator.js";

const TRAINING_SEED = 42;

function createRng(seed) {
    let state = seed >>> 0;
    return () => {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 0x100000000;
    };
}

function mulberry32(a) {
    return function () {
        a |= 0; a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function shuffleInPlace(items, rng) {
    for (let i = items.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [items[i], items[j]] = [items[j], items[i]];
    }
}

/**OneCycleLR scheduler implementation matching PyTorch.
 * 
 * Matches Python: torch.optim.lr_scheduler.OneCycleLR
 *     - div_factor = 25 (initial_lr = max_lr / 25)
 *     - final_div_factor = 10000 (min_lr = max_lr / (25 * 10000))
 *     - pct_start = 0.3
 *     - anneal_strategy = 'cos'
 * 
 * Args:
 *     max_lr: Maximum learning rate
 *     total_steps: Total number of training steps
 *     pct_start: Percentage of steps for warmup phase (default 0.3)
 */
class OneCycleLR {
    constructor(max_lr, total_steps, pct_start = 0.3) {
        this.max_lr = max_lr;
        this.initial_lr = max_lr / 25.0; // div_factor = 25
        this.min_lr = this.initial_lr / 10000.0; // final_div_factor = 1e4
        // PyTorch phase boundaries: [pct_start*total_steps - 1, total_steps - 1]
        this.phase1_end = pct_start * total_steps - 1.0;
        this.phase2_end = total_steps - 1.0;
        this.step_count = 0;
        this._last_lr = this.initial_lr;
    }

    /**Cosine annealing between two values (PyTorch _annealing_cos).*/
    static _anneal_cos(start, end, pct) {
        return end + ((start - end) / 2.0) * (Math.cos(Math.PI * pct) + 1.0);
    }

    /**Learning rate and beta1 (cycled momentum) for the current step.
     * anneal_strategy='cos' applies to BOTH phases in PyTorch; with an
     * Adam-family optimizer, cycle_momentum=True cycles betas[0] between
     * max_momentum=0.95 and base_momentum=0.85, inversely to the LR.
     */
    get_lr() {
        const s = this.step_count;
        let lr, beta1;
        if (s <= this.phase1_end) {
            const pct = s / this.phase1_end;
            lr = OneCycleLR._anneal_cos(this.initial_lr, this.max_lr, pct);
            beta1 = OneCycleLR._anneal_cos(0.95, 0.85, pct);
        } else {
            const pct = (s - this.phase1_end) / (this.phase2_end - this.phase1_end);
            lr = OneCycleLR._anneal_cos(this.max_lr, this.min_lr, pct);
            beta1 = OneCycleLR._anneal_cos(0.85, 0.95, pct);
        }
        this._last_lr = lr;
        return { lr, beta1 };
    }

    step() {
        this.step_count++;
    }

    get_last_lr() {
        return this._last_lr;
    }
}

/**AdamW optimizer matching torch.optim.AdamW semantics exactly:
 * decoupled weight decay applied to ALL parameters BEFORE the Adam update,
 * eps=1e-8, and bias correction computed from the CURRENT beta1 (so that
 * OneCycleLR momentum cycling behaves as in PyTorch).
 */
class AdamW {
    constructor(vars, { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01 } = {}) {
        this.vars = vars;
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weight_decay = weight_decay;
        this.step_count = 0;
        this.m = vars.map((v) => tf.variable(tf.zerosLike(v), false));
        this.v = vars.map((v) => tf.variable(tf.zerosLike(v), false));
    }

    /**Apply one update. grads is an array aligned with this.vars.*/
    step(grads) {
        this.step_count += 1;
        const bc1 = 1 - Math.pow(this.beta1, this.step_count);
        const bc2 = 1 - Math.pow(this.beta2, this.step_count);
        for (let i = 0; i < this.vars.length; i++) {
            const g = grads[i];
            if (!g) continue;
            const p = this.vars[i];
            tf.tidy(() => {
                // Decoupled weight decay: p *= (1 - lr*wd), before the update
                p.assign(tf.mul(p, 1 - this.lr * this.weight_decay));
                this.m[i].assign(
                    tf.add(tf.mul(this.m[i], this.beta1), tf.mul(g, 1 - this.beta1))
                );
                this.v[i].assign(
                    tf.add(tf.mul(this.v[i], this.beta2), tf.mul(tf.square(g), 1 - this.beta2))
                );
                const mhat = tf.div(this.m[i], bc1);
                const vhat = tf.div(this.v[i], bc2);
                p.assign(
                    tf.sub(p, tf.div(tf.mul(mhat, this.lr), tf.add(tf.sqrt(vhat), this.eps)))
                );
            });
        }
    }
}

/**Pad sequences to same length with zeros (PAD=0).
 * 
 * Matches Python: nn.utils.rnn.pad_sequence(seqs, batch_first=True)
 * 
 * Args:
 *     sequences: Array of arrays of integer token IDs
 *     
 * Returns:
 *     2D tensor of shape (batch, max_len) padded with 0s
 */
/**Label-smoothed cross-entropy with padding index 0 ignored.
 *
 * Extracted from the training loop rather than restated by callers, so that the
 * E2c gradient probe (analysis/e2b/run_nodejs_grad.mjs) measures the loss this
 * implementation actually trains against. A probe that restates the objective
 * measures the probe; that failure mode is documented in thesis 9.5 and this
 * function exists to avoid a fifth instance of it.
 *
 * Equivalent to torch.nn.functional.cross_entropy(..., ignore_index=0,
 * label_smoothing=eps): with smoothed targets one_hot*(1-eps) + eps/V,
 *   -sum(smoothed * log_probs) = -((1-eps)*logp[target] + (eps/V)*sum_j logp[j])
 * which avoids materializing an [N, V] one-hot per step.
 *
 * Args:
 *     logits: [B, T, V] decoder logits
 *     trg_target: [B, T] target token ids (trg[:, 1:])
 *     label_smoothing: smoothing factor (0.1)
 *
 * Returns:
 *     Scalar loss tensor
 */
function label_smoothed_ce(logits, trg_target, label_smoothing) {
    const vocab_size = logits.shape[2];
    const logits_flat = logits.reshape([-1, vocab_size]);
    const targets_flat = trg_target.reshape([-1]).toInt();

    const log_probs = tf.logSoftmax(logits_flat);
    const n_rows = logits_flat.shape[0];
    // Flat 1-D gather (tf.gather has a registered gradient in tfjs; gatherND
    // does not): index of [row, target] in the flattened [N*V] view is
    // row*V + target.
    const row_starts = tf.range(0, n_rows * vocab_size, vocab_size, "int32");
    const flat_idx = tf.add(row_starts, targets_flat);
    const logp_target = tf.gather(log_probs.reshape([-1]), flat_idx);
    const sum_logp = tf.sum(log_probs, -1);
    let loss = tf.neg(
        tf.add(
            tf.mul(logp_target, 1 - label_smoothing),
            tf.mul(sum_logp, label_smoothing / vocab_size)
        )
    );

    // Mask out padding (index 0), matching ignore_index=0.
    const mask = tf.cast(tf.notEqual(targets_flat, 0), "float32");
    return tf.div(tf.sum(tf.mul(loss, mask)), tf.maximum(tf.sum(mask), 1));
}

function pad_sequence(sequences) {
    const max_len = Math.max(...sequences.map((s) => s.length));
    const padded = sequences.map((s) => {
        const pad = new Array(max_len - s.length).fill(0);
        return s.concat(pad);
    });
    return tf.tensor2d(padded, [sequences.length, max_len], "int32");
}

/**Clip gradient norm of trainable variables to max_norm.
 * 
 * Matches Python: torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
 * 
 * Args:
 *     grads: Object mapping variable names to gradient tensors
 *     max_norm: Maximum gradient norm (default 1.0)
 *     
 * Returns:
 *     Clipped gradients object
 */
function clip_grad_norm(grads, max_norm = 1.0) {
    // Compute the global norm with a single device->host sync (one dataSync)
    // instead of one per gradient; semantics identical to
    // torch.nn.utils.clip_grad_norm_(parameters, max_norm).
    const totalNorm = tf.tidy(() => {
        const sq = Object.keys(grads).map((name) => tf.sum(tf.square(grads[name])));
        return Math.sqrt(tf.addN(sq).dataSync()[0]);
    });
    const clipCoeff = max_norm / (totalNorm + 1e-6);

    if (clipCoeff < 1.0) {
        for (const name in grads) {
            const original = grads[name];
            grads[name] = tf.mul(original, clipCoeff);
            original.dispose();
        }
    }
    return grads;
}

/**Run full training loop.
 * 
 * Matches Python: def run_training(cfg, data)
 * 
 * Features:
 *     - 95/5 train/val split
 *     - AdamW optimizer with weight_decay=0.01
 *     - OneCycleLR scheduler with pct_start=0.3, cosine annealing
 *     - Gradient clipping with max_norm=1.0
 *     - Label smoothing
 *     - Model checkpoint saving
 * 
 * Args:
 *     cfg: Config object with training hyperparameters
 *     data: Array of [english, bengali] sentence pairs
 */
async function run_training(cfg, data) {
    if (!data || data.length === 0) return;
    const rng = createRng(TRAINING_SEED);

    // Shuffle and split data (95% train, 5% val)
    // Hold out the final 200 pairs; seed the shuffle for cross-
    // implementation identity (see analysis/patch_disjoint_split.py).
    if (data.length > 200) data.length = data.length - 200;
    console.log(`Split: pool=${data.length} seed=20260815`);
    shuffleInPlace(data, mulberry32(20260815));
    const split = Math.floor(data.length * 0.95);
    const train_data = data.slice(0, split);

    // Remove existing checkpoint
    if (fs.existsSync("translator.json")) {
        fs.unlinkSync("translator.json");
    }

    // Tokenizers: load the shared frozen artifacts when present (thesis
    // equivalence protocol E1 requires one tokenizer artifact across all
    // language implementations); otherwise train from the local split.
    const en_tok = new SPTokenizer(cfg.vocab_size);
    const bn_tok = new SPTokenizer(cfg.vocab_size);
    if ((await en_tok.load("en_tok.model")) && (await bn_tok.load("bn_tok.model"))) {
        console.log("Loaded shared tokenizer artifacts (en_tok.model, bn_tok.model)");
    } else {
        await en_tok.train(train_data.map(([e, _]) => e));
        await bn_tok.train(train_data.map(([_, b]) => b));
    }

    // Initialize model
    const m = new Translator(cfg, en_tok.vocab_size, bn_tok.vocab_size);

    // OneCycleLR scheduler + AdamW (weight_decay=0.01), matching PyTorch
    const total_steps =
        cfg.epochs * (Math.floor(train_data.length / cfg.batch_size) + 1);
    const scheduler = new OneCycleLR(cfg.lr, total_steps, 0.3);
    const trainable_vars = m.getWeights();
    const optimizer = new AdamW(trainable_vars, { lr: cfg.lr, weight_decay: 0.01 });
    const batches_per_epoch = Math.ceil(train_data.length / cfg.batch_size);
    const progress_every = Math.max(1, Math.floor(batches_per_epoch / 10));

    console.log(
        `Training (Data: ${data.length} pairs | Steps: ${total_steps} | Backend: ${backendName} | Seed: ${TRAINING_SEED})`
    );

    for (let ep = 0; ep < cfg.epochs; ep++) {
        // Shuffle train_data each epoch
        shuffleInPlace(train_data, rng);

        let total_loss = 0;
        let batch_count = 0;
        const epoch_start = Date.now();

        for (let i = 0; i < train_data.length; i += cfg.batch_size) {
            const batch = train_data.slice(i, i + cfg.batch_size);
            if (batch.length < 2) continue;

            // Learning rate and cycled beta1 for this step
            const { lr, beta1 } = scheduler.get_lr();
            optimizer.lr = lr;
            optimizer.beta1 = beta1;

            // Prepare batch tensors with SOS and EOS tokens
            const src_seqs = batch.map(([e, _]) => [1, ...en_tok.encode(e), 2]);
            const trg_seqs = batch.map(([_, b]) => [1, ...bn_tok.encode(b), 2]);

            const src = pad_sequence(src_seqs);
            const trg = pad_sequence(trg_seqs);

            // Compute loss and gradients manually
            const trainable = m.getWeights();

            const lossFunc = () => {
                // trg[:, :-1] as decoder input
                const trg_input = trg.slice(
                    [0, 0],
                    [batch.length, trg.shape[1] - 1]
                );
                // trg[:, 1:] as target labels
                const trg_target = trg.slice(
                    [0, 1],
                    [batch.length, trg.shape[1] - 1]
                );

                const logits = m.forward(src, trg_input, true);
                return label_smoothed_ce(logits, trg_target, cfg.label_smoothing);
            };

            // Compute gradients
            const { value: loss_val, grads } = tf.variableGrads(lossFunc, trainable);

            // Gradient clipping (max_norm=1.0)
            const clipped_grads = clip_grad_norm(grads, 1.0);

            // AdamW step (decoupled weight decay + Adam update, PyTorch order)
            const grad_list = trainable.map((v) => clipped_grads[v.name]);
            optimizer.step(grad_list);

            scheduler.step();

            total_loss += loss_val.dataSync()[0];
            batch_count++;
            const batch_index = Math.floor(i / cfg.batch_size) + 1;
            if (
                batch_index === 1 ||
                batch_index === batches_per_epoch ||
                batch_index % progress_every === 0
            ) {
                const avg_so_far = total_loss / batch_count;
                console.log(
                    `Epoch ${ep + 1}/${cfg.epochs} | Batch ${batch_index}/${batches_per_epoch} | Loss: ${avg_so_far.toFixed(4)} | LR: ${lr.toFixed(7)}`
                );
            }

            // Dispose tensors
            src.dispose();
            trg.dispose();
            loss_val.dispose();
            for (const name in grads) {
                grads[name].dispose();
            }
        }

        // Epoch logging
        const avg_loss = batch_count > 0 ? total_loss / batch_count : 0;
        const current_lr = scheduler.get_last_lr();
        const elapsed_sec = ((Date.now() - epoch_start) / 1000).toFixed(1);
        console.log(
            `Epoch ${ep + 1}/${cfg.epochs} done | Loss: ${avg_loss.toFixed(4)} | LR: ${current_lr.toFixed(7)} | Time: ${elapsed_sec}s`
        );
    }

    // Save model and tokenizers
    const checkpoint = {
        m: m.saveWeights(),
        en: { vocab_size: en_tok._vocab_size, model_content: en_tok.model_content ? en_tok.model_content.toString("base64") : null },
        bn: { vocab_size: bn_tok._vocab_size, model_content: bn_tok.model_content ? bn_tok.model_content.toString("base64") : null },
        cfg: {
            max_len: cfg.max_len,
            dim: cfg.dim,
            heads: cfg.heads,
            layers: cfg.layers,
            ff_mult: cfg.ff_mult,
            vocab_size: cfg.vocab_size,
            lr: cfg.lr,
            epochs: cfg.epochs,
            batch_size: cfg.batch_size,
            label_smoothing: cfg.label_smoothing,
            dropout: cfg.dropout,
            seed: TRAINING_SEED,
            backend: backendName,
        },
    };
    fs.writeFileSync("translator.json", JSON.stringify(checkpoint));
    en_tok.save("en_tok.model");
    bn_tok.save("bn_tok.model");
    console.log("Saved to 'translator.json'");
}

export { run_training, label_smoothed_ce };
