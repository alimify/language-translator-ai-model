/**CLI entry point for training and chat modes.
 * 
 * Matches Python: main.py
 */
import fs from "fs";
import readline from "readline";
import { Config } from "./src/config.js";
import { SPTokenizer } from "./src/tokenizer.js";
import { Translator } from "./src/model/translator.js";
import { get_raw_data } from "./src/data.js";
import { run_training } from "./src/training.js";
import { translate_sentence, DECODING_POLICY } from "./src/inference.js";

async function main() {
    const cfg = new Config();
    const cmd = process.argv[2] || "";

    if (cmd === "train") {
        const data = await get_raw_data(50000);
        await run_training(cfg, data);
    } else if (cmd === "chat") {
        // Load tokenizers
        const en_tok = new SPTokenizer(cfg.vocab_size);
        const bn_tok = new SPTokenizer(cfg.vocab_size);

        if (!(await en_tok.load("en_tok.model")) || !(await bn_tok.load("bn_tok.model"))) {
            console.log("Model file not found. Run 'train' first.");
            process.exit();
        }

        if (!fs.existsSync("translator.json")) {
            console.log("Model file not found. Run 'train' first.");
            process.exit();
        }

        // Load model weights
        const ck = JSON.parse(fs.readFileSync("translator.json", "utf8"));
        const m = new Translator(cfg, en_tok.vocab_size, bn_tok.vocab_size);
        m.loadWeights(ck.m);

        console.log(`Model Loaded! (Vocab: ${en_tok.vocab_size} words)`);
        console.log("--------------------------------------------------");

        // Interactive loop
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        const ask = () => {
            rl.question("\nEN (type 'exit'): ", (t) => {
                t = t.trim();
                if (t.toLowerCase() === "exit") {
                    rl.close();
                    return;
                }
                if (!t) {
                    ask();
                    return;
                }
                const pred = translate_sentence(m, en_tok, bn_tok, t);
                console.log(`BN: ${pred}`);
                ask();
            });
        };

        ask();
    } else if (cmd === "validate") {
        // Equivalence-protocol validation mode (E1/E2/E3), language-agnostic
        // machine-diffable output schema.
        const { tf } = await import("./src/tf.js");

        const en_tok = new SPTokenizer(cfg.vocab_size);
        const bn_tok = new SPTokenizer(cfg.vocab_size);
        if (!(await en_tok.load("en_tok.model")) || !(await bn_tok.load("bn_tok.model")) ||
            !fs.existsSync("translator.json")) {
            console.log("Model file not found. Run 'train' first.");
            process.exit();
        }
        const ck = JSON.parse(fs.readFileSync("translator.json", "utf8"));
        const m = new Translator(cfg, en_tok.vocab_size, bn_tok.vocab_size);
        m.loadWeights(ck.m);

        console.log("VALIDATE START lang=nodejs");
        console.log(`VOCAB en=${en_tok.vocab_size} bn=${bn_tok.vocab_size}`);

        // E1: canonical strings -> encoder input IDs ([SOS] + ids + [EOS])
        const e1_cases = [
            ["T1", "Hello, world!"],
            ["T2", "I love programming."],
            ["T3", "Neural machine translation is challenging."],
            ["T4", "Data-driven systems require careful evaluation."],
        ];
        for (const [tid, text] of e1_cases) {
            const ids = [1, ...en_tok.encode(text), 2];
            console.log(`E1 ${tid} ids=[${ids.join(", ")}]`);
        }

        // E2: shape checkpoints for a fixed 2-sentence batch
        const seqs = e1_cases.slice(0, 2).map(([_, t]) => [1, ...en_tok.encode(t), 2]);
        const maxLen = Math.max(...seqs.map((s) => s.length));
        const padded = seqs.map((s) => s.concat(new Array(maxLen - s.length).fill(0)));
        const src = tf.tensor2d(padded, [2, maxLen], "int32");
        const trg = tf.tensor2d([[1, 4, 5, 2], [1, 6, 7, 2]], [2, 4], "int32");
        console.log(`E2 enc_input_ids shape=[${src.shape.join(", ")}]`);
        tf.tidy(() => {
            const scale = Math.sqrt(m.dim);
            const emb = tf.mul(tf.gather(m.se, src), scale);
            console.log(`E2 enc_embed shape=[${emb.shape.join(", ")}]`);
            const sm = tf.cast(tf.notEqual(src, 0), "float32").expandDims(1).expandDims(2);
            let e = emb;
            for (const layer of m.enc) {
                e = layer.forward(e, null, sm, null, m.freqs, false);
            }
            console.log(`E2 enc_out shape=[${e.shape.join(", ")}]`);
            console.log(`E2 dec_input_ids shape=[${trg.shape.join(", ")}]`);
            const logits = m.forward(src, trg, false);
            console.log(`E2 dec_logits shape=[${logits.shape.join(", ")}]`);
        });
        src.dispose();
        trg.dispose();

        // E3: beam-search decision trace for a fixed input
        const dp = DECODING_POLICY;
        console.log(`E3 params beam=${dp.beam} lp=${dp.lp} rp=${dp.rep_penalty} ` +
                    `max_steps=${dp.max_steps} temp=${dp.temperature}`);
        const result = translate_sentence(m, en_tok, bn_tok, "Hello.",
                                          dp.beam, dp.lp, dp.rep_penalty, true);
        console.log(`E3 result=${result}`);
        console.log("VALIDATE END");
    } else {
        console.log("Usage: node main.js [train|chat|validate]");
    }
}

main();
