/**SentencePiece tokenizer with serialization support.*/
import fs from "fs";
import { SentencePieceProcessor } from "@sctg/sentencepiece-js";
import { execFileSync } from "child_process";

/**Count SentencePiece vocabulary entries in a serialized ModelProto.
 *
 * The @sctg/sentencepiece-js wasm wrapper does not expose GetPieceSize, so
 * the actual trained piece count (Python: sp.GetPieceSize()) is derived by
 * counting top-level `pieces` records (field 1, length-delimited) in the
 * protobuf. Embedded message bodies are skipped, so only top-level fields
 * are considered.
 */
function count_model_pieces(buf) {
    let i = 0;
    let count = 0;
    const readVarint = () => {
        let v = 0;
        let shift = 0;
        for (;;) {
            const b = buf[i++];
            v += (b & 0x7f) * Math.pow(2, shift);
            if (!(b & 0x80)) break;
            shift += 7;
        }
        return v;
    };
    while (i < buf.length) {
        const key = readVarint();
        const field = Math.floor(key / 8);
        const wire = key & 7;
        if (wire === 2) {
            const len = readVarint();
            if (field === 1) count++;
            i += len;
        } else if (wire === 0) {
            readVarint();
        } else if (wire === 5) {
            i += 4;
        } else if (wire === 1) {
            i += 8;
        } else {
            break;
        }
    }
    return count;
}

class SPTokenizer {
    /**SentencePiece tokenizer wrapper supporting English and Bengali.
     * 
     * Special tokens are handled by SentencePiece:
     *     - <PAD>: 0
     *     - <SOS>: 1
     *     - <EOS>: 2
     *     - <UNK>: 3
     */

    constructor(vocab_size) {
        this._vocab_size = vocab_size;
        this._actual_vocab_size = vocab_size;
        this.sp = null;
        this.model_content = null; // Store model binary for serialization
    }

    /**Train SentencePiece model on texts.
     * 
     * Matches Python: def train(self, texts)
     * 
     * Args:
     *     texts: Array of strings to train on
     */
    async train(texts) {
        // Create temp file for training (trailing newline matches Python,
        // which writes each text as `t + "\n"`)
        fs.writeFileSync("temp_train.txt", texts.join("\n") + "\n", "utf-8");

        // Train model
        const model_prefix = "sp_model";

        const trainCode = [
            "import sentencepiece as spm",
            "spm.SentencePieceTrainer.train(",
            "input='temp_train.txt',",
            "model_prefix='sp_model',",
            `vocab_size=${this._vocab_size},`,
            "model_type='unigram',",
            "pad_id=0,bos_id=1,eos_id=2,unk_id=3,",
            "pad_piece='<PAD>',bos_piece='<SOS>',eos_piece='<EOS>',unk_piece='<UNK>',",
            "character_coverage=1.0,",
            "hard_vocab_limit=False)",
        ].join("\n");
        execFileSync("python", ["-c", trainCode], { stdio: "inherit" });

        // Load model content
        this.model_content = fs.readFileSync(`${model_prefix}.model`);
        this._actual_vocab_size = fs
            .readFileSync(`${model_prefix}.vocab`, "utf-8")
            .split(/\r?\n/)
            .filter((line) => line.trim().length > 0).length;

        // Load into processor
        this.sp = new SentencePieceProcessor();
        await this.sp.load(`${model_prefix}.model`);

        // Cleanup
        fs.unlinkSync("temp_train.txt");
        fs.unlinkSync(`${model_prefix}.model`);
        fs.unlinkSync(`${model_prefix}.vocab`);
    }

    /**Convert text to token IDs.
     * Matches Python: def encode(self, text) -> list[int]
     */
    encode(text) {
        return this.sp.encodeIds(text);
    }

    /**Convert token IDs back to text.
     * Matches Python: def decode(self, ids) -> str
     */
    decode(ids) {
        return this.sp.decodeIds(Int32Array.from(ids));
    }

    /**Get vocabulary size.
     * Matches Python: @property vocab_size
     */
    get vocab_size() {
        if (this.sp?.processor?.GetPieceSize) {
            return this.sp.processor.GetPieceSize();
        }
        return this._actual_vocab_size;
    }

    /**Save tokenizer model to file.
     * Matches Python: __getstate__ (serialization)
     */
    save(filename) {
        if (!this.model_content) return false;
        fs.writeFileSync(filename, this.model_content);
        return true;
    }

    /**Load tokenizer model from file.
     * Matches Python: __setstate__ (deserialization)
     */
    async load(filename) {
        if (!fs.existsSync(filename)) return false;
        this.model_content = fs.readFileSync(filename);

        // Write temp model file and load
        const tmp = `_tmp_sp_${Date.now()}.model`;
        fs.writeFileSync(tmp, this.model_content);
        this.sp = new SentencePieceProcessor();
        await this.sp.load(tmp);
        // Actual trained piece count (Python: sp.GetPieceSize()), derived
        // from the model protobuf because the wasm wrapper does not expose it.
        this._actual_vocab_size = count_model_pieces(this.model_content);
        fs.unlinkSync(tmp);
        return true;
    }
}

export { SPTokenizer };
