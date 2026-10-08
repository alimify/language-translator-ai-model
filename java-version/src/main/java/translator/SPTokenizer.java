package translator;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.sentencepiece.SentencePieceProcessor;
import org.bytedeco.sentencepiece.SentencePieceTrainer;
import org.bytedeco.sentencepiece.IntVector;
import org.bytedeco.sentencepiece.Status;

import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * SentencePiece tokenizer wrapper.
 * Matches Python: class SPTokenizer
 *
 * Uses the in-process SentencePiece native bindings (org.bytedeco.sentencepiece)
 * exactly like Python uses the sentencepiece package in-process: training via
 * SentencePieceTrainer.Train, encode via EncodeAsIds semantics, decode via
 * DecodeIds semantics, and vocab_size() returning the ACTUAL trained piece
 * count (GetPieceSize), matching Python's `self.sp.GetPieceSize()`.
 */
public class SPTokenizer {

    private int _vocab_size;
    private byte[] model_content;
    private SentencePieceProcessor sp;

    public SPTokenizer(int vocab_size) {
        this._vocab_size = vocab_size;
        this.sp = new SentencePieceProcessor();
    }

    /**
     * Train tokenizer on texts.
     * Matches Python: def train(self, texts)
     */
    public void train(List<String> texts) {
        try {
            // Write temp training file (UTF-8, LF line endings, trailing
            // newline — matches Python's `f.write(t + "\n")`)
            try (BufferedWriter w = Files.newBufferedWriter(
                    Paths.get("temp_train.txt"), StandardCharsets.UTF_8)) {
                for (String t : texts) {
                    w.write(t);
                    w.write("\n");
                }
            }

            String model_prefix = "sp_model";
            String args = "--input=temp_train.txt"
                    + " --model_prefix=" + model_prefix
                    + " --vocab_size=" + _vocab_size
                    + " --model_type=unigram"
                    + " --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3"
                    + " --pad_piece=<PAD> --bos_piece=<SOS> --eos_piece=<EOS> --unk_piece=<UNK>"
                    + " --character_coverage=1.0"
                    + " --hard_vocab_limit=false";

            Status status = SentencePieceTrainer.Train(args);
            if (!status.ok()) {
                throw new RuntimeException("SentencePiece training failed: " + status.ToString());
            }

            // Read model content into memory, load it, then clean up
            model_content = Files.readAllBytes(Paths.get(model_prefix + ".model"));
            _load_model_from_content();

            Files.deleteIfExists(Paths.get("temp_train.txt"));
            Files.deleteIfExists(Paths.get(model_prefix + ".model"));
            Files.deleteIfExists(Paths.get(model_prefix + ".vocab"));

        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException("Training failed", e);
        }
    }

    /**
     * Encode text to token IDs.
     * Matches Python: def encode(self, text) -> list[int]
     */
    public List<Long> encode(String text) {
        IntVector ids = new IntVector();
        // The bindings only expose Encode(String, IntVector); JavaCPP passes
        // Java strings to native code as JNI modified UTF-8, which is
        // byte-identical to real UTF-8 for all BMP text (Bengali included),
        // so the IDs match Python's UTF-8 encode for this corpus.
        sp.Encode(text, ids);
        List<Long> result = new ArrayList<>((int) ids.size());
        for (long i = 0; i < ids.size(); i++) {
            result.add((long) ids.get(i));
        }
        return result;
    }

    /**
     * Decode token IDs to text.
     * Matches Python: def decode(self, ids) -> str
     */
    public String decode(List<Long> ids) {
        IntVector iv = new IntVector();
        iv.resize(ids.size());
        for (int i = 0; i < ids.size(); i++) {
            iv.put(i, ids.get(i).intValue());
        }
        try (BytePointer out = new BytePointer("")) {
            sp.Decode(iv, out);
            return out.getString(StandardCharsets.UTF_8);
        }
    }

    /**
     * Get actual vocabulary size from the trained model.
     * Matches Python: @property vocab_size -> self.sp.GetPieceSize()
     */
    public int vocab_size() {
        if (model_content == null) {
            return _vocab_size;
        }
        return sp.GetPieceSize();
    }

    /**
     * Load model from file.
     * Matches Python: __setstate__ (deserialization)
     */
    public boolean load(String filename) {
        try {
            Path path = Paths.get(filename);
            if (!Files.exists(path))
                return false;
            model_content = Files.readAllBytes(path);
            if (model_content.length == 0)
                return false;
            _load_model_from_content();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Save model to file.
     * Matches Python: __getstate__ (serialization)
     */
    public boolean save(String filename) {
        if (model_content == null || model_content.length == 0)
            return false;
        try {
            Files.write(Paths.get(filename), model_content);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private void _load_model_from_content() throws Exception {
        // The bindings load from a file path; write the in-memory bytes to a
        // temp file, load, then remove it (equivalent to Python's
        // LoadFromSerializedProto on the stored model bytes).
        Path tmp = Files.createTempFile("sptok_", ".model");
        Files.write(tmp, model_content);
        Status status = sp.Load(tmp.toString());
        Files.deleteIfExists(tmp);
        if (!status.ok()) {
            throw new RuntimeException("Failed to load trained model: " + status.ToString());
        }
    }
}
