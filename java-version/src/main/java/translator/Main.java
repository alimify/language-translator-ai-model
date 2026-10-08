package translator;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import translator.model.Translator;

import java.io.File;
import java.util.*;

/**
 * CLI entry point.
 * Matches Python: main.py
 */
public class Main {

    public static void main(String[] args) throws Exception {
        // Force UTF-8 console I/O so Bengali output is not corrupted by the
        // platform default charset (pre-JEP-400 JVMs default to cp1252 on
        // Windows). Python's CLI is UTF-8 via PYTHONUTF8/PYTHONIOENCODING.
        System.setOut(new java.io.PrintStream(new java.io.FileOutputStream(java.io.FileDescriptor.out), true, "UTF-8"));
        System.setErr(new java.io.PrintStream(new java.io.FileOutputStream(java.io.FileDescriptor.err), true, "UTF-8"));

        Config cfg = new Config();
        String cmd = args.length > 0 ? args[0] : "";

        // Device selection
        Device device = cuda_is_available() ? new Device(DeviceType.CUDA) : new Device(DeviceType.CPU);

        switch (cmd) {
            case "train": {
                List<String[]> data = DataLoader.get_raw_data(50000);
                Training.run_training(cfg, data, device);
                break;
            }

            case "chat": {
                SPTokenizer en_tok = new SPTokenizer(cfg.vocab_size);
                SPTokenizer bn_tok = new SPTokenizer(cfg.vocab_size);

                if (!en_tok.load("en_tok.model") || !bn_tok.load("bn_tok.model")) {
                    System.out.println("Model file not found. Run 'train' first.");
                    return;
                }

                // Load model
                Translator m = new Translator(cfg, en_tok.vocab_size(), bn_tok.vocab_size());
                try {
                    InputArchive archive = new InputArchive();
                    archive.load_from("translator.pth");
                    m.load(archive);
                } catch (Exception e) {
                    System.out.println("Model file not found. Run 'train' first.");
                    return;
                }
                m.to(device, false);

                System.out.println("Model Loaded! (Vocab: " + en_tok.vocab_size() + " words)");
                System.out.println("--------------------------------------------------");

                Scanner scanner = new Scanner(System.in, "UTF-8");
                while (true) {
                    System.out.print("\nEN (type 'exit'): ");
                    if (!scanner.hasNextLine())
                        break;
                    String t = scanner.nextLine().trim();

                    if (t.toLowerCase().equals("exit"))
                        break;
                    if (t.isEmpty())
                        continue;

                    String pred = Inference.translate_sentence(m, en_tok, bn_tok, t, 3, 0.6f, 1.4f);
                    System.out.println("BN: " + pred);
                }
                break;
            }

            case "e2b": {
                // E2b numeric conformance (thesis 8.8). Needs no trained
                // checkpoint: weights come from the shared neutral file.
                String dir = args.length > 1 ? args[1] : "../../analysis/e2b";
                System.exit(E2b.run(dir, "java"));
                return;
            }
            case "e2c": {
                // E2c gradient conformance (thesis 8.9). Like e2b, needs no
                // trained checkpoint: weights come from the shared file.
                String dir = args.length > 1 ? args[1] : "../../analysis/e2b";
                System.exit(E2c.run(dir, "java"));
            }

            case "validate": {
                // Equivalence-protocol validation mode (E1 tokenizer IDs, E2
                // shape checkpoints, E3 beam trace) using the trained
                // checkpoint. The output schema is language-agnostic and
                // machine-diffable. Mirrors Python main.py's `validate` branch.
                SPTokenizer en_tok = new SPTokenizer(cfg.vocab_size);
                SPTokenizer bn_tok = new SPTokenizer(cfg.vocab_size);

                if (!en_tok.load("en_tok.model") || !bn_tok.load("bn_tok.model")) {
                    System.out.println("Model file not found. Run 'train' first.");
                    return;
                }

                Translator m = new Translator(cfg, en_tok.vocab_size(), bn_tok.vocab_size());
                try {
                    InputArchive archive = new InputArchive();
                    archive.load_from("translator.pth");
                    m.load(archive);
                } catch (Exception e) {
                    System.out.println("Model file not found. Run 'train' first.");
                    return;
                }
                m.to(device, false);
                m.eval();

                System.out.println("VALIDATE START lang=java");
                System.out.println("VOCAB en=" + en_tok.vocab_size() + " bn=" + bn_tok.vocab_size());

                // E1: canonical strings -> encoder input IDs ([SOS] + ids + [EOS])
                String[][] e1_cases = {
                        { "T1", "Hello, world!" },
                        { "T2", "I love programming." },
                        { "T3", "Neural machine translation is challenging." },
                        { "T4", "Data-driven systems require careful evaluation." },
                };
                for (String[] c : e1_cases) {
                    List<Long> enc = en_tok.encode(c[1]);
                    StringBuilder ids = new StringBuilder("[1");
                    for (Long id : enc)
                        ids.append(", ").append(id);
                    ids.append(", 2]");
                    System.out.println("E1 " + c[0] + " ids=" + ids);
                }

                // E2: shape checkpoints for a fixed 2-sentence batch
                try (NoGradGuard no_grad = new NoGradGuard()) {
                    Tensor[] seqs = new Tensor[2];
                    for (int i = 0; i < 2; i++) {
                        List<Long> enc = en_tok.encode(e1_cases[i][1]);
                        long[] ids = new long[enc.size() + 2];
                        ids[0] = 1;
                        for (int k = 0; k < enc.size(); k++)
                            ids[k + 1] = enc.get(k);
                        ids[ids.length - 1] = 2;
                        seqs[i] = tensor(ids);
                    }
                    Tensor src = pad_sequence(new TensorVector(seqs), true, 0.0, "right");
                    Tensor trg = tensor(new long[] { 1, 4, 5, 2, 1, 6, 7, 2 }).reshape(2, 4);
                    System.out.println("E2 enc_input_ids shape=" + Translator.shapeList(src));
                    m.printShapeCheckpoints(src, trg);
                    System.out.println("E2 dec_input_ids shape=" + Translator.shapeList(trg));
                    Tensor logits = m.forward(src, trg);
                    System.out.println("E2 dec_logits shape=" + Translator.shapeList(logits));
                }

                // E3: beam-search decision trace for a fixed input
                System.out.println(DecodingPolicy.e3Line());
                String result = Inference.translate_sentence(m, en_tok, bn_tok, "Hello.",
                        DecodingPolicy.BEAM, DecodingPolicy.LP, DecodingPolicy.REP_PENALTY, true);
                System.out.println("E3 result=" + result);
                System.out.println("VALIDATE END");
                break;
            }

            default:
                System.out.println("Usage: java -jar translator.jar [train|chat|validate|e2b <dir>|e2c <dir>]");
                break;
        }
    }
}
