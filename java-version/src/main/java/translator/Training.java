package translator;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import static org.bytedeco.pytorch.global.torch.*;
import translator.model.Translator;

import java.io.File;
import java.util.*;

/**
 * Training loop.
 * Matches Python: training.py
 * 
 * Features:
 * - 95/5 train/val split
 * - AdamW optimizer with weight_decay=0.01
 * - OneCycleLR scheduler with pct_start=0.3, cosine annealing
 * - Gradient clipping with max_norm=1.0
 * - Label smoothing
 * - Model checkpoint saving
 */
public class Training {

    /**
     * Run full training loop.
     * Matches Python: def run_training(cfg, data)
     */
    public static void run_training(Config cfg, List<String[]> data, Device device) {
        if (data.isEmpty())
            return;

        // Shuffle and split data (95% train, 5% val)
        // Hold out the final 200 pairs; seed the shuffle for cross-
        // implementation identity (see analysis/patch_disjoint_split.py).
        if (data.size() > 200) {
            data = new ArrayList<>(data.subList(0, data.size() - 200));
        }
        System.out.println("Split: pool=" + data.size() + " seed=20260815");
        Collections.shuffle(data, new java.util.Random(20260815L));
        int split = (int) (data.size() * 0.95);
        List<String[]> train_data = new ArrayList<>(data.subList(0, split));

        // Remove existing checkpoint
        new File("translator.pth").delete();

        // Tokenizers: load the shared frozen artifacts when present (thesis
        // equivalence protocol E1 requires one tokenizer artifact across all
        // language implementations); otherwise train from the local split.
        SPTokenizer en_tok = new SPTokenizer(cfg.vocab_size);
        SPTokenizer bn_tok = new SPTokenizer(cfg.vocab_size);

        if (en_tok.load("en_tok.model") && bn_tok.load("bn_tok.model")) {
            System.out.println("Loaded shared tokenizer artifacts (en_tok.model, bn_tok.model)");
        } else {
            List<String> en_texts = new ArrayList<>();
            List<String> bn_texts = new ArrayList<>();
            for (String[] pair : train_data) {
                en_texts.add(pair[0]);
                bn_texts.add(pair[1]);
            }
            en_tok.train(en_texts);
            bn_tok.train(bn_texts);
        }

        // Initialize model
        Translator m = new Translator(cfg, en_tok.vocab_size(), bn_tok.vocab_size());
        m.to(device, false);

        // Optimizer with weight decay
        AdamWOptions opts = new AdamWOptions(cfg.lr);
        opts.weight_decay().put(0.01);
        AdamW optimizer = new AdamW(m.parameters(), opts);

        // OneCycleLR scheduler, matching torch.optim.lr_scheduler.OneCycleLR
        // with anneal_strategy='cos', pct_start=0.3, div_factor=25 (default),
        // final_div_factor=1e4 (default), cycle_momentum=True (default; cycles
        // betas[0] between max_momentum=0.95 and base_momentum=0.85).
        int total_steps = cfg.epochs * (train_data.size() / cfg.batch_size + 1);
        final double max_lr = cfg.lr;
        final double initial_lr = max_lr / 25.0;
        final double min_lr = initial_lr / 10000.0;
        final double phase1_end = 0.3 * total_steps - 1.0;
        final double phase2_end = (double) total_steps - 1.0;

        // PyTorch sets the LR/momentum for step 0 at scheduler construction,
        // so the very first optimizer.step() runs at initial_lr, beta1=0.95.
        double[] lrBeta = oneCycle(0, phase1_end, phase2_end, initial_lr, max_lr, min_lr);
        applyLrBeta(optimizer, lrBeta[0], lrBeta[1]);

        System.out.printf("Training (Data: %d pairs | Steps: %d)%n", data.size(), total_steps);

        int global_step = 0;
        Random rng = new Random();

        for (int ep = 0; ep < cfg.epochs; ep++) {
            Collections.shuffle(train_data, rng);
            m.train(true);
            float total_loss = 0;
            int batch_count = 0;

            for (int i = 0; i < train_data.size(); i += cfg.batch_size) {
                // Build batch
                List<String[]> batch = new ArrayList<>();
                for (int j = i; j < i + cfg.batch_size && j < train_data.size(); j++) {
                    batch.add(train_data.get(j));
                }
                if (batch.size() < 2)
                    continue;

                // PointerScope frees all Java-side native tensor wrappers
                // created in this batch as soon as the scope closes. Without
                // it, native allocations accumulate until GC and the process
                // eventually crashes inside c10.dll (observed). ATen tensors
                // are refcounted, so releasing temporaries never touches the
                // module's own parameters. Behavior-neutral memory management.
                try (org.bytedeco.javacpp.PointerScope scope = new org.bytedeco.javacpp.PointerScope()) {

                // Prepare batch tensors with SOS and EOS tokens
                List<Tensor> src_list = new ArrayList<>();
                List<Tensor> trg_list = new ArrayList<>();

                for (String[] pair : batch) {
                    List<Long> en_enc = en_tok.encode(pair[0]);
                    long[] src_ids = new long[en_enc.size() + 2];
                    src_ids[0] = 1; // SOS
                    for (int k = 0; k < en_enc.size(); k++)
                        src_ids[k + 1] = en_enc.get(k);
                    src_ids[src_ids.length - 1] = 2; // EOS
                    src_list.add(tensor(src_ids));

                    List<Long> bn_enc = bn_tok.encode(pair[1]);
                    long[] trg_ids = new long[bn_enc.size() + 2];
                    trg_ids[0] = 1;
                    for (int k = 0; k < bn_enc.size(); k++)
                        trg_ids[k + 1] = bn_enc.get(k);
                    trg_ids[trg_ids.length - 1] = 2;
                    trg_list.add(tensor(trg_ids));
                }

                // pad_sequence (batch_first=True, pad with 0); tensors are
                // created on CPU which is the device in use here.
                Tensor src = pad_sequence(src_list);
                Tensor trg = pad_sequence(trg_list);

                // Forward pass
                optimizer.zero_grad();
                Tensor logits = m.forward(src,
                        trg.slice(1, new LongOptional(0), new LongOptional(-1), 1));

                // Cross-entropy loss with label smoothing
                // (weight=None, reduction=Mean(1), ignore_index=0,
                // label_smoothing=cfg.label_smoothing)
                Tensor loss = cross_entropy_loss(
                        logits.reshape(-1, logits.size(2)),
                        trg.slice(1, new LongOptional(1), new LongOptional(), 1).reshape(-1),
                        new TensorOptional(), 1L, 0L, cfg.label_smoothing);

                // Backward pass
                loss.backward();

                // Gradient clipping
                clip_grad_norm_(m.parameters(), 1.0);

                // Optimizer step, then scheduler step (same order as Python:
                // optimizer.step(); scheduler.step())
                optimizer.step();
                global_step++;
                lrBeta = oneCycle(global_step, phase1_end, phase2_end, initial_lr, max_lr, min_lr);
                applyLrBeta(optimizer, lrBeta[0], lrBeta[1]);

                total_loss += loss.item_float();
                batch_count++;
                } // PointerScope
            }

            // Epoch logging (mirrors scheduler.get_last_lr()[0])
            float avg_loss = batch_count > 0 ? total_loss / batch_count : 0;
            System.out.printf("Epoch %d/%d | Loss: %.4f | LR: %.7f%n",
                    ep + 1, cfg.epochs, avg_loss, lrBeta[0]);
        }

        // Save model and tokenizers
        OutputArchive archive = new OutputArchive();
        m.save(archive);
        archive.save_to("translator.pth");
        en_tok.save("en_tok.model");
        bn_tok.save("bn_tok.model");
        System.out.println("Saved to 'translator.pth'");
    }

    /**
     * OneCycleLR value for a step: returns {lr, beta1}. Cosine annealing on
     * both phases (PyTorch anneal_strategy='cos'); beta1 cycles 0.95 -> 0.85
     * during warmup, then 0.85 -> 0.95 (cycle_momentum=True defaults).
     */
    private static double[] oneCycle(int stepNum, double phase1End, double phase2End,
            double initialLr, double maxLr, double minLr) {
        double lr, beta1;
        if (stepNum <= phase1End) {
            double pct = stepNum / phase1End;
            lr = annealCos(initialLr, maxLr, pct);
            beta1 = annealCos(0.95, 0.85, pct);
        } else {
            double pct = (stepNum - phase1End) / (phase2End - phase1End);
            lr = annealCos(maxLr, minLr, pct);
            beta1 = annealCos(0.85, 0.95, pct);
        }
        return new double[] { lr, beta1 };
    }

    private static double annealCos(double start, double end, double pct) {
        return end + (start - end) / 2.0 * (Math.cos(Math.PI * pct) + 1.0);
    }

    /** Slot of beta1 inside the native std::tuple<double,double>; detected at runtime. */
    private static int beta1Slot = -1;

    /**
     * Write lr and beta1 into every optimizer param group. The betas tuple is
     * exposed as a two-double native pointer; tuple member order is
     * implementation-defined, so the beta1 slot is detected once by looking
     * for the beta2 default (0.999) in the initial values.
     */
    private static void applyLrBeta(AdamW optimizer, double lr, double beta1) {
        OptimizerParamGroupVector groups = optimizer.param_groups();
        for (long g = 0; g < groups.size(); g++) {
            // options() returns the base OptimizerOptions wrapper; the native
            // object is an AdamWOptions, so reinterpret via the Pointer
            // constructor (JavaCPP downcast idiom).
            AdamWOptions o = new AdamWOptions(groups.get(g).options());
            o.lr().put(lr);
            // betas() exposes the native std::tuple<double,double> as a
            // two-double pointer; tuple member order is implementation-defined,
            // so detect the beta1 slot once by finding the beta2 default 0.999.
            org.bytedeco.javacpp.DoublePointer betas = o.betas().capacity(2);
            if (beta1Slot < 0) {
                beta1Slot = Math.abs(betas.get(0) - 0.999) < 1e-9 ? 1 : 0;
            }
            betas.put(beta1Slot, beta1);
        }
    }

    /**
     * Pad sequences to same length, batch_first=True, pad with 0.
     * Matches Python: nn.utils.rnn.pad_sequence(seqs, batch_first=True)
     */
    private static Tensor pad_sequence(List<Tensor> seqs) {
        // LibTorch provides the same operator as Python's
        // nn.utils.rnn.pad_sequence: batch_first=true, padding_value=0.
        TensorVector vec = new TensorVector(seqs.toArray(new Tensor[0]));
        return org.bytedeco.pytorch.global.torch.pad_sequence(vec, true, 0.0, "right");
    }
}
