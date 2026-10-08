package translator;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import translator.model.Translator;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.*;
import java.util.*;

/**
 * E2c gradient conformance runner (thesis §8.9, §9.7).
 *
 * <p>E2b checks that this implementation computes the same decoder logits as
 * the reference when both load the same weights, which verifies the forward
 * pass. This verifies the backward one: it loads the same shared weight file,
 * runs one training step on the fixed probe using the same loss call
 * {@code Training} uses, calls {@code backward()}, and writes every parameter
 * gradient as little-endian float32 in sorted-parameter-name order.
 *
 * <p>Why it matters: §9.2 reports that the five implementations differ in
 * held-out quality and that the difference cannot be attributed, because each
 * was trained once from its own initialisation. If the five agree on the
 * gradient at step one they were descending the same surface, and the
 * divergence is down to initialisation and data order alone.
 *
 * <p>Gradients are read before any {@code clip_grad_norm_}, which projects onto
 * a ball of radius 1.0 and would mask a disagreement in magnitude.
 *
 * <p>Deliberately does not read {@code translator.pth}: like E2b, it needs no
 * trained checkpoint of its own.
 */
public final class E2c {

    private E2c() {
    }

    private static final class Entry {
        long offset;
        int count;
        long[] dims;
    }

    private static Map<String, String> readArch(Path p) throws IOException {
        Map<String, String> kv = new LinkedHashMap<>();
        for (String line : Files.readAllLines(p)) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            int eq = line.indexOf('=');
            if (eq > 0) kv.put(line.substring(0, eq), line.substring(eq + 1));
        }
        return kv;
    }

    private static Map<String, Entry> readManifest(Path p) throws IOException {
        Map<String, Entry> out = new LinkedHashMap<>();
        for (String line : Files.readAllLines(p)) {
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] f = line.split("\t");
            if (f.length < 4) continue;
            Entry e = new Entry();
            e.offset = Long.parseLong(f[1].trim());
            e.count = Integer.parseInt(f[2].trim());
            String[] d = f[3].trim().split(",");
            List<Long> dims = new ArrayList<>();
            for (String s : d) {
                if (!s.isEmpty()) dims.add(Long.parseLong(s.trim()));
            }
            e.dims = new long[dims.size()];
            for (int i = 0; i < dims.size(); i++) e.dims[i] = dims.get(i);
            out.put(f[0], e);
        }
        return out;
    }

    private static Tensor toTensor(List<long[]> rows) {
        int R = rows.size(), C = rows.get(0).length;
        long[] flat = new long[R * C];
        for (int r = 0; r < R; r++) {
            System.arraycopy(rows.get(r), 0, flat, r * C, C);
        }
        return from_blob(new org.bytedeco.javacpp.LongPointer(flat),
                new long[]{R, C}, new TensorOptions(ScalarType.Long)).clone();
    }

    /**
     * Run the E2c gradient conformance probe.
     *
     * @param dir directory holding arch.txt, manifest.tsv, probe.txt, model.bin
     * @param tag output tag; writes grads_&lt;tag&gt;.bin and grads_&lt;tag&gt;.txt
     * @return 0 on success
     */
    public static int run(String dir, String tag) {
        try {
            Path base = Paths.get(dir);
            Map<String, String> arch = readArch(base.resolve("arch.txt"));
            Map<String, Entry> man = readManifest(base.resolve("manifest.tsv"));
            if (arch.isEmpty() || man.isEmpty()) {
                System.err.println("E2c: cannot read arch.txt or manifest.tsv");
                return 1;
            }
            if (!"out_in".equals(arch.get("linear_layout"))) {
                System.err.println("E2c: unexpected linear_layout");
                return 1;
            }

            Config cfg = new Config();
            int sv = Integer.parseInt(arch.get("vocab_src"));
            int tv = Integer.parseInt(arch.get("vocab_trg"));

            Translator m = new Translator(cfg, sv, tv);
            // eval() disables dropout without disabling autograd -- eval mode
            // governs dropout and normalisation behaviour only, not gradient
            // recording. The other runners reach the same state by setting
            // cfg.dropout = 0.0; Config.dropout is final here, so this is the
            // route to the identical computation. Determinism is required: a
            // stochastic mask would make the gradient a random variable and the
            // cross-implementation comparison meaningless.
            m.eval();

            byte[] blob = Files.readAllBytes(base.resolve("model.bin"));
            ByteBuffer bb = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);

            int loaded = 0;
            List<String> problems = new ArrayList<>();
            Map<String, Tensor> named = m.namedTensors();

            try (PointerScope scope = new PointerScope();
                 NoGradGuard noGrad = new NoGradGuard()) {
                for (Map.Entry<String, Tensor> kv : named.entrySet()) {
                    Entry e = man.get(kv.getKey());
                    if (e == null) {
                        problems.add("no manifest entry for " + kv.getKey());
                        continue;
                    }
                    float[] vals = new float[e.count];
                    bb.position((int) e.offset);
                    bb.asFloatBuffer().get(vals, 0, vals.length);
                    Tensor t = from_blob(new FloatPointer(vals), e.dims,
                            new TensorOptions(ScalarType.Float)).clone();
                    Tensor target = kv.getValue();
                    if (target.numel() != t.numel()) {
                        problems.add(kv.getKey() + " numel "
                                + target.numel() + " vs " + t.numel());
                        continue;
                    }
                    target.copy_(t);
                    loaded++;
                }
            }
            for (String name : man.keySet()) {
                if (!named.containsKey(name)) {
                    problems.add("manifest entry " + name + " has no destination");
                }
            }
            if (!problems.isEmpty()) {
                System.err.println("E2c: " + problems.size() + " problem(s):");
                for (int i = 0; i < Math.min(12, problems.size()); i++) {
                    System.err.println("   " + problems.get(i));
                }
                return 1;
            }

            // ---- probe ----
            List<long[]> srcRows = new ArrayList<>(), trgRows = new ArrayList<>();
            int mode = 0, remaining = 0;
            for (String line : Files.readAllLines(base.resolve("probe.txt"))) {
                line = line.trim();
                if (line.startsWith("src_rows=")) {
                    remaining = Integer.parseInt(line.substring(9)); mode = 0;
                } else if (line.startsWith("src_cols=")) {
                    mode = 1;
                } else if (line.startsWith("trg_rows=")) {
                    remaining = Integer.parseInt(line.substring(9)); mode = 0;
                } else if (line.startsWith("trg_cols=")) {
                    mode = 2;
                } else if (mode != 0 && remaining > 0 && !line.isEmpty()) {
                    String[] c = line.split(",");
                    long[] row = new long[c.length];
                    for (int i = 0; i < c.length; i++) row[i] = Long.parseLong(c[i].trim());
                    if (mode == 1) srcRows.add(row); else trgRows.add(row);
                    remaining--;
                }
            }
            if (srcRows.isEmpty() || trgRows.isEmpty()) {
                System.err.println("E2c: probe.txt did not parse");
                return 1;
            }

            // Zero any stale gradient before the step. set_to_none=false keeps
            // the gradient tensors allocated and zeroed, matching the
            // reference runner's zero_grad(set_to_none=False); with true, an
            // untouched parameter would report an undefined gradient and be
            // silently skipped below instead of contributing zeros.
            m.zero_grad(false);

            Tensor src = toTensor(srcRows);
            Tensor trg = toTensor(trgRows);
            long cols = trg.size(1);
            // Teacher forcing exactly as Training.java does it.
            Tensor trgIn = trg.slice(1, new LongOptional(0),
                    new LongOptional(cols - 1), 1);
            Tensor trgOut = trg.slice(1, new LongOptional(1),
                    new LongOptional(), 1);

            Tensor logits = m.forward(src, trgIn);
            Tensor loss = cross_entropy_loss(
                    logits.reshape(-1, logits.size(2)),
                    trgOut.reshape(-1),
                    new TensorOptional(), 1L, 0L, cfg.label_smoothing);
            loss.backward();
            double lossVal = loss.item_double();

            // Emit in sorted parameter-name order, matching the reference.
            List<String> names = new ArrayList<>(named.keySet());
            Collections.sort(names);

            double totalSq = 0.0;
            int withGrad = 0;
            long values = 0;
            StringBuilder orderCsv = new StringBuilder();
            List<float[]> chunks = new ArrayList<>();
            for (String name : names) {
                Tensor p = named.get(name);
                Tensor g = p.grad();
                if (g == null || !g.defined()) continue;
                Tensor gc = g.contiguous();
                int n = (int) gc.numel();
                float[] vals = new float[n];
                gc.data_ptr_float().get(vals, 0, n);
                for (float v : vals) totalSq += (double) v * (double) v;
                chunks.add(vals);
                values += n;
                if (orderCsv.length() > 0) orderCsv.append(",");
                orderCsv.append(name);
                withGrad++;
            }
            double gradNorm = Math.sqrt(totalSq);

            ByteBuffer ob = ByteBuffer.allocate((int) values * 4)
                    .order(ByteOrder.LITTLE_ENDIAN);
            for (float[] c : chunks) {
                for (float v : c) ob.putFloat(v);
            }
            Path outPath = base.resolve("grads_" + tag + ".bin");
            Files.write(outPath, ob.array());

            StringBuilder sc = new StringBuilder();
            sc.append("tag=").append(tag).append("\n");
            sc.append(String.format("loss=%.10f%n", lossVal));
            sc.append(String.format("grad_norm=%.10f%n", gradNorm));
            sc.append("grad_values=").append(values).append("\n");
            sc.append("tensors_with_grad=").append(withGrad).append("\n");
            sc.append("params_loaded=").append(loaded).append("\n");
            sc.append("order=").append(orderCsv).append("\n");
            Files.write(base.resolve("grads_" + tag + ".txt"),
                    sc.toString().getBytes("UTF-8"));

            System.out.printf("%s: loss=%.10f grad_norm=%.10f values=%d tensors=%d%n",
                    tag, lossVal, gradNorm, values, withGrad);
            System.out.println("  wrote " + outPath);
            return 0;
        } catch (Exception ex) {
            System.err.println("E2c failed: " + ex);
            ex.printStackTrace();
            return 1;
        }
    }
}
