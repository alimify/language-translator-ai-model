package translator;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.LongPointer;
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
 * E2b numeric conformance runner (thesis 8.8, 9.6).
 *
 * <p>Loads the framework-neutral weight file shared by all five
 * implementations, runs the fixed probe forward pass, and writes decoder
 * logits as little-endian float32 so the numeric paths can be compared.
 *
 * <p>Deliberately does not read {@code translator.pth} and needs no trained
 * checkpoint of its own: the point of the shared file is that an
 * implementation without trained weights can still participate.
 *
 * <p>The sidecar formats are parsed with the standard library alone; requiring
 * a JSON dependency here would contradict the reason the format exists.
 */
public final class E2b {

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
            e.dims = new long[d.length];
            for (int i = 0; i < d.length; i++) e.dims[i] = Long.parseLong(d[i].trim());
            out.put(f[0], e);
        }
        return out;
    }

    public static int run(String dir, String tag) {
        try {
            Path base = Paths.get(dir);
            Map<String, String> arch = readArch(base.resolve("arch.txt"));
            Map<String, Entry> man = readManifest(base.resolve("manifest.tsv"));
            if (arch.isEmpty() || man.isEmpty()) {
                System.err.println("E2b: cannot read arch.txt / manifest.tsv");
                return 1;
            }
            // JavaCPP binds LibTorch, which stores linear weights [out, in] --
            // the convention the file declares -- so no transpose is required.
            if (!"out_in".equals(arch.get("linear_layout"))) {
                System.err.println("E2b: unexpected linear_layout");
                return 1;
            }

            // Config's fields are compile-time constants in this
            // implementation, so rather than overwrite them the manifest is
            // checked against them. A disagreement is a real finding -- it
            // would mean the port and the reference no longer share a
            // specification -- and is reported rather than silently patched.
            Config cfg = new Config();
            if (cfg.dim != Integer.parseInt(arch.get("dim"))
                    || cfg.heads != Integer.parseInt(arch.get("heads"))
                    || cfg.layers != Integer.parseInt(arch.get("layers"))
                    || cfg.ff_mult != Integer.parseInt(arch.get("ff_mult"))
                    || cfg.max_len != Integer.parseInt(arch.get("max_len"))) {
                System.err.println("E2b: this implementation's Config disagrees "
                        + "with the manifest architecture");
                return 1;
            }
            int sv = Integer.parseInt(arch.get("vocab_src"));
            int tv = Integer.parseInt(arch.get("vocab_trg"));

            Translator m = new Translator(cfg, sv, tv);
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
            // Every tensor in the file must be consumed: a manifest entry with
            // no destination means this port is missing a parameter the
            // reference has, which is a conformance failure in itself.
            for (String name : man.keySet()) {
                if (!named.containsKey(name)) {
                    problems.add("manifest entry " + name + " has no destination");
                }
            }
            if (!problems.isEmpty()) {
                System.err.println("E2b: " + problems.size() + " problem(s):");
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
                System.err.println("E2b: probe.txt did not parse");
                return 1;
            }

            float[] out;
            long[] shp;
            try (PointerScope scope = new PointerScope();
                 NoGradGuard noGrad = new NoGradGuard()) {
                Tensor src = toTensor(srcRows);
                Tensor trg = toTensor(trgRows);
                Tensor logits = m.forward(src, trg).contiguous();
                int n = (int) logits.numel();
                out = new float[n];
                logits.data_ptr_float().get(out, 0, n);
                LongPointer sizes = logits.sizes().data();
                shp = new long[(int) logits.dim()];
                sizes.get(shp, 0, shp.length);
            }

            ByteBuffer ob = ByteBuffer.allocate(out.length * 4)
                    .order(ByteOrder.LITTLE_ENDIAN);
            for (float v : out) ob.putFloat(v);
            Path outPath = base.resolve("logits_" + tag + ".bin");
            Files.write(outPath, ob.array());

            double sum = 0; float mn = Float.MAX_VALUE, mx = -Float.MAX_VALUE;
            for (float v : out) { sum += v; if (v < mn) mn = v; if (v > mx) mx = v; }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < shp.length; i++) {
                if (i > 0) sb.append(",");
                sb.append(shp[i]);
            }
            System.out.printf("%s: shape=[%s] params=%d sum=%.6f min=%.6f max=%.6f%n",
                    tag, sb, loaded, sum, mn, mx);
            System.out.println("  wrote " + outPath);
            return 0;
        } catch (Exception ex) {
            System.err.println("E2b failed: " + ex);
            ex.printStackTrace();
            return 1;
        }
    }

    private static Tensor toTensor(List<long[]> rows) {
        int r = rows.size(), c = rows.get(0).length;
        long[] flat = new long[r * c];
        for (int i = 0; i < r; i++) System.arraycopy(rows.get(i), 0, flat, i * c, c);
        return from_blob(new LongPointer(flat), new long[]{ r, c },
                new TensorOptions(ScalarType.Long)).clone();
    }

    private E2b() {}
}
