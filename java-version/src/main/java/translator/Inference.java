package translator;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import translator.model.Translator;

import java.util.*;

/**
 * Beam search inference.
 * Matches Python: inference.py
 */
public class Inference {

    /**
     * Translate an English sentence to Bengali using beam search.
     * Matches Python: def translate_sentence(m, en_tok, bn_tok, sentence,
     * beam_size=3, lp=0.6, rep_penalty=1.4, trace=False)
     */
    public static String translate_sentence(Translator m, SPTokenizer en_tok,
            SPTokenizer bn_tok, String sentence,
            int beam_size, float lp, float rep_penalty) {
        return translate_sentence(m, en_tok, bn_tok, sentence, beam_size, lp, rep_penalty, false);
    }

    public static String translate_sentence(Translator m, SPTokenizer en_tok,
            SPTokenizer bn_tok, String sentence,
            int beam_size, float lp, float rep_penalty, boolean trace) {
        m.eval();

        // Encode source sentence
        List<Long> tokens = en_tok.encode(sentence);
        if (tokens.isEmpty())
            return "";

        // Create source tensor with SOS and EOS
        long[] src_data = new long[tokens.size() + 2];
        src_data[0] = 1; // SOS
        for (int i = 0; i < tokens.size(); i++)
            src_data[i + 1] = tokens.get(i);
        src_data[src_data.length - 1] = 2; // EOS

        Tensor src = tensor(src_data).unsqueeze(0);

        // Initialize beams: (normalized_score, log_sum, sequence)
        List<float[]> beam_scores = new ArrayList<>(); // [norm_score, log_sum]
        List<List<Long>> beam_seqs = new ArrayList<>();
        beam_scores.add(new float[] { 0.0f, 0.0f });
        beam_seqs.add(new ArrayList<>(List.of(1L)));

        for (int step = 0; step < DecodingPolicy.MAX_STEPS; step++) {
            List<float[]> new_scores = new ArrayList<>();
            List<List<Long>> new_seqs = new ArrayList<>();

            for (int b = 0; b < beam_scores.size(); b++) {
                List<Long> seq = beam_seqs.get(b);
                float log_sum = beam_scores.get(b)[1];

                // If sequence ended, keep it as-is
                if (seq.get(seq.size() - 1) == 2) {
                    float ns = (float) (log_sum / Math.pow(seq.size(), lp));
                    new_scores.add(new float[] { ns, log_sum });
                    new_seqs.add(seq);
                    continue;
                }

                // Create target tensor
                long[] trg_data = new long[seq.size()];
                for (int j = 0; j < seq.size(); j++)
                    trg_data[j] = seq.get(j);
                Tensor trg = tensor(trg_data).unsqueeze(0);

                // Forward pass (no grad)
                Tensor logits;
                try (NoGradGuard no_grad = new NoGradGuard()) {
                    logits = m.forward(src, trg);
                }

                // Get last step logits, scaled by the policy temperature
                Tensor curr_logits = logits.select(1, logits.size(1) - 1)
                        .div(new Scalar(DecodingPolicy.TEMPERATURE));

                // Apply repetition penalty
                Map<Long, Integer> token_counts = new HashMap<>();
                for (Long t : seq) {
                    if (t > 2)
                        token_counts.merge(t, 1, Integer::sum);
                }

                for (Map.Entry<Long, Integer> entry : token_counts.entrySet()) {
                    long token = entry.getKey();
                    int count = entry.getValue();
                    float penalty = count == 1 ? rep_penalty : rep_penalty * 1.5f;

                    float val = curr_logits.select(0, 0).select(0, (int) token).item_float();
                    if (val > 0) {
                        curr_logits.select(0, 0).select(0, (int) token).fill_(new Scalar(val / penalty));
                    } else {
                        curr_logits.select(0, 0).select(0, (int) token).fill_(new Scalar((float) (val * penalty)));
                    }
                }

                // Get top-k candidates
                Tensor probs = log_softmax(curr_logits.squeeze(0), -1);
                T_TensorTensor_T top_k = probs.topk(beam_size);
                Tensor top_v = top_k.get0();
                Tensor top_i = top_k.get1();

                for (int i = 0; i < beam_size; i++) {
                    long t_id = top_i.select(0, i).item_long();
                    float new_sum = log_sum + top_v.select(0, i).item_float();
                    List<Long> new_seq = new ArrayList<>(seq);
                    new_seq.add(t_id);
                    float normalized_score = (float) (new_sum / Math.pow(new_seq.size(), lp));
                    new_scores.add(new float[] { normalized_score, new_sum });
                    new_seqs.add(new_seq);
                }
            }

            // Sort by normalized_score descending and keep top beams
            Integer[] indices = new Integer[new_scores.size()];
            for (int j = 0; j < indices.length; j++)
                indices[j] = j;
            Arrays.sort(indices, (a, c) -> Float.compare(new_scores.get(c)[0], new_scores.get(a)[0]));

            beam_scores.clear();
            beam_seqs.clear();
            for (int j = 0; j < Math.min(beam_size, indices.length); j++) {
                beam_scores.add(new_scores.get(indices[j]));
                beam_seqs.add(new_seqs.get(indices[j]));
            }

            // Structured beam trace (equivalence protocol E3)
            if (trace) {
                for (int rank = 0; rank < beam_seqs.size(); rank++) {
                    List<Long> s = beam_seqs.get(rank);
                    System.out.printf("E3 step=%d rank=%d tok=%d score=%.6f%n",
                            step + 1, rank, s.get(s.size() - 1), beam_scores.get(rank)[0]);
                }
            }

            // Early stopping if all beams ended
            boolean all_eos = true;
            for (List<Long> seq : beam_seqs) {
                if (seq.get(seq.size() - 1) != 2)
                    all_eos = false;
            }
            if (all_eos)
                break;
        }

        return bn_tok.decode(beam_seqs.get(0));
    }
}
