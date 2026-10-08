package translator;

/**
 * The decoding policy actually in force.
 *
 * <p>One definition, used by the decoder and read by the E3 conformance check.
 * Previously that check printed these five values as a string literal while the
 * decoder took its own arguments, so the two could disagree without anything
 * failing -- the check could not detect a divergence in the policy it claimed
 * to verify (thesis 9.5). Changing a value here now changes both what runs and
 * what is reported.
 */
public final class DecodingPolicy {
    public static final int BEAM = 3;
    public static final float LP = 0.6f;
    public static final float REP_PENALTY = 1.4f;
    public static final int MAX_STEPS = 30;
    public static final float TEMPERATURE = 0.8f;

    /** The E3 parameter line, rendered from the values the decoder uses. */
    public static String e3Line() {
        return "E3 params beam=" + BEAM + " lp=" + LP + " rp=" + REP_PENALTY
                + " max_steps=" + MAX_STEPS + " temp=" + TEMPERATURE;
    }

    private DecodingPolicy() {}
}
