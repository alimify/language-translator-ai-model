package translator;

/**
 * Configuration for the Translator model.
 * Matches Python: @dataclass Config
 */
public class Config {
    public final int max_len = 64;
    public final int dim = 256;
    public final int heads = 8;
    public final int layers = 6;
    public final int ff_mult = 4;
    public final int vocab_size = 12000;
    public final float lr = 8e-4f;
    public final int epochs = 40;
    public final int batch_size = 32;
    public final float label_smoothing = 0.1f;
    public final float dropout = 0.2f;
}
