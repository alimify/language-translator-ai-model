/**Configuration for the Translator model.*/

class Config {
    /**Hyperparameters for the Translator model.
     * 
     * All values match the original implementation exactly.
     */
    constructor() {
        this.max_len = 64;           // Maximum sequence length
        this.dim = 256;              // Model dimension
        this.heads = 8;              // Number of attention heads
        this.layers = 6;             // Number of encoder/decoder layers
        this.ff_mult = 4;            // Feed-forward multiplier
        this.vocab_size = 12000;     // Maximum vocabulary size
        this.lr = 8e-4;             // Learning rate
        this.epochs = 40;            // Training epochs
        this.batch_size = 32;        // Batch size
        this.label_smoothing = 0.1;  // Label smoothing factor
        this.dropout = 0.2;          // Dropout rate
    }
}

export { Config };
