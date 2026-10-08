# Translator AI (Java)

English-to-Bengali Neural Machine Translation using DJL (Deep Java Library) with PyTorch backend.

## Features

- **Transformer Encoder-Decoder** with 6 layers each
- **Rotary Position Embedding (RoPE)** using sin/cos rotation
- **RMSNorm** for efficient layer normalization
- **SiLU activation** in feed-forward networks
- **Weight tying** between target embedding and output projection
- **Beam search** with length penalty and repetition penalty

## Project Structure

```
translator-ai-java/
├── pom.xml
├── README.md
└── src/main/java/translator/
    ├── Main.java           # CLI entry point
    ├── Config.java         # Hyperparameters
    ├── WordTokenizer.java  # Tokenization
    ├── DataLoader.java     # Dataset loading
    ├── Trainer.java        # Training loop
    ├── Inference.java      # Beam search translation
    └── model/
        ├── RoPE.java           # Rotary Position Embedding
        ├── RMSNorm.java        # RMS Normalization
        ├── MultiHeadAttention.java
        ├── TransformerBlock.java
        └── Translator.java     # Main model
```

## Build

```bash
cd translator-ai-java
mvn clean package
```

## Usage

### Training
```bash
java -jar target/translator-ai-java-1.0.0.jar train
```

### Chat Mode
```bash
java -jar target/translator-ai-java-1.0.0.jar chat
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| maxLen | 64 | Maximum sequence length |
| dim | 256 | Model dimension |
| heads | 8 | Attention heads |
| layers | 6 | Encoder/Decoder layers |
| ffMult | 4 | Feed-forward multiplier |
| vocabSize | 12000 | Maximum vocabulary size |
| lr | 8e-4 | Learning rate |
| epochs | 40 | Training epochs |
| batchSize | 32 | Batch size |
| labelSmoothing | 0.1 | Label smoothing factor |
| dropout | 0.2 | Dropout rate |
