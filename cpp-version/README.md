# Translator AI (C++)

English-to-Bengali Neural Machine Translation using LibTorch with a Transformer architecture.

## Features

- **Transformer Encoder-Decoder** with 6 layers each
- **Rotary Position Embedding (RoPE)** using complex polar form
- **RMSNorm** for efficient layer normalization
- **SiLU activation** in feed-forward networks
- **Weight tying** between target embedding and output projection
- **Beam search** with length penalty and repetition penalty

## Project Structure

```
translator-ai-cpp/
├── CMakeLists.txt
├── README.md
└── src/
    ├── config.hpp
    ├── tokenizer.hpp / .cpp
    ├── data.hpp / .cpp
    ├── training.hpp / .cpp
    ├── inference.hpp / .cpp
    ├── main.cpp
    └── model/
        ├── rope.hpp / .cpp
        ├── normalization.hpp / .cpp
        ├── attention.hpp / .cpp
        ├── block.hpp / .cpp
        └── translator.hpp / .cpp
```

## Build

Requires LibTorch (PyTorch C++ API).

```bash
cd translator-ai-cpp
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```

## Usage

### Training
```bash
./translator-ai-cpp train
```

### Chat Mode
```bash
./translator-ai-cpp chat
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| max_len | 64 | Maximum sequence length |
| dim | 256 | Model dimension |
| heads | 8 | Attention heads |
| layers | 6 | Encoder/Decoder layers |
| ff_mult | 4 | Feed-forward multiplier |
| vocab_size | 12000 | Maximum vocabulary size |
| lr | 8e-4 | Learning rate |
| epochs | 40 | Training epochs |
| batch_size | 32 | Batch size |
| label_smoothing | 0.1 | Label smoothing factor |
| dropout | 0.2 | Dropout rate |
