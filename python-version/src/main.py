"""CLI entry point for training and chat modes."""
import sys
import torch

from . import DEVICE
from .config import Config
from .tokenizer import SPTokenizer
from .model import Translator
from .data import get_raw_data
from .training import run_training
from .inference import translate_sentence, decoding_policy


def main():
    """Main entry point with train and chat commands."""
    cfg = Config()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    
    if cmd == "train":
        data = get_raw_data(50000)
        run_training(cfg, data)
        
    elif cmd == "chat":
        import os
        if not os.path.exists("translator.pth"):
            print("Model file not found. Run 'train' first.")
            sys.exit()
        
        # Load model with safe globals
        torch.serialization.add_safe_globals([SPTokenizer, Config])
        ck = torch.load("translator.pth", map_location=DEVICE, weights_only=False)
        
        m = Translator(ck["cfg"], ck["en"].vocab_size, ck["bn"].vocab_size).to(DEVICE)
        m.load_state_dict(ck["m"])
        
        print(f"Model Loaded! (Vocab: {ck['en'].vocab_size} words)")
        print("--------------------------------------------------")
        
        while True:
            t = input("\nEN (type 'exit'): ").strip()
            if t.lower() == 'exit':
                break
            if not t:
                continue
            pred = translate_sentence(m, ck["en"], ck["bn"], t)
            print(f"BN: {pred}")
    
    elif cmd == "validate":
        # Equivalence-protocol validation mode (E1 tokenizer IDs, E2 shape
        # checkpoints, E3 beam trace) using the trained checkpoint. The output
        # schema is language-agnostic and machine-diffable.
        import os
        if not os.path.exists("translator.pth"):
            print("Model file not found. Run 'train' first.")
            sys.exit()

        torch.serialization.add_safe_globals([SPTokenizer, Config])
        ck = torch.load("translator.pth", map_location=DEVICE, weights_only=False)
        m = Translator(ck["cfg"], ck["en"].vocab_size, ck["bn"].vocab_size).to(DEVICE)
        m.load_state_dict(ck["m"])
        m.eval()
        en, bn = ck["en"], ck["bn"]

        print("VALIDATE START lang=python")
        print(f"VOCAB en={en.vocab_size} bn={bn.vocab_size}")

        # E1: canonical strings -> encoder input IDs ([SOS] + ids + [EOS])
        e1_cases = [
            ("T1", "Hello, world!"),
            ("T2", "I love programming."),
            ("T3", "Neural machine translation is challenging."),
            ("T4", "Data-driven systems require careful evaluation."),
        ]
        for tid, text in e1_cases:
            ids = [1] + en.encode(text) + [2]
            print(f"E1 {tid} ids={ids}")

        # E2: shape checkpoints for a fixed 2-sentence batch
        import torch.nn as nn
        with torch.no_grad():
            seqs = [torch.tensor([1] + en.encode(t) + [2]) for _, t in e1_cases[:2]]
            src = nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(DEVICE)
            trg = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]], device=DEVICE)
            print(f"E2 enc_input_ids shape={list(src.shape)}")
            emb = m.se(src) * (m.dim ** 0.5)
            print(f"E2 enc_embed shape={list(emb.shape)}")
            sm = (src != 0).unsqueeze(1).unsqueeze(2)
            e = emb
            for layer in m.enc:
                e = layer(e, sm=sm, freqs=m.freqs)
            print(f"E2 enc_out shape={list(e.shape)}")
            print(f"E2 dec_input_ids shape={list(trg.shape)}")
            logits = m(src, trg)
            print(f"E2 dec_logits shape={list(logits.shape)}")

        # E3: beam-search decision trace for a fixed input. The parameter line is
        # read from the decoding path (inference.decoding_policy) rather than
        # restated here; see that function for why.
        _p = decoding_policy()
        print(f"E3 params beam={_p['beam']} lp={_p['lp']} rp={_p['rp']} "
              f"max_steps={_p['max_steps']} temp={_p['temp']}")
        result = translate_sentence(m, en, bn, "Hello.", trace=True)
        print(f"E3 result={result}")
        print("VALIDATE END")

    else:
        print("Usage: python -m src.main [train|chat|validate]")


if __name__ == "__main__":
    main()
