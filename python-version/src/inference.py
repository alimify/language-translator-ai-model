"""Beam search inference with length and repetition penalties."""
import inspect

import torch
import torch.nn.functional as F
from typing import List

from . import DEVICE
from .tokenizer import SPTokenizer


def translate_sentence(
    m,
    en_tok: SPTokenizer,
    bn_tok: SPTokenizer,
    sentence: str,
    beam_size: int = 3,
    lp: float = 0.6,
    rep_penalty: float = 1.4,
    max_steps: int = 30,
    temperature: float = 0.8,
    trace: bool = False,
) -> str:
    """Translate an English sentence to Bengali using beam search.
    
    Args:
        m: Trained Translator model
        en_tok: English tokenizer
        bn_tok: Bengali tokenizer
        sentence: Input English sentence
        beam_size: Number of beams to maintain
        lp: Length penalty exponent (default 0.6)
        rep_penalty: Repetition penalty factor (default 1.4)
            - First occurrence: divided/multiplied by 1.4
            - Subsequent occurrences: divided/multiplied by 1.4 * 1.5
            
    Returns:
        Translated Bengali sentence
    """
    m.eval()
    
    # Encode source sentence
    tokens = en_tok.encode(sentence)
    if not tokens:
        return ""
    
    src = torch.tensor([[1] + tokens + [2]], device=DEVICE)
    
    # Initialize beams: (normalized_score, log_sum, sequence)
    beams = [(0.0, 0.0, [1])]
    
    for step in range(max_steps):  # Maximum generation steps
        new_beams = []
        
        for _, log_sum, seq in beams:
            # If sequence ended, keep it as-is
            if seq[-1] == 2:
                new_beams.append((log_sum / (len(seq) ** lp), log_sum, seq))
                continue
            
            with torch.no_grad():
                logits = m(src, torch.tensor([seq], device=DEVICE))
                curr_logits = logits[0, -1, :] / temperature  # Temperature scaling
                
                # Apply repetition penalty
                for token in set(seq):
                    if token > 2:  # Skip special tokens
                        count = seq.count(token)
                        # Stronger penalty for repeated tokens
                        penalty = rep_penalty if count == 1 else rep_penalty * 1.5
                        
                        if curr_logits[token] > 0:
                            curr_logits[token] /= penalty
                        else:
                            curr_logits[token] *= penalty
                
                # Get top-k candidates
                probs = F.log_softmax(curr_logits, dim=-1)
                top_v, top_i = probs.topk(beam_size)
                
                for i in range(beam_size):
                    t_id = top_i[i].item()
                    new_sum = log_sum + top_v[i].item()
                    new_seq = seq + [t_id]
                    normalized_score = new_sum / (len(new_seq) ** lp)
                    new_beams.append((normalized_score, new_sum, new_seq))
        
        # Keep top beams
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

        # Structured beam trace (equivalence protocol E3)
        if trace:
            for rank, (ns, _, s) in enumerate(beams):
                print(f"E3 step={step + 1} rank={rank} tok={s[-1]} score={ns:.6f}")

        # Early stopping if all beams ended
        if all(s[-1] == 2 for _, _, s in beams):
            break
    
    return bn_tok.decode(beams[0][2])


def decoding_policy() -> dict:
    """The decoding policy actually in force, read from the decoding path.

    Derived from `translate_sentence`'s own signature rather than restated, so a
    change to any parameter is necessarily reflected in what the E3 layer
    reports. This matters: an earlier version of the protocol printed these five
    values as a hardcoded string literal, which meant the E3 parameter check
    compared a constant against itself and could not fail. Three injected
    decoding-policy defects went undetected as a direct result, and the check
    passed on every run of the study while verifying nothing (thesis §9.5).

    A conformance check must obtain its values from the path under test.
    """
    sig = inspect.signature(translate_sentence)
    return {
        "beam": sig.parameters["beam_size"].default,
        "lp": sig.parameters["lp"].default,
        "rp": sig.parameters["rep_penalty"].default,
        "max_steps": sig.parameters["max_steps"].default,
        "temp": sig.parameters["temperature"].default,
    }
