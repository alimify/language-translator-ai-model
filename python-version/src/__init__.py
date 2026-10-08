"""Translator AI - English to Bengali Neural Machine Translation."""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .config import Config
from .tokenizer import SPTokenizer
from .model import Translator
from .data import get_raw_data
from .training import run_training
from .inference import translate_sentence

__all__ = [
    "Config",
    "SPTokenizer", 
    "Translator",
    "get_raw_data",
    "run_training",
    "translate_sentence",
    "DEVICE",
]
