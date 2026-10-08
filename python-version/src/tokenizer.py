"""SentencePiece tokenizer with pickling support."""
import os
import io
import sentencepiece as spm
from typing import List

class SPTokenizer:
    """SentencePiece tokenizer wrapper supporting English and Bengali.
    
    Special tokens are handled by SentencePiece:
        - <PAD>: 0
        - <SOS>: 1
        - <EOS>: 2
        - <UNK>: 3
    """
    
    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size
        self.sp = spm.SentencePieceProcessor()
        self.model_content = None # Store model binary for pickling

    def train(self, texts: List[str]) -> None:
        """Train SentencePiece model on texts."""
        # Create temp file for training
        with open("temp_train.txt", "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t + "\n")
        
        # Train model
        model_prefix = "sp_model"
        spm.SentencePieceTrainer.train(
            input="temp_train.txt",
            model_prefix=model_prefix,
            vocab_size=self._vocab_size,
            user_defined_symbols=[],
            model_type="unigram",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            pad_piece="<PAD>",
            bos_piece="<SOS>",
            eos_piece="<EOS>",
            unk_piece="<UNK>",
            character_coverage=1.0,
            hard_vocab_limit=False
        )
        
        # Load model content
        with open(f"{model_prefix}.model", "rb") as f:
            self.model_content = f.read()
            
        # Load into processor
        self.sp.LoadFromSerializedProto(self.model_content)
        
        # Cleanup
        os.remove("temp_train.txt")
        os.remove(f"{model_prefix}.model")
        os.remove(f"{model_prefix}.vocab")

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return self.sp.DecodeIds(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def __getstate__(self):
        """Return state for pickling."""
        return {
            "vocab_size": self._vocab_size,
            "model_content": self.model_content
        }

    def __setstate__(self, state):
        """Restore state from pickle."""
        self._vocab_size = state["vocab_size"]
        self.model_content = state["model_content"]
        self.sp = spm.SentencePieceProcessor()
        if self.model_content:
            self.sp.LoadFromSerializedProto(self.model_content)
