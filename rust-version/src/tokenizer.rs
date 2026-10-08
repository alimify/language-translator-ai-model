//! SentencePiece tokenizer wrapper.
//!
//! Matches Python: tokenizer.py

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use sentencepiece::SentencePieceProcessor;
use anyhow::{ Result, Context };

/// SentencePiece tokenizer wrapper.
///
/// Special tokens (handled by SP):
/// - `<PAD>`: 0
/// - `<SOS>`: 1
/// - `<EOS>`: 2
/// - `<UNK>`: 3
pub struct SPTokenizer {
    processor: Option<SentencePieceProcessor>,
    _vocab_size: i64,
    model_content: Option<Vec<u8>>,
}

impl SPTokenizer {
    /// Create new tokenizer.
    /// Matches Python: def __init__(self, vocab_size)
    pub fn new(vocab_size: i64) -> Self {
        Self {
            processor: None,
            _vocab_size: vocab_size,
            model_content: None,
        }
    }

    /// Train SentencePiece model on texts.
    /// Matches Python: def train(self, texts)
    ///
    /// The `sentencepiece` crate exposes processing only (no in-process
    /// trainer), so training shells out to the `spm_train` CLI with the exact
    /// flags Python passes to SentencePieceTrainer.train. In the thesis
    /// pipeline the shared frozen artifacts (en_tok.model / bn_tok.model) are
    /// normally loaded instead, so this path is a fallback.
    pub fn train(&mut self, texts: &[String]) -> Result<()> {
        // Write temp training file
        {
            let mut file = fs::File::create("temp_train.txt")?;
            for text in texts {
                writeln!(file, "{}", text)?;
            }
        }

        let status = Command::new("spm_train")
            .args([
                "--input=temp_train.txt",
                "--model_prefix=sp_model",
                &format!("--vocab_size={}", self._vocab_size),
                "--model_type=unigram",
                "--pad_id=0",
                "--bos_id=1",
                "--eos_id=2",
                "--unk_id=3",
                "--pad_piece=<PAD>",
                "--bos_piece=<SOS>",
                "--eos_piece=<EOS>",
                "--unk_piece=<UNK>",
                "--character_coverage=1.0",
                "--hard_vocab_limit=false",
            ])
            .status()
            .context("Failed to run spm_train (is it on PATH?)")?;
        if !status.success() {
            anyhow::bail!("spm_train failed with status {}", status);
        }

        // Store model content for serialization and load it
        let content = fs::read("sp_model.model")?;
        self.processor = Some(
            SentencePieceProcessor::from_serialized_proto(&content).context(
                "Failed to load trained SentencePiece model"
            )?
        );
        self.model_content = Some(content);

        // Cleanup
        let _ = fs::remove_file("temp_train.txt");
        let _ = fs::remove_file("sp_model.model");
        let _ = fs::remove_file("sp_model.vocab");

        Ok(())
    }

    /// Save tokenizer model to file.
    /// Matches Python: __getstate__ / save
    pub fn save(&self, path: &str) -> Result<()> {
        if let Some(content) = &self.model_content {
            fs::write(path, content)?;
        }
        Ok(())
    }

    /// Load model from file.
    /// Matches Python: __setstate__ / load
    pub fn load(&mut self, path: &str) -> Result<()> {
        if !Path::new(path).exists() {
            anyhow::bail!("Model file not found: {}", path);
        }
        let content = fs::read(path)?;
        if content.is_empty() {
            anyhow::bail!("Model file is empty: {}", path);
        }
        self.processor = Some(SentencePieceProcessor::from_serialized_proto(&content)?);
        self.model_content = Some(content);
        Ok(())
    }

    /// Convert text to token IDs.
    /// Matches Python: def encode(self, text) -> list[int]
    pub fn encode(&self, text: &str) -> Vec<i64> {
        match &self.processor {
            Some(sp) =>
                sp
                    .encode(text)
                    .map(|pieces|
                        pieces
                            .into_iter()
                            .map(|p| p.id as i64)
                            .collect()
                    )
                    .unwrap_or_default(),
            None => vec![],
        }
    }

    /// Convert token IDs back to text.
    /// Matches Python: def decode(self, ids) -> str
    pub fn decode(&self, ids: &[i64]) -> String {
        match &self.processor {
            Some(sp) => {
                let ids_u32: Vec<u32> = ids
                    .iter()
                    .map(|&id| id as u32)
                    .collect();
                sp.decode_piece_ids(&ids_u32).unwrap_or_default()
            }
            None => String::new(),
        }
    }

    /// Get current vocabulary size (actual trained piece count).
    /// Matches Python: @property vocab_size -> sp.GetPieceSize()
    pub fn vocab_size(&self) -> i64 {
        match &self.processor {
            Some(sp) => sp.len() as i64,
            None => self._vocab_size,
        }
    }
}
