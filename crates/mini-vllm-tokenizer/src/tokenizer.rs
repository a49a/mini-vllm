//! Hugging Face `tokenizer.json` wrapper plus streaming-safe incremental
//! detokenization.

use std::path::Path;
use std::sync::Arc;

use tokenizers::Tokenizer;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer from {path}: {source}")]
    Load {
        path: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("tokenizer encode/decode failed: {0}")]
    Operation(String),
}

pub type Result<T> = std::result::Result<T, TokenizerError>;

/// Thread-safe wrapper around `tokenizers::Tokenizer`.
#[derive(Debug)]
pub struct TokenizerWrapper {
    inner: Tokenizer,
}

impl TokenizerWrapper {
    /// Load a Hugging Face `tokenizer.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = Tokenizer::from_file(path).map_err(|e| TokenizerError::Load {
            path: path.display().to_string(),
            source: e,
        })?;
        Ok(Self { inner })
    }

    /// Load `tokenizer.json` from a model directory.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_file(dir.as_ref().join("tokenizer.json"))
    }

    /// Wrap an already-constructed tokenizer (test fixture support).
    pub fn from_inner(inner: Tokenizer) -> Self {
        Self { inner }
    }

    /// Encode text into token ids.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::Operation(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token ids into text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        if let Some(id) = ids.iter().find(|&&id| self.inner.id_to_token(id).is_none()) {
            return Err(TokenizerError::Operation(format!(
                "token id {id} absent from tokenizer vocabulary"
            )));
        }
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Operation(e.to_string()))
    }

    pub fn vocab_size(&self, with_added_tokens: bool) -> usize {
        self.inner.get_vocab_size(with_added_tokens)
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.inner
            .token_to_id("<|im_end|>")
            .or_else(|| self.inner.token_to_id("<|endoftext|>"))
            .or_else(|| self.inner.token_to_id("</s>"))
    }
}

/// Streams tokens out as text incrementally, with integrated stop-string
/// truncation.
///
/// Uses the same sliding-window algorithm as HF's `DecodeStream`: only a
/// small window of ids (prefix + pending tail) is kept around, so each
/// [`push`](Self::push) is O(window), not O(sequence) — decoding stays cheap
/// regardless of output length. The full text is still accumulated for
/// stop-string matching, but the search itself is restricted to the region
/// where a *new* match can appear (previous pushes already verified the
/// prefix).
///
/// When a stop string appears, the retained text is truncated at its first
/// occurrence and later pushes emit nothing. Potential stop prefixes are
/// withheld until disambiguated. Call `finish` on normal EOS/length completion
/// to release any unmatched suffix.
#[derive(Debug)]
pub struct IncrementalDetokenizer {
    tokenizer: Arc<TokenizerWrapper>,
    /// Sliding window of ids needed to produce the next chunk.
    window_ids: Vec<u32>,
    /// Decoded text of `window_ids` — trimmed off the next chunk.
    prefix: String,
    /// Length of `prefix`'s prefix inside `window_ids` (drain cursor).
    prefix_index: usize,
    /// Full generated text, kept for stop-string matching and `text()`.
    full: String,
    /// Byte offset through which text has been emitted.
    emitted: usize,
    /// Set when a stop string was found; holds the truncated final text.
    truncated_text: Option<String>,
    /// Total tokens pushed (the id window is trimmed, so no length source).
    tokens_seen: usize,
}

impl IncrementalDetokenizer {
    pub fn new(tokenizer: Arc<TokenizerWrapper>) -> Self {
        Self {
            tokenizer,
            window_ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            full: String::new(),
            emitted: 0,
            truncated_text: None,
            tokens_seen: 0,
        }
    }

    /// Append a token.
    ///
    /// Returns the incremental text delta (possibly empty) and whether a
    /// stop string now occurs in the decoded text (the delta is already
    /// truncated so it never includes characters past the stop).
    pub fn push(&mut self, token_id: u32, stop_strings: &[String]) -> Result<(String, bool)> {
        self.tokens_seen += 1;
        if self.truncated_text.is_some() {
            return Ok((String::new(), true));
        }

        // --- Incremental decode over the small window (HF DecodeStream). ---
        self.window_ids.push(token_id);
        let decoded = self.tokenizer.decode(&self.window_ids, true)?;
        let mut delta = String::new();
        if decoded.len() > self.prefix.len()
            && !decoded.ends_with('\u{FFFD}')
            && decoded.starts_with(self.prefix.as_str())
        {
            delta = decoded[self.prefix.len()..].to_string();
            let new_prefix_index = self.window_ids.len() - self.prefix_index;
            self.window_ids = self.window_ids.drain(self.prefix_index..).collect();
            self.prefix = self.tokenizer.decode(&self.window_ids, true)?;
            self.prefix_index = new_prefix_index;
        }

        // --- Stop-string matching over the region a new match can span. ---
        let before = self.full.len();
        self.full.push_str(&delta);
        if !stop_strings.is_empty() {
            let max_stop_len = stop_strings.iter().map(|s| s.len()).max().unwrap_or(0);
            // A new match must extend past `before`, so it starts within
            // `max_stop_len` bytes of the boundary.
            let mut from = before.saturating_sub(max_stop_len.saturating_sub(1));
            while from > 0 && !self.full.is_char_boundary(from) {
                from -= 1;
            }
            let mut stop_at: Option<usize> = None;
            for stop in stop_strings {
                if stop.is_empty() {
                    continue;
                }
                if let Some(pos) = self.full[from..].find(stop.as_str()) {
                    let abs = from + pos;
                    stop_at = Some(match stop_at {
                        Some(existing) => existing.min(abs),
                        None => abs,
                    });
                }
            }
            if let Some(pos) = stop_at {
                self.truncated_text = Some(self.full[..pos].to_string());
                let delta = self.full[self.emitted..pos].to_string();
                self.emitted = pos;
                return Ok((delta, true));
            }
        }

        // Retain the longest suffix that could complete a stop next time.
        let mut end = self.full.len();
        for stop in stop_strings.iter().filter(|s| !s.is_empty()) {
            let from = self.full.len().saturating_sub(stop.len());
            for (offset, _) in self.full[self.emitted..].char_indices() {
                let start = self.emitted + offset;
                if start >= from && stop.starts_with(&self.full[start..]) {
                    end = end.min(start);
                    break;
                }
            }
        }
        let delta = self.full[self.emitted..end].to_string();
        self.emitted = end;
        Ok((delta, false))
    }

    /// Release a withheld, unmatched stop prefix on normal EOS/length finish.
    pub fn finish(&mut self) -> String {
        if self.truncated_text.is_some() {
            return String::new();
        }
        let tail = self.full[self.emitted..].to_string();
        self.emitted = self.full.len();
        tail
    }

    /// Final text after truncation (if a stop string matched).
    pub fn final_text(&self) -> Option<&str> {
        self.truncated_text.as_deref()
    }

    /// Full text generated so far (truncated when a stop matched).
    pub fn text(&self) -> Result<String> {
        Ok(match &self.truncated_text {
            Some(t) => t.clone(),
            None => self.full.clone(),
        })
    }

    pub fn tokens_seen(&self) -> usize {
        self.tokens_seen
    }
}

#[cfg(feature = "test-util")]
pub mod testutil {
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer;
    use tokenizers::processors::byte_level::ByteLevel as ByteLevelPostProcessor;
    use tokenizers::{AddedToken, Tokenizer};

    /// A tiny deterministic tokenizer: byte-level pre-tokenization over a
    /// hand-built vocabulary `a b c d` (plus space-prefixed forms) and the
    /// Qwen chat special tokens. Good enough to exercise encode/decode,
    /// incremental detokenization and chat rendering in tests.
    pub fn char_tokenizer() -> Tokenizer {
        let entries: Vec<(&str, u32)> = vec![
            ("<unk>", 0),
            ("<|im_start|>", 1),
            ("<|im_end|>", 2),
            ("<|endoftext|>", 3),
            ("a", 4),
            ("b", 5),
            ("c", 6),
            ("d", 7),
            ("Ġa", 8),
            ("Ġb", 9),
            ("Ġc", 10),
            ("Ġd", 11),
        ];
        let vocab = entries.iter().map(|(s, i)| (s.to_string(), *i)).collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        let _ = tok.with_pre_tokenizer(Some(ByteLevelPreTokenizer::new(false, true, true))); // Qwen-style: no prefix space
        let _ = tok.with_decoder(Some(ByteLevelDecoder::default()));
        let _ = tok.with_post_processor(Some(ByteLevelPostProcessor::default()));
        let specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
            .iter()
            .map(|s| AddedToken::from(*s, true))
            .collect::<Vec<_>>();
        let _ = tok.add_special_tokens(&specials);
        tok
    }

    pub fn wrapper() -> super::TokenizerWrapper {
        super::TokenizerWrapper::from_inner(char_tokenizer())
    }
}

#[cfg(all(test, feature = "test-util"))]
impl IncrementalDetokenizer {
    /// Test-only visibility into the sliding-window size.
    pub fn window_len_for_test(&self) -> usize {
        self.window_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_files_error_cleanly() {
        assert!(TokenizerWrapper::from_file("/nonexistent/tokenizer.json").is_err());
    }

    #[cfg(feature = "test-util")]
    mod incremental {
        use super::super::testutil;
        use super::*;
        use std::sync::Arc;

        fn detok() -> IncrementalDetokenizer {
            IncrementalDetokenizer::new(Arc::new(testutil::wrapper()))
        }

        fn ids(text: &str) -> Vec<u32> {
            testutil::wrapper().encode(text, false).unwrap()
        }

        #[test]
        fn concatenates_deltas_into_full_text() {
            let tok = testutil::wrapper();
            let all = ids("a b c d");
            let mut d = detok();
            let mut acc = String::new();
            for id in all {
                let (delta, stopped) = d.push(id, &[]).unwrap();
                assert!(!stopped);
                acc.push_str(&delta);
            }
            assert_eq!(acc, "a b c d");
            assert_eq!(d.text().unwrap(), "a b c d");
            assert_eq!(d.tokens_seen(), tok.encode("a b c d", false).unwrap().len());
        }

        #[test]
        fn stop_prefix_is_withheld_across_tokens() {
            for stops in [vec!["b c".into()], vec!["b c d".into(), "b c".into()]] {
                let mut d = detok();
                let mut out = String::new();
                for id in ids("a b c d") {
                    let (delta, stop) = d.push(id, &stops).unwrap();
                    out.push_str(&delta);
                    if stop {
                        break;
                    }
                }
                assert_eq!(out, "a ");
                assert_eq!(d.final_text(), Some(out.as_str()));
                assert_eq!(d.finish(), "");
            }
        }

        #[test]
        fn unmatched_stop_prefix_is_released_on_finish_or_mismatch() {
            for input in ["a b", "a b d"] {
                let mut d = detok();
                let mut out = String::new();
                for id in ids(input) {
                    let (delta, stop) = d.push(id, &["b c".into()]).unwrap();
                    assert!(!stop);
                    out.push_str(&delta);
                }
                out.push_str(&d.finish());
                assert_eq!(out, input);
                assert_eq!(d.finish(), "");
            }
        }

        #[test]
        fn multibyte_stop_prefixes_preserve_utf8_boundaries() {
            let vocab = [
                ("你".to_string(), 0),
                ("好".to_string(), 1),
                ("呀".to_string(), 2),
            ]
            .into_iter()
            .collect();
            let model = tokenizers::models::wordlevel::WordLevel::builder()
                .vocab(vocab)
                .build()
                .unwrap();
            let mut tok = Tokenizer::new(model);
            tok.with_decoder(Some(tokenizers::decoders::fuse::Fuse::new()));
            let tokenizer = Arc::new(TokenizerWrapper::from_inner(tok));
            let mut d = IncrementalDetokenizer::new(tokenizer.clone());
            assert_eq!(d.push(0, &["你好".into()]).unwrap(), (String::new(), false));
            assert_eq!(d.push(1, &["你好".into()]).unwrap(), (String::new(), true));
            assert_eq!(d.final_text(), Some(""));
            let mut d = IncrementalDetokenizer::new(tokenizer);
            assert_eq!(d.push(0, &["你好".into()]).unwrap().0, "");
            assert_eq!(d.push(2, &["你好".into()]).unwrap().0, "你呀");
        }

        #[test]
        fn window_stays_small() {
            // Long generation: the id window must not grow with output length.
            let mut d = detok();
            for _ in 0..200 {
                for &id in &ids("a b c d") {
                    d.push(id, &[]).unwrap();
                }
            }
            assert!(
                d.window_len_for_test() <= 8,
                "window grew: {}",
                d.window_len_for_test()
            );
        }

        #[test]
        fn stop_string_truncates_delta_and_final_text() {
            let mut d = detok();
            let mut out = String::new();
            let mut stopped = false;
            for id in ids("a b c d") {
                let (delta, s) = d.push(id, &["c".to_string()]).unwrap();
                out.push_str(&delta);
                stopped |= s;
                if s {
                    break;
                }
            }
            assert!(stopped);
            assert_eq!(out, "a b ");
            assert_eq!(d.final_text(), Some("a b "));
            // Later pushes emit nothing.
            let (delta, s) = d.push(4, &["c".to_string()]).unwrap();
            assert_eq!(delta, "");
            assert!(s);
        }
    }
}
