//! Batch construction for decode steps (design doc §30/§31).

use mini_vllm_core::RequestId;

use crate::sequence::SequenceGroup;

/// The decode batch scheduled for one engine iteration:
/// one token per active sequence, each with its own position.
#[derive(Debug, Clone, Default)]
pub struct DecodeBatch {
    pub request_ids: Vec<RequestId>,
    pub input_token_ids: Vec<u32>,
    pub positions: Vec<u32>,
}

impl DecodeBatch {
    /// Build the batch from the sequences participating in this step.
    pub fn from_sequences(seqs: &[&SequenceGroup]) -> Self {
        Self {
            request_ids: seqs.iter().map(|s| s.request.id.clone()).collect(),
            input_token_ids: seqs.iter().map(|s| s.next_token).collect(),
            positions: seqs.iter().map(|s| s.next_input_position()).collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.request_ids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.request_ids.len()
    }

    /// Every sequence contributes exactly one token in a decode batch.
    pub fn total_tokens(&self) -> usize {
        self.input_token_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::SequenceGroup;
    use mini_vllm_core::{GenerationRequest, SamplingParams};
    use tokio::sync::mpsc;

    fn seq(id: &str, prompt: usize, generated: &[u32]) -> SequenceGroup {
        let (tx, _rx) = mpsc::channel(4);
        let mut s = SequenceGroup::new(
            GenerationRequest {
                id: id.into(),
                prompt_token_ids: vec![1; prompt],
                sampling: SamplingParams::default(),
                max_new_tokens: 16,
                stop_token_ids: vec![],
                stop_strings: vec![],
            },
            mini_vllm_sampling::Sampler::from_seed(0),
            None,
            tx,
            vec![],
            4,
        );
        for &t in generated {
            s.record_sampled(t);
        }
        s
    }

    #[test]
    fn builds_batch_from_sequences() {
        let a = seq("a", 4, &[10, 11]);
        let b = seq("b", 7, &[20]);
        let seqs: Vec<&SequenceGroup> = vec![&a, &b];
        let batch = DecodeBatch::from_sequences(&seqs);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.request_ids, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(batch.input_token_ids, vec![11, 20]);
        // a: pos = 4 + 2 - 1 = 5; b: pos = 7 + 1 - 1 = 7
        assert_eq!(batch.positions, vec![5, 7]);
        assert_eq!(batch.total_tokens(), 2);
    }
}
