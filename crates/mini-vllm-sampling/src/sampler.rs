//! Deterministic, seeded sampling pipeline.

use std::collections::HashSet;

use mini_vllm_core::SamplingParams;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A per-request sampler holding the seeded RNG.
#[derive(Debug)]
pub struct Sampler {
    rng: StdRng,
}

impl Sampler {
    /// Build a sampler from an explicit seed (request seed, or engine seed
    /// mixed with a request counter by the engine).
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Run the full pipeline over one row of logits and return a token id.
    ///
    /// `context` is the set of tokens already seen (prompt + generated) for
    /// the repetition penalty; the engine maintains it incrementally so this
    /// stays O(|context|) instead of rebuilding and sorting per token.
    pub fn sample(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
        context: &HashSet<u32>,
    ) -> u32 {
        let mut logits = logits.to_vec();
        if let Some(penalty) = params.repetition_penalty {
            apply_repetition_penalty(&mut logits, context, penalty);
        }
        if params.temperature <= f32::EPSILON {
            return argmax(&logits);
        }
        for l in &mut logits {
            *l /= params.temperature;
        }
        let candidates = top_k_filter(&logits, params.top_k);
        let probs = softmax(&candidates);
        let probs = top_p_filter(&probs, params.top_p);

        sample_from_distribution(&candidates, &probs, &mut self.rng)
    }
}

/// Greedy argmax; ties resolve to the lowest token id (deterministic).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Repetition penalty (as in CTRL / HF `repetition_penalty`):
/// for every token present in `context`, scale the logit down if positive
/// and up if negative.
pub fn apply_repetition_penalty(logits: &mut [f32], context: &HashSet<u32>, penalty: f32) {
    if penalty == 1.0 || context.is_empty() {
        return;
    }
    for &id in context {
        let idx = id as usize;
        if idx < logits.len() {
            logits[idx] = if logits[idx] > 0.0 {
                logits[idx] / penalty
            } else {
                logits[idx] * penalty
            };
        }
    }
}

/// Keep *exactly* the `k` largest logits (zeroing the rest). `None` or
/// `k == 0` disables. Ties at the cut are broken deterministically by the
/// lower token id, so the support never silently grows beyond `k`.
pub fn top_k_filter(logits: &[f32], top_k: Option<usize>) -> Vec<f32> {
    let k = match top_k {
        Some(k) if k > 0 => k.min(logits.len()),
        _ => return logits.to_vec(),
    };
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let mut keep = vec![false; logits.len()];
    for &i in order.iter().take(k) {
        keep[i] = true;
    }
    logits
        .iter()
        .zip(keep)
        .map(|(&l, keep)| if keep { l } else { f32::NEG_INFINITY })
        .collect()
}

/// Numerically stable softmax over a vector (`-inf` entries get probability 0).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        // All filtered out — should not happen, but stay defined.
        let mut v = vec![f32::NEG_INFINITY; logits.len()];
        if !v.is_empty() {
            v[0] = 0.0;
        }
        return v;
    }
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Nucleus (top-p) filtering: zero out probabilities after the cumulative
/// distribution reaches `top_p` (the minimal prefix of tokens sorted by
/// descending probability always survives). `None` or `p >= 1` disables.
pub fn top_p_filter(probs: &[f32], top_p: Option<f32>) -> Vec<f32> {
    let p = match top_p {
        Some(p) if p < 1.0 => p,
        _ => return probs.to_vec(),
    };
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let mut out = vec![0.0f32; probs.len()];
    let mut cum = 0.0f32;
    for &i in order.iter() {
        if probs[i] <= 0.0 {
            break;
        }
        out[i] = probs[i];
        cum += probs[i];
        // Always keep at least one token; stop once we cross the threshold.
        if cum >= p {
            break;
        }
    }
    out
}

/// Draw one index from a finite discrete distribution.
fn sample_from_distribution(logits: &[f32], probs: &[f32], rng: &mut StdRng) -> u32 {
    let total: f32 = probs.iter().sum();
    debug_assert!(total > 0.0 && total.is_finite(), "invalid probability mass");
    let pick: f64 = rng.random::<f64>();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p / total;
        if (pick as f32) < acc {
            return i as u32;
        }
    }
    // Floating point residue: fall back to the most probable candidate.
    argmax(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn params(temp: f32, top_k: Option<usize>, top_p: Option<f32>) -> SamplingParams {
        SamplingParams {
            temperature: temp,
            top_k,
            top_p,
            repetition_penalty: None,
            seed: Some(42),
        }
    }

    fn set(ids: &[u32]) -> HashSet<u32> {
        ids.iter().copied().collect()
    }

    #[test]
    fn greedy_argmax_is_deterministic() {
        let logits = [0.1, 3.0, 2.9, -1.0];
        let mut s = Sampler::from_seed(0);
        for _ in 0..10 {
            assert_eq!(
                s.sample(&logits, &params(0.0, None, None), &HashSet::new()),
                1
            );
        }
    }

    #[test]
    fn temperature_zero_matches_argmax() {
        let logits = [0.1, 3.0, 2.9, -1.0];
        let mut s = Sampler::from_seed(7);
        assert_eq!(
            s.sample(&logits, &params(0.0, Some(2), Some(0.5)), &HashSet::new()),
            1
        );
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let logits: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin()).collect();
        let a: Vec<u32> = {
            let mut s = Sampler::from_seed(123);
            (0..20)
                .map(|_| s.sample(&logits, &params(1.2, None, None), &HashSet::new()))
                .collect()
        };
        let b: Vec<u32> = {
            let mut s = Sampler::from_seed(123);
            (0..20)
                .map(|_| s.sample(&logits, &params(1.2, None, None), &HashSet::new()))
                .collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn top_k_restricts_support_exactly() {
        let logits = [0.0, 10.0, 9.0, -5.0];
        let mut s = Sampler::from_seed(1);
        for _ in 0..50 {
            let t = s.sample(&logits, &params(5.0, Some(2), None), &HashSet::new());
            assert!(t == 1 || t == 2, "got {t}");
        }
    }

    #[test]
    fn top_k_keeps_exactly_k_even_with_ties() {
        // All-equal logits: every value ties at the threshold.
        let logits = [1.0f32; 8];
        let out = top_k_filter(&logits, Some(3));
        assert_eq!(out.iter().filter(|&&l| l > f32::NEG_INFINITY).count(), 3);
        // Ties break toward lower token ids (deterministic).
        assert!(out[0] > f32::NEG_INFINITY);
        assert!(out[1] > f32::NEG_INFINITY);
        assert!(out[2] > f32::NEG_INFINITY);
        assert!(out[3] == f32::NEG_INFINITY);
    }

    #[test]
    fn top_p_restricts_support() {
        // One dominant token (p ≈ 0.9997 at temp 1) plus a long tail.
        let mut logits = vec![-20.0f32; 100];
        logits[3] = 8.0;
        let mut s = Sampler::from_seed(2);
        for _ in 0..50 {
            let t = s.sample(&logits, &params(1.0, None, Some(0.9)), &HashSet::new());
            assert_eq!(t, 3);
        }
    }

    #[test]
    fn repetition_penalty_flips_argmax() {
        let logits = vec![1.0f32, 0.9];
        let mut p = params(0.0, None, None);
        p.repetition_penalty = Some(2.0);
        let mut s = Sampler::from_seed(3);
        assert_eq!(s.sample(&logits, &p, &set(&[0])), 1);
        // Without penalty the argmax is 0 for the same logits.
        let mut s = Sampler::from_seed(3);
        assert_eq!(
            s.sample(&logits, &params(0.0, None, None), &HashSet::new()),
            0
        );
    }

    #[test]
    fn repetition_penalty_scales_both_signs() {
        let mut logits = vec![2.0f32, -2.0];
        apply_repetition_penalty(&mut logits, &set(&[0, 1, 1]), 2.0);
        assert_eq!(logits[0], 1.0); // positive divided
        assert_eq!(logits[1], -4.0); // negative multiplied
    }

    #[test]
    fn softmax_sums_to_one_and_ignores_neg_inf() {
        let p = softmax(&[f32::NEG_INFINITY, 1.0, 0.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(p[0], 0.0);
    }

    #[test]
    fn top_p_always_keeps_best_token() {
        let probs = [0.5, 0.3, 0.2];
        let out = top_p_filter(&probs, Some(0.01));
        assert_eq!(out[0], 0.5);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn sample_distribution_handles_single_survivor() {
        let mut rng = StdRng::seed_from_u64(9);
        assert_eq!(
            sample_from_distribution(&[1.0, 2.0], &[0.0, 1.0], &mut rng),
            1
        );
    }
}
