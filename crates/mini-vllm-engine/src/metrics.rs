//! Engine metrics (design doc §49): counters and latency aggregates
//! maintained with atomics so the HTTP layer can read a snapshot at any
//! time without touching engine internals.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Snapshot exposed over `GET /metrics` (JSON for now; Prometheus later).
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    pub uptime_seconds: f64,
    pub requests_total: u64,
    pub requests_running: i64,
    pub requests_waiting: i64,
    pub requests_finishing: u64,
    pub cached_prefix_tokens: u64,
    pub kv_storage_allocations_total: u64,
    pub trace_events_dropped: u64,
    pub trace_writer_errors: u64,
    pub trace_shutdown_timeouts: u64,
    pub requests_finished: u64,
    pub requests_cancelled: u64,
    pub requests_failed: u64,
    pub prompt_tokens_total: u64,
    pub generated_tokens_total: u64,
    pub prefill_steps_total: u64,
    pub decode_steps_total: u64,
    pub average_decode_batch_size: f64,
    pub time_to_first_token_ms_avg: f64,
    pub inter_token_latency_ms_avg: f64,
    pub request_latency_ms_avg: f64,
    /// Generated model tokens in the trailing ten seconds / observed window.
    pub tokens_per_second: f64,
    pub lifetime_tokens_per_second: f64,
    pub prefix_cache_hit_tokens: u64,
    pub scheduled_tokens_total: u64,
    pub model_steps_total: u64,
    pub kv_blocks_total: u64,
    pub kv_blocks_used: u64,
    pub kv_blocks_free: u64,
    pub kv_active_sequences: usize,
}

#[derive(Debug, Default)]
pub struct Metrics {
    started_at_unix_ms: AtomicU64,
    recent_tokens: Mutex<VecDeque<(Instant, u64)>>,
    pub prefix_cache_hit_tokens: AtomicU64,
    pub scheduled_tokens_total: AtomicU64,
    pub model_steps_total: AtomicU64,
    pub requests_total: AtomicU64,
    pub requests_finished: AtomicU64,
    pub requests_cancelled: AtomicU64,
    pub requests_failed: AtomicU64,
    pub requests_running: AtomicI64,
    pub requests_waiting: AtomicI64,
    pub requests_finishing: AtomicU64,
    pub cached_prefix_tokens: AtomicU64,
    pub kv_storage_allocations_total: std::sync::Arc<AtomicU64>,
    pub trace_events_dropped: std::sync::Arc<AtomicU64>,
    pub trace_writer_errors: std::sync::Arc<AtomicU64>,
    pub trace_shutdown_timeouts: std::sync::Arc<AtomicU64>,
    pub prompt_tokens_total: AtomicU64,
    pub generated_tokens_total: AtomicU64,
    pub prefill_steps_total: AtomicU64,
    pub decode_steps_total: AtomicU64,
    /// Sum of per-step batch sizes, for the running average.
    pub decode_batch_tokens_total: AtomicU64,
    /// Microsecond sums to avoid float atomics.
    pub ttft_us_sum: AtomicU64,
    pub ttft_count: AtomicU64,
    pub itl_us_sum: AtomicU64,
    pub itl_count: AtomicU64,
    pub request_latency_us_sum: AtomicU64,
    pub request_latency_count: AtomicU64,
    pub kv_blocks_total: AtomicU64,
    pub kv_blocks_used: AtomicU64,
    pub kv_active_sequences: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            started_at_unix_ms: AtomicU64::new(now),
            ..Default::default()
        }
    }

    pub fn record_token(&self) {
        let now = Instant::now();
        let mut recent = self.recent_tokens.lock().unwrap_or_else(|e| e.into_inner());
        while recent
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > Duration::from_secs(10))
        {
            recent.pop_front();
        }
        // Bucket by 100ms so metric storage is bounded independently of TPS.
        if let Some((t, n)) = recent.back_mut() {
            if now.duration_since(*t) < Duration::from_millis(100) {
                *n += 1;
                return;
            }
        }
        recent.push_back((now, 1));
    }

    /// Build a snapshot from the atomics. KV gauges are refreshed by the
    /// engine every step, so this never blocks on the engine thread.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let or = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let ori = |a: &AtomicI64| a.load(Ordering::Relaxed);
        let uptime_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
            .saturating_sub(self.started_at_unix_ms.load(Ordering::Relaxed))
            as f64
            / 1000.0;

        let ttft_count = or(&self.ttft_count);
        let itl_count = or(&self.itl_count);
        let lat_count = or(&self.request_latency_count);
        let decode_steps = or(&self.decode_steps_total);
        let generated = or(&self.generated_tokens_total);

        MetricsSnapshot {
            uptime_seconds: uptime_secs,
            requests_total: or(&self.requests_total),
            requests_running: ori(&self.requests_running),
            requests_waiting: ori(&self.requests_waiting),
            requests_finishing: or(&self.requests_finishing),
            cached_prefix_tokens: or(&self.cached_prefix_tokens),
            kv_storage_allocations_total: or(&self.kv_storage_allocations_total),
            trace_events_dropped: or(&self.trace_events_dropped),
            trace_writer_errors: or(&self.trace_writer_errors),
            trace_shutdown_timeouts: or(&self.trace_shutdown_timeouts),
            requests_finished: or(&self.requests_finished),
            requests_cancelled: or(&self.requests_cancelled),
            requests_failed: or(&self.requests_failed),
            prompt_tokens_total: or(&self.prompt_tokens_total),
            generated_tokens_total: generated,
            prefill_steps_total: or(&self.prefill_steps_total),
            decode_steps_total: decode_steps,
            average_decode_batch_size: if decode_steps > 0 {
                or(&self.decode_batch_tokens_total) as f64 / decode_steps as f64
            } else {
                0.0
            },
            time_to_first_token_ms_avg: if ttft_count > 0 {
                or(&self.ttft_us_sum) as f64 / ttft_count as f64 / 1000.0
            } else {
                0.0
            },
            inter_token_latency_ms_avg: if itl_count > 0 {
                or(&self.itl_us_sum) as f64 / itl_count as f64 / 1000.0
            } else {
                0.0
            },
            request_latency_ms_avg: if lat_count > 0 {
                or(&self.request_latency_us_sum) as f64 / lat_count as f64 / 1000.0
            } else {
                0.0
            },
            tokens_per_second: {
                let now = Instant::now();
                let recent = self.recent_tokens.lock().unwrap_or_else(|e| e.into_inner());
                let count: u64 = recent
                    .iter()
                    .filter(|(t, _)| now.duration_since(*t) <= Duration::from_secs(10))
                    .map(|(_, n)| n)
                    .sum();
                count as f64 / uptime_secs.clamp(0.001, 10.0)
            },
            prefix_cache_hit_tokens: or(&self.prefix_cache_hit_tokens),
            scheduled_tokens_total: or(&self.scheduled_tokens_total),
            model_steps_total: or(&self.model_steps_total),
            lifetime_tokens_per_second: if uptime_secs > 0.0 {
                generated as f64 / uptime_secs
            } else {
                0.0
            },
            kv_blocks_total: or(&self.kv_blocks_total),
            kv_blocks_used: or(&self.kv_blocks_used),
            kv_blocks_free: or(&self.kv_blocks_total).saturating_sub(or(&self.kv_blocks_used)),
            kv_active_sequences: or(&self.kv_active_sequences) as usize,
        }
    }
}
