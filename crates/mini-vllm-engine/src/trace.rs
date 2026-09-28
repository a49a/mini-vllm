//! Opt-in teaching trace. A bounded channel keeps disk I/O off the engine thread.
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
        Arc,
    },
    time::{Duration, Instant},
};

const TRACE_LIMIT: usize = 100_000;
const QUEUE_CAPACITY: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceShutdown {
    Complete,
    Failed,
    TimedOut,
}

pub struct TraceWriter {
    sender: Option<SyncSender<serde_json::Value>>,
    finished: Option<Receiver<bool>>,
    shutdown_result: Option<TraceShutdown>,
    errors: Arc<AtomicU64>,
    timeouts: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    started: Instant,
    count: usize,
}

impl TraceWriter {
    pub fn new(path: Option<&Path>) -> std::io::Result<Self> {
        Self::new_with_counters(
            path,
            Default::default(),
            Default::default(),
            Default::default(),
        )
    }

    pub fn new_with_counters(
        path: Option<&Path>,
        dropped: Arc<AtomicU64>,
        errors: Arc<AtomicU64>,
        timeouts: Arc<AtomicU64>,
    ) -> std::io::Result<Self> {
        if let Some(path) = path {
            let file = OpenOptions::new().write(true).create_new(true).open(path)?;
            Self::with_sink(file, dropped, errors, timeouts)
        } else {
            Ok(Self {
                sender: None,
                finished: None,
                shutdown_result: None,
                dropped,
                errors,
                timeouts,
                started: Instant::now(),
                count: 0,
            })
        }
    }

    fn with_sink(
        sink: impl Write + Send + 'static,
        dropped: Arc<AtomicU64>,
        errors: Arc<AtomicU64>,
        timeouts: Arc<AtomicU64>,
    ) -> std::io::Result<Self> {
        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        let (done, finished) = mpsc::channel();
        let worker_dropped = dropped.clone();
        let worker_errors = errors.clone();
        std::thread::Builder::new()
            .name("mini-vllm-trace".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut file = BufWriter::new(sink);
                    let result = drain_trace(&mut file, receiver, &worker_dropped);
                    // Do not retry a failed flush in BufWriter::drop.
                    let (sink, _) = file.into_parts();
                    drop(sink);
                    result
                }));
                if !matches!(result, Ok(Ok(()))) {
                    worker_errors.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        "teaching trace writer failed or panicked; trace may be incomplete"
                    );
                }
                let _ = done.send(matches!(result, Ok(Ok(()))));
            })?;
        Ok(Self {
            sender: Some(sender),
            finished: Some(finished),
            shutdown_result: None,
            dropped,
            errors,
            timeouts,
            started: Instant::now(),
            count: 0,
        })
    }

    /// Sticky outcome of the first shutdown attempt. A timeout detaches the
    /// worker; subsequent calls return TimedOut even if it later finishes.
    pub fn shutdown(&mut self, timeout: Duration) -> TraceShutdown {
        if let Some(result) = self.shutdown_result {
            return result;
        }
        self.sender.take();
        let result = match self.finished.take().map(|done| done.recv_timeout(timeout)) {
            None | Some(Ok(true)) => TraceShutdown::Complete,
            Some(Ok(false)) => TraceShutdown::Failed,
            Some(Err(mpsc::RecvTimeoutError::Timeout)) => {
                self.timeouts.fetch_add(1, Ordering::Relaxed);
                tracing::warn!("teaching trace shutdown timed out; writer detached");
                TraceShutdown::TimedOut
            }
            Some(Err(mpsc::RecvTimeoutError::Disconnected)) => {
                self.errors.fetch_add(1, Ordering::Relaxed);
                TraceShutdown::Failed
            }
        };
        self.shutdown_result = Some(result);
        result
    }

    pub fn emit(&mut self, event: impl FnOnce() -> serde_json::Value) {
        let Some(sender) = &self.sender else {
            return;
        };
        if self.count > TRACE_LIMIT {
            return;
        }
        let mut event = if self.count == TRACE_LIMIT {
            serde_json::json!({"event":"truncated","limit":TRACE_LIMIT})
        } else {
            event()
        };
        event["schema_version"] = 1.into();
        event["elapsed_us"] = (self.started.elapsed().as_micros() as u64).into();
        self.count += 1;
        if let Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) = sender.try_send(event) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl Drop for TraceWriter {
    fn drop(&mut self) {
        self.shutdown(Duration::from_millis(250));
    }
}

fn write_event(file: &mut impl Write, event: &serde_json::Value) -> std::io::Result<()> {
    serde_json::to_writer(&mut *file, event).map_err(std::io::Error::other)?;
    file.write_all(b"\n")
}

fn drain_trace(
    file: &mut impl Write,
    receiver: Receiver<serde_json::Value>,
    dropped: &AtomicU64,
) -> std::io::Result<()> {
    for event in receiver {
        write_event(file, &event)?;
    }
    let count = dropped.load(Ordering::Relaxed);
    if count > 0 {
        write_event(
            file,
            &serde_json::json!({"event":"trace_dropped","count":count,"schema_version":1}),
        )?;
    }
    file.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_trace_is_lazy_and_existing_files_are_never_overwritten() {
        TraceWriter::new(None)
            .unwrap()
            .emit(|| panic!("disabled trace must not allocate events"));
        let path = std::env::temp_dir().join(format!(
            "mini-vllm-trace-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut trace = TraceWriter::new(Some(&path)).unwrap();
        assert!(TraceWriter::new(Some(&path)).is_err());
        trace.count = TRACE_LIMIT - 1;
        trace.emit(|| serde_json::json!({"event":"queued","request_id":"test"}));
        trace.emit(|| panic!("limit emits truncation instead"));
        trace.emit(|| panic!("beyond limit stays lazy"));
        drop(trace);
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["schema_version"], 1);
        assert_eq!(lines[1]["event"], "truncated");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn full_queue_drops_without_waiting() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let mut trace = TraceWriter {
            sender: Some(sender),
            finished: None,
            shutdown_result: None,
            errors: Default::default(),
            timeouts: Default::default(),
            dropped: Arc::new(AtomicU64::new(0)),
            started: Instant::now(),
            count: 0,
        };
        trace.emit(|| serde_json::json!({"event":"first"}));
        trace.emit(|| serde_json::json!({"event":"second"}));
        assert_eq!(trace.dropped_count(), 1);
        assert_eq!(receiver.try_recv().unwrap()["event"], "first");
    }

    #[test]
    fn drain_flushes_events_and_dropped_count() {
        let (sender, receiver) = mpsc::sync_channel(2);
        sender.send(serde_json::json!({"event":"first"})).unwrap();
        drop(sender);
        let dropped = AtomicU64::new(3);
        let mut bytes = Vec::new();
        drain_trace(&mut bytes, receiver, &dropped).unwrap();
        let lines: Vec<serde_json::Value> = bytes
            .split(|b| *b == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["event"], "first");
        assert_eq!(lines[1]["event"], "trace_dropped");
        assert_eq!(lines[1]["count"], 3);
    }

    #[test]
    fn write_failure_closes_the_receiver() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("disk unavailable"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let (sender, receiver) = mpsc::sync_channel(1);
        sender.send(serde_json::json!({"event":"first"})).unwrap();
        assert!(drain_trace(&mut Broken, receiver, &AtomicU64::new(0)).is_err());
        assert!(sender
            .try_send(serde_json::json!({"event":"second"}))
            .is_err());
    }
    #[test]
    fn blocked_flush_does_not_block_shutdown() {
        struct Slow {
            entered: mpsc::Sender<()>,
            release: Receiver<()>,
        }
        impl Write for Slow {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
                Ok(())
            }
        }
        let (entered, waiting) = mpsc::channel();
        let (release, blocked) = mpsc::channel();
        let timeouts = Arc::new(AtomicU64::new(0));
        let mut trace = TraceWriter::with_sink(
            Slow {
                entered,
                release: blocked,
            },
            Default::default(),
            Default::default(),
            timeouts.clone(),
        )
        .unwrap();
        trace.emit(|| serde_json::json!({"event":"test"}));
        trace.sender.take();
        waiting.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(trace.shutdown(Duration::ZERO), TraceShutdown::TimedOut);
        assert_eq!(
            trace.shutdown(Duration::from_secs(2)),
            TraceShutdown::TimedOut
        );
        drop(trace); // Must return while the writer is still blocked.
        assert_eq!(timeouts.load(Ordering::Relaxed), 1);
        release.send(()).unwrap();
    }

    #[test]
    fn background_failure_is_observable_without_another_emit() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("disk unavailable"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let errors = Arc::new(AtomicU64::new(0));
        let mut trace = TraceWriter::with_sink(
            Broken,
            Default::default(),
            errors.clone(),
            Default::default(),
        )
        .unwrap();
        trace.emit(|| serde_json::json!({"event":"test"}));
        assert_eq!(
            trace.shutdown(Duration::from_secs(2)),
            TraceShutdown::Failed
        );
        assert_eq!(trace.shutdown(Duration::ZERO), TraceShutdown::Failed);
        assert_eq!(errors.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn successful_shutdown_is_sticky_and_disabled_trace_completes() {
        let mut trace = TraceWriter::with_sink(
            Vec::new(),
            Default::default(),
            Default::default(),
            Default::default(),
        )
        .unwrap();
        trace.emit(|| serde_json::json!({"event":"test"}));
        assert_eq!(
            trace.shutdown(Duration::from_secs(2)),
            TraceShutdown::Complete
        );
        assert_eq!(trace.shutdown(Duration::ZERO), TraceShutdown::Complete);
        assert_eq!(
            TraceWriter::new(None).unwrap().shutdown(Duration::ZERO),
            TraceShutdown::Complete
        );
    }
}
