//! Opt-in teaching trace. A bounded channel keeps disk I/O off the engine thread.
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, SyncSender, TrySendError},
        Arc,
    },
    thread::JoinHandle,
    time::Instant,
};

const TRACE_LIMIT: usize = 100_000;
const QUEUE_CAPACITY: usize = 1024;

pub struct TraceWriter {
    sender: Option<SyncSender<serde_json::Value>>,
    worker: Option<JoinHandle<()>>,
    dropped: Arc<AtomicU64>,
    started: Instant,
    count: usize,
}

impl TraceWriter {
    pub fn new(path: Option<&Path>) -> std::io::Result<Self> {
        Self::new_with_counter(path, Arc::new(AtomicU64::new(0)))
    }

    pub fn new_with_counter(path: Option<&Path>, dropped: Arc<AtomicU64>) -> std::io::Result<Self> {
        let (sender, worker) = if let Some(path) = path {
            let file = OpenOptions::new().write(true).create_new(true).open(path)?;
            let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
            let dropped_for_worker = dropped.clone();
            let worker = std::thread::Builder::new()
                .name("mini-vllm-trace".into())
                .spawn(move || {
                    let mut file = BufWriter::new(file);
                    for event in receiver {
                        if let Err(error) = write_event(&mut file, &event) {
                            tracing::warn!(%error, "teaching trace writer stopped after write failure");
                            return;
                        }
                    }
                    let count = dropped_for_worker.load(Ordering::Relaxed);
                    if count > 0 {
                        if let Err(error) = write_event(&mut file, &serde_json::json!({"event":"trace_dropped","count":count,"schema_version":1})) {
                            tracing::warn!(%error, "could not write dropped trace count");
                        }
                    }
                    if let Err(error) = file.flush() {
                        tracing::warn!(%error, "could not flush teaching trace");
                    }
                })?;
            (Some(sender), Some(worker))
        } else {
            (None, None)
        };
        Ok(Self {
            sender,
            worker,
            dropped,
            started: Instant::now(),
            count: 0,
        })
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
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                tracing::warn!("teaching trace writer panicked");
            }
        }
    }
}

fn write_event(
    file: &mut BufWriter<std::fs::File>,
    event: &serde_json::Value,
) -> std::io::Result<()> {
    serde_json::to_writer(&mut *file, event).map_err(std::io::Error::other)?;
    file.write_all(b"\n")
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
            worker: None,
            dropped: Arc::new(AtomicU64::new(0)),
            started: Instant::now(),
            count: 0,
        };
        trace.emit(|| serde_json::json!({"event":"first"}));
        trace.emit(|| serde_json::json!({"event":"second"}));
        assert_eq!(trace.dropped_count(), 1);
        assert_eq!(receiver.try_recv().unwrap()["event"], "first");
    }
}
