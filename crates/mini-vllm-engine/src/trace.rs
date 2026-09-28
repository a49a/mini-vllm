//! Opt-in bounded teaching trace. Synchronous file I/O adds measurement overhead.
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

pub struct TraceWriter {
    file: Option<BufWriter<File>>,
    started: Instant,
    count: usize,
}
impl TraceWriter {
    pub fn new(path: Option<&Path>) -> std::io::Result<Self> {
        let file = path
            .map(|p| {
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(p)
                    .map(BufWriter::new)
            })
            .transpose()?;
        Ok(Self {
            file,
            started: Instant::now(),
            count: 0,
        })
    }
    pub fn emit(&mut self, event: impl FnOnce() -> serde_json::Value) {
        if self.file.is_none() {
            return;
        }
        if self.count > 100_000 {
            return;
        }
        let mut event = if self.count == 100_000 {
            serde_json::json!({"event":"truncated","limit":100000})
        } else {
            event()
        };
        event["schema_version"] = 1.into();
        event["elapsed_us"] = (self.started.elapsed().as_micros() as u64).into();
        self.count += 1;
        let file = self.file.as_mut().unwrap();
        let result = serde_json::to_writer(&mut *file, &event)
            .map_err(std::io::Error::other)
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.flush());
        if let Err(error) = result {
            tracing::warn!(%error,"teaching trace disabled after write failure");
            self.file = None;
        }
    }
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
        trace.count = 99_999;
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
}
