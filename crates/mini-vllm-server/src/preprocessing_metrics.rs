//! Per-request guards keep gauges correct across errors, cancellation and panic.
use serde::Serialize;
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

#[derive(Debug, Default, Clone, Serialize)]
pub struct Stage {
    pub in_flight: u64,
    pub completed: u64,
    pub total_us: u64,
    pub max_us: u64,
}
#[derive(Debug, Default, Clone, Serialize)]
pub struct Snapshot {
    pub admitted_total: u64,
    pub rejected_total: u64,
    pub timeouts_total: u64,
    pub callers_cancelled_total: u64,
    pub job_errors_total: u64,
    pub job_panics_total: u64,
    pub detached_jobs: u64,
    pub detached_jobs_total: u64,
    pub body_read: Stage,
    pub queue_wait: Stage,
    pub execution: Stage,
}
pub type Metrics = Arc<Mutex<Snapshot>>;
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Body,
    Queue,
    Execution,
    Done,
}
struct Local {
    phase: Phase,
    since: Instant,
    caller_gone: bool,
}
pub struct Observation {
    metrics: Metrics,
    local: Mutex<Local>,
}
impl Observation {
    pub fn new(metrics: Metrics) -> Arc<Self> {
        {
            let mut m = metrics.lock().unwrap_or_else(|e| e.into_inner());
            m.admitted_total += 1;
            m.body_read.in_flight += 1;
        }
        Arc::new(Self {
            metrics,
            local: Mutex::new(Local {
                phase: Phase::Body,
                since: Instant::now(),
                caller_gone: false,
            }),
        })
    }
    pub fn phase(&self, phase: Phase) {
        let mut state = self.local.lock().unwrap_or_else(|e| e.into_inner());
        if state.phase == Phase::Done || state.phase == phase {
            return;
        }
        let mut m = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
        let stage = match state.phase {
            Phase::Body => &mut m.body_read,
            Phase::Queue => &mut m.queue_wait,
            Phase::Execution => &mut m.execution,
            Phase::Done => unreachable!(),
        };
        stage.in_flight -= 1;
        stage.completed += 1;
        let elapsed = state.since.elapsed().as_micros().min(u64::MAX as u128) as u64;
        stage.total_us = stage.total_us.saturating_add(elapsed);
        stage.max_us = stage.max_us.max(elapsed);
        match phase {
            Phase::Body => m.body_read.in_flight += 1,
            Phase::Queue => m.queue_wait.in_flight += 1,
            Phase::Execution => m.execution.in_flight += 1,
            Phase::Done => {}
        }
        if state.phase == Phase::Execution && state.caller_gone {
            m.detached_jobs -= 1;
        }
        state.phase = phase;
        state.since = Instant::now();
    }
    pub fn timeout(&self) {
        self.metrics
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .timeouts_total += 1;
    }
    pub fn job_error(&self, panic: bool) {
        let mut m = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
        if panic {
            m.job_panics_total += 1;
        } else {
            m.job_errors_total += 1;
        }
    }
    fn caller_gone(&self, cancelled: bool) {
        let mut state = self.local.lock().unwrap_or_else(|e| e.into_inner());
        let mut m = self.metrics.lock().unwrap_or_else(|e| e.into_inner());
        if cancelled {
            m.callers_cancelled_total += 1;
        }
        if state.phase == Phase::Execution {
            state.caller_gone = true;
            m.detached_jobs += 1;
            m.detached_jobs_total += 1;
        } else {
            drop(m);
            drop(state);
            self.phase(Phase::Done);
        }
    }
}
pub struct Caller {
    pub observation: Arc<Observation>,
    pub completed: bool,
}
impl Drop for Caller {
    fn drop(&mut self) {
        self.observation.caller_gone(!self.completed);
    }
}
pub struct Job(pub Arc<Observation>);
impl Drop for Job {
    fn drop(&mut self) {
        self.0.phase(Phase::Done);
    }
}
