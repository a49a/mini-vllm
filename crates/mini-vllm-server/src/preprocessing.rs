//! Bounded admission before reading JSON; CPU work never runs on Tokio workers.
// Axum rejection responses are returned by value at this HTTP boundary.
// Newer Clippy versions also check the Result output of async functions.
#![allow(clippy::result_large_err)]
use crate::preprocessing_metrics::{Caller, Job, Metrics, Observation, Phase, Snapshot};
use axum::{
    extract::{FromRequest, Request},
    response::Response,
    Json,
};
use serde::de::DeserializeOwned;
use std::{sync::Arc, time::Duration};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore},
    time::Instant,
};

#[derive(Debug, Clone, Copy)]
pub struct PreprocessConfig {
    pub workers: usize,
    pub waiting: usize,
    /// Total budget: body read, queue wait, and preprocessing result.
    pub timeout: Duration,
}
impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            workers: 2,
            waiting: 16,
            timeout: Duration::from_secs(10),
        }
    }
}
#[derive(Clone)]
pub struct Preprocessor {
    admission: Arc<Semaphore>,
    workers: Arc<Semaphore>,
    timeout: Duration,
    metrics: Metrics,
}
impl Preprocessor {
    pub fn new(config: PreprocessConfig) -> Result<Self, &'static str> {
        let total = config
            .workers
            .checked_add(config.waiting)
            .ok_or("preprocessing capacity overflow")?;
        if config.workers == 0
            || total > Semaphore::MAX_PERMITS
            || config.timeout.is_zero()
            || Instant::now().checked_add(config.timeout).is_none()
        {
            return Err("preprocessing requires positive workers/timeout and bounded capacity");
        }
        Ok(Self {
            admission: Arc::new(Semaphore::new(total)),
            workers: Arc::new(Semaphore::new(config.workers)),
            timeout: config.timeout,
            metrics: Default::default(),
        })
    }
    pub fn snapshot(&self) -> Snapshot {
        self.metrics
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    fn admit(&self) -> Result<Ticket, Response> {
        let permit = self.admission.clone().try_acquire_owned().map_err(|_| {
            self.metrics
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .rejected_total += 1;
            busy("preprocessing queue is full")
        })?;
        let observation = Observation::new(self.metrics.clone());
        Ok(Ticket {
            caller: Caller {
                observation,
                completed: false,
            },
            permit,
            workers: self.workers.clone(),
            deadline: Instant::now() + self.timeout,
        })
    }
}
fn busy(message: &str) -> Response {
    crate::api::api_error(
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        message,
        "preprocessing_unavailable",
    )
}
fn expired() -> Response {
    crate::api::api_error(
        axum::http::StatusCode::REQUEST_TIMEOUT,
        "preprocessing deadline exceeded",
        "request_timeout",
    )
}

pub struct Ticket {
    caller: Caller,
    permit: OwnedSemaphorePermit,
    workers: Arc<Semaphore>,
    deadline: Instant,
}
impl Ticket {
    pub async fn run<T: Send + 'static>(
        self,
        work: impl FnOnce() -> Result<T, String> + Send + 'static,
    ) -> Result<T, Response> {
        let deadline = self.deadline;
        let mut caller = self.caller;
        caller.completed = false;
        let observation = caller.observation.clone();
        observation.phase(Phase::Queue);
        let outcome = tokio::time::timeout_at(deadline, async move {
            if Instant::now() >= deadline {
                return Err(expired());
            }
            let worker = self
                .workers
                .acquire_owned()
                .await
                .map_err(|_| busy("preprocessing stopped"))?;
            if Instant::now() >= deadline {
                return Err(expired());
            }
            let admission = self.permit;
            observation.phase(Phase::Execution);
            let job = Job(observation.clone());
            let result = tokio::task::spawn_blocking(move || {
                // Cancellation/timeouts cannot cancel running CPU work. Keep BOTH
                // permits here until it actually ends, preventing over-admission.
                let (_worker, _admission) = (worker, admission);
                // Publish completion before making capacity reusable.
                let _job = job;
                if Instant::now() >= deadline {
                    None
                } else {
                    Some(
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
                            Ok(Ok(value)) => Ok(value),
                            Ok(Err(error)) => {
                                observation.job_error(false);
                                Err(Some(error))
                            }
                            Err(_) => {
                                observation.job_error(true);
                                Err(None)
                            }
                        },
                    )
                }
            })
            .await
            .map_err(|_| {
                crate::api::api_error(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "preprocessing worker failed",
                    "internal_error",
                )
            })?;
            if Instant::now() >= deadline {
                return Err(expired());
            }
            result.ok_or_else(expired)?.map_err(|e| {
                crate::api::api_error(
                    if e.is_some() {
                        axum::http::StatusCode::BAD_REQUEST
                    } else {
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR
                    },
                    e.clone()
                        .unwrap_or_else(|| "preprocessing worker failed".into()),
                    if e.is_some() {
                        "invalid_request_error"
                    } else {
                        "internal_error"
                    },
                )
            })
        })
        .await
        .unwrap_or_else(|_| Err(expired()));
        if outcome
            .as_ref()
            .is_err_and(|response| response.status() == axum::http::StatusCode::REQUEST_TIMEOUT)
        {
            caller.observation.timeout();
        }
        caller.completed = true;
        outcome
    }
}

pub struct Input<T>(pub T, pub Ticket);
impl<S: Send + Sync, T: DeserializeOwned + Send> FromRequest<S> for Input<T> {
    type Rejection = Response;
    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let processor = request.extensions().get::<Preprocessor>().ok_or_else(|| {
            crate::api::api_error(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "preprocessing not configured",
                "internal_error",
            )
        })?;
        let mut ticket = processor.admit()?;
        let parsed =
            tokio::time::timeout_at(ticket.deadline, Json::<T>::from_request(request, state)).await;
        let value = match parsed {
            Ok(Ok(Json(value))) => value,
            Ok(Err(e)) => {
                ticket.caller.completed = true;
                return Err(crate::api::api_error(
                    e.status(),
                    e.body_text(),
                    "invalid_request_error",
                ));
            }
            Err(_) => {
                ticket.caller.completed = true;
                ticket.caller.observation.timeout();
                return Err(expired());
            }
        };
        ticket.caller.observation.phase(Phase::Queue);
        ticket.caller.completed = true;
        Ok(Self(value, ticket))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Release(Option<std::sync::mpsc::Sender<()>>);
    impl Drop for Release {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
        }
    }
    #[tokio::test]
    async fn cancelled_cpu_work_keeps_its_capacity_until_it_really_finishes() {
        let gate = Preprocessor::new(PreprocessConfig {
            workers: 1,
            waiting: 0,
            timeout: Duration::from_secs(5),
        })
        .unwrap();
        let (started, entered) = tokio::sync::oneshot::channel();
        let (tx, rx) = std::sync::mpsc::channel();
        let release = Release(Some(tx));
        let ticket = gate.admit().unwrap();
        let task = tokio::spawn(ticket.run(move || {
            let _ = started.send(());
            let _ = rx.recv();
            Ok(())
        }));
        tokio::time::timeout(Duration::from_secs(2), entered)
            .await
            .unwrap()
            .unwrap();
        task.abort();
        let _ = task.await;
        let snapshot = gate.snapshot();
        assert_eq!(snapshot.callers_cancelled_total, 1);
        assert_eq!(snapshot.detached_jobs, 1);
        assert_eq!(snapshot.execution.in_flight, 1);
        assert_eq!(
            snapshot.body_read.in_flight + snapshot.queue_wait.in_flight,
            0
        );
        assert!(
            gate.admit().is_err(),
            "abort cannot release a running CPU job's permit"
        );
        drop(release);
        tokio::time::timeout(Duration::from_secs(2), async {
            while gate.admission.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let snapshot = gate.snapshot();
        assert_eq!(snapshot.detached_jobs, 0);
        assert_eq!(snapshot.detached_jobs_total, 1);
        assert_eq!(snapshot.execution.in_flight, 0);
        assert_eq!(snapshot.execution.completed, 1);
        assert!(snapshot.execution.total_us > 0);
        assert!(gate.admit().is_ok());
    }
    #[tokio::test]
    async fn timed_out_worker_and_waiter_do_not_over_admit_or_execute_expired_work() {
        let gate = Preprocessor::new(PreprocessConfig {
            workers: 1,
            waiting: 1,
            timeout: Duration::from_millis(200),
        })
        .unwrap();
        let (started, entered) = tokio::sync::oneshot::channel();
        let (tx, rx) = std::sync::mpsc::channel();
        let release = Release(Some(tx));
        let task = tokio::spawn(gate.admit().unwrap().run(move || {
            let _ = started.send(());
            let _ = rx.recv();
            Ok(())
        }));
        tokio::time::timeout(Duration::from_secs(2), entered)
            .await
            .unwrap()
            .unwrap();
        let queued = gate.admit().unwrap();
        assert!(gate.admit().is_err());
        let result = queued
            .run(|| -> Result<(), String> { panic!("expired queued work must never start") })
            .await;
        assert_eq!(
            result.unwrap_err().status(),
            axum::http::StatusCode::REQUEST_TIMEOUT
        );
        assert_eq!(
            task.await.unwrap().unwrap_err().status(),
            axum::http::StatusCode::REQUEST_TIMEOUT
        );
        assert_eq!(gate.admission.available_permits(), 1);
        assert_eq!(gate.workers.available_permits(), 0);
        let snapshot = gate.snapshot();
        assert_eq!(snapshot.timeouts_total, 2);
        assert_eq!(snapshot.rejected_total, 1);
        assert_eq!(snapshot.callers_cancelled_total, 0);
        assert_eq!(snapshot.detached_jobs, 1);
        assert_eq!(snapshot.queue_wait.in_flight, 0);
        drop(release);
    }
    #[test]
    fn invalid_capacities_are_errors_not_panics() {
        for config in [
            PreprocessConfig {
                workers: 0,
                ..Default::default()
            },
            PreprocessConfig {
                workers: usize::MAX,
                ..Default::default()
            },
            PreprocessConfig {
                timeout: Duration::ZERO,
                ..Default::default()
            },
        ] {
            assert!(Preprocessor::new(config).is_err());
        }
    }
    #[tokio::test]
    async fn expired_ticket_never_starts_work_and_panics_release_capacity() {
        let gate = Preprocessor::new(PreprocessConfig {
            workers: 1,
            waiting: 0,
            ..Default::default()
        })
        .unwrap();
        let mut ticket = gate.admit().unwrap();
        ticket.deadline = Instant::now();
        assert_eq!(
            ticket
                .run(|| -> Result<(), String> { panic!("must not execute") })
                .await
                .unwrap_err()
                .status(),
            axum::http::StatusCode::REQUEST_TIMEOUT
        );
        let result = gate
            .admit()
            .unwrap()
            .run(|| -> Result<(), String> { panic!("worker panic") })
            .await;
        assert_eq!(
            result.unwrap_err().status(),
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        );
        assert!(gate.admit().is_ok());
    }
    #[tokio::test]
    async fn success_and_work_failures_leave_no_inflight_jobs() {
        let gate = Preprocessor::new(Default::default()).unwrap();
        assert_eq!(gate.admit().unwrap().run(|| Ok(42)).await.unwrap(), 42);
        assert!(gate
            .admit()
            .unwrap()
            .run(|| -> Result<(), String> { Err("invalid input".into()) })
            .await
            .is_err());
        assert!(gate
            .admit()
            .unwrap()
            .run(|| -> Result<(), String> { panic!("injected panic") })
            .await
            .is_err());
        let snapshot = gate.snapshot();
        assert_eq!(snapshot.admitted_total, 3);
        assert_eq!(snapshot.job_errors_total, 1);
        assert_eq!(snapshot.job_panics_total, 1);
        assert_eq!(snapshot.execution.completed, 3);
        assert_eq!(
            snapshot.execution.in_flight
                + snapshot.queue_wait.in_flight
                + snapshot.body_read.in_flight,
            0
        );
        assert_eq!(
            snapshot.detached_jobs + snapshot.callers_cancelled_total + snapshot.timeouts_total,
            0
        );
    }
}
