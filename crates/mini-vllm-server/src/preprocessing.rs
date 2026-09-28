//! Bounded admission before reading JSON; CPU work never runs on Tokio workers.
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
        })
    }
    // Axum rejection responses are intentionally returned by value.
    #[allow(clippy::result_large_err)]
    fn admit(&self) -> Result<Ticket, Response> {
        let permit = self
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| busy("preprocessing queue is full"))?;
        Ok(Ticket {
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
        if Instant::now() >= deadline {
            return Err(expired());
        }
        tokio::time::timeout_at(deadline, async move {
            let worker = self
                .workers
                .acquire_owned()
                .await
                .map_err(|_| busy("preprocessing stopped"))?;
            if Instant::now() >= deadline {
                return Err(expired());
            }
            let admission = self.permit;
            let result = tokio::task::spawn_blocking(move || {
                // Cancellation/timeouts cannot cancel running CPU work. Keep BOTH
                // permits here until it actually ends, preventing over-admission.
                let (_worker, _admission) = (worker, admission);
                if Instant::now() >= deadline {
                    None
                } else {
                    Some(work())
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
                    axum::http::StatusCode::BAD_REQUEST,
                    e,
                    "invalid_request_error",
                )
            })
        })
        .await
        .map_err(|_| expired())?
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
        let ticket = processor.admit()?;
        let Json(value) =
            tokio::time::timeout_at(ticket.deadline, Json::<T>::from_request(request, state))
                .await
                .map_err(|_| expired())?
                .map_err(|e| {
                    crate::api::api_error(e.status(), e.body_text(), "invalid_request_error")
                })?;
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
}
