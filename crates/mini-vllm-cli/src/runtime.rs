//! Explicit runtime teardown on success, error, and unwinding.
use std::{future::Future, time::Duration};

pub struct BoundedRuntime(Option<tokio::runtime::Runtime>);
impl BoundedRuntime {
    pub fn new() -> std::io::Result<Self> {
        tokio::runtime::Runtime::new().map(|runtime| Self(Some(runtime)))
    }
    pub fn block_on<F: Future>(&self, future: F) -> F::Output {
        self.0
            .as_ref()
            .expect("runtime is present until drop")
            .block_on(future)
    }
}
impl Drop for BoundedRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.0.take() {
            // Running blocking closures cannot be cancelled. Do not let them
            // turn a bounded HTTP/engine shutdown into an unbounded process wait.
            runtime.shutdown_timeout(Duration::from_millis(250));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Run the real Drop in a subprocess: a regression must not hang cargo test.
    #[test]
    fn blocked_runtime_child() {
        let Ok(mode) = std::env::var("MINI_VLLM_RUNTIME_TEST_MODE") else {
            return;
        };
        let execute = || -> Result<(), &'static str> {
            let runtime = BoundedRuntime::new().unwrap();
            runtime.block_on(async {
                let (started, ready) = tokio::sync::oneshot::channel();
                tokio::task::spawn_blocking(move || {
                    let _ = started.send(());
                    loop {
                        std::thread::park();
                    }
                });
                ready.await.unwrap();
            });
            match mode.as_str() {
                "error" => Err("injected error"),
                "panic" => panic!("injected panic"),
                _ => Ok(()),
            }
        };
        let _ = std::panic::catch_unwind(execute);
    }
    #[test]
    fn teardown_is_bounded_on_success_error_and_panic() {
        for mode in ["success", "error", "panic"] {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "runtime::tests::blocked_runtime_child",
                    "--nocapture",
                ])
                .env("MINI_VLLM_RUNTIME_TEST_MODE", mode)
                .spawn()
                .unwrap();
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "{mode}");
                    break;
                }
                if std::time::Instant::now() >= deadline {
                    child.kill().unwrap();
                    let _ = child.wait();
                    panic!("unbounded runtime teardown: {mode}");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
