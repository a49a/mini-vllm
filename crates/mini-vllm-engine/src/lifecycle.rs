//! Bounded in-flight registry and cancellation independent of command capacity.
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

pub type Registry = Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>;

#[derive(Debug)]
pub struct RequestLease {
    pub id: String,
    pub submitted_at: std::time::Instant,
    pub cancelled: Arc<AtomicBool>,
    pub registry: Registry,
}
impl RequestLease {
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}
impl Drop for RequestLease {
    fn drop(&mut self) {
        self.registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.id);
    }
}
