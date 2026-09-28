//! Device and dtype selection.
//!
//! `--device auto` prefers CUDA, then Metal, then CPU. An *explicitly*
//! requested device that is unavailable is a hard error — never a silent
//! fallback.

use candle_core::{DType, Device};

use crate::error::{Error, Result};

#[cfg(all(target_os = "macos", feature = "metal"))]
fn try_metal() -> Option<Device> {
    use std::panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe};
    // `candle`'s `MetalDevice::new` panics (`swap_remove` on an empty
    // device list) instead of returning an error when no Metal device is
    // visible to the process — e.g. sandboxed or headless sessions.
    // Neutralize the panic (silenced via a temporary hook; this runs once
    // at startup, before the engine thread exists) and treat it as
    // "unavailable" so `auto` falls back to CPU and an explicit `metal`
    // request surfaces a clean error.
    let previous_hook = take_hook();
    set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| Device::new_metal(0)));
    set_hook(previous_hook);
    result.ok().and_then(|r| r.ok())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn try_metal() -> Option<Device> {
    None
}

#[cfg(feature = "cuda")]
fn try_cuda() -> Option<Device> {
    if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).ok()
    } else {
        None
    }
}

#[cfg(not(feature = "cuda"))]
fn try_cuda() -> Option<Device> {
    None
}

/// Resolve `cpu|metal|cuda|auto` into a device plus a printable name.
pub fn resolve_device(requested: &str) -> Result<(Device, &'static str)> {
    match requested {
        "cpu" => Ok((Device::Cpu, "cpu")),
        "auto" => {
            if let Some(d) = try_cuda() {
                Ok((d, "cuda"))
            } else if let Some(d) = try_metal() {
                Ok((d, "metal"))
            } else {
                Ok((Device::Cpu, "cpu"))
            }
        }
        "metal" => try_metal().map(|d| (d, "metal")).ok_or_else(|| {
            Error::UnsupportedDevice(
                "metal (no Metal device visible to this process, or not built with the `metal` feature)"
                    .into(),
            )
        }),
        "cuda" => try_cuda()
            .map(|d| (d, "cuda"))
            .ok_or_else(|| Error::UnsupportedDevice("cuda (build with the `cuda` feature)".into())),
        other => Err(Error::UnsupportedDevice(other.to_string())),
    }
}

/// Resolve `auto|f32|f16|bf16` into a dtype. `auto` uses F32 (correctness
/// first; pass an explicit dtype to trade precision for speed/memory).
pub fn resolve_dtype(requested: &str) -> Result<DType> {
    match requested {
        "auto" | "f32" | "float32" => Ok(DType::F32),
        "f16" | "float16" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        other => Err(Error::UnsupportedDtype(other.to_string())),
    }
}
