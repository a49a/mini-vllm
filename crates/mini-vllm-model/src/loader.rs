//! Weight loading: Safetensors discovery (single file or sharded), header
//! inspection, and device-tensor loading.
//!
//! Flow: model directory → config.json → tokenizer files (owned by the
//! tokenizer crate) → Safetensors index → tensor mapping → device tensors.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::qwen2::CausalLm;

/// Loaded weight tensors, keyed by HF name.
pub type Tensors = HashMap<String, Tensor>;

/// Metadata for one tensor, parsed straight from the Safetensors header
/// (no weight bytes are read — this powers `mini-vllm inspect`).
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub num_elements: usize,
}

/// Discover the Safetensors files backing a model directory:
/// `model.safetensors` or the shards listed in `model.safetensors.index.json`.
pub fn discover_weight_files(dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let dir = dir.as_ref();
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
        let value: serde_json::Value = serde_json::from_str(&json)?;
        let weight_map = value
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| Error::Config("safetensors index missing `weight_map`".into()))?;
        let mut files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        files.sort();
        files.dedup();
        let paths: Vec<PathBuf> = files.into_iter().map(|f| dir.join(f)).collect();
        for p in &paths {
            if !p.exists() {
                return Err(Error::Config(format!(
                    "shard {} referenced by index is missing",
                    p.display()
                )));
            }
        }
        return Ok(paths);
    }
    Err(Error::NoWeights)
}

fn read_header(path: &Path) -> Result<HashMap<String, serde_json::Value>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    Ok(serde_json::from_slice(&header_bytes)?)
}

/// Read tensor metadata from every weight file without loading tensors.
pub fn read_headers(files: &[PathBuf]) -> Result<Vec<(String, Vec<TensorMeta>)>> {
    let mut out = Vec::with_capacity(files.len());
    for path in files {
        let header = read_header(path)?;
        let mut metas: Vec<TensorMeta> = header
            .into_iter()
            .filter_map(|(name, meta)| {
                if name == "__metadata__" {
                    return None;
                }
                let dtype = meta.get("dtype")?.as_str()?.to_string();
                let shape: Vec<usize> = meta
                    .get("shape")?
                    .as_array()?
                    .iter()
                    .filter_map(|d| d.as_u64().map(|d| d as usize))
                    .collect();
                let num_elements = shape.iter().product::<usize>();
                Some(TensorMeta {
                    name,
                    dtype,
                    shape,
                    num_elements,
                })
            })
            .collect();
        metas.sort_by(|a, b| a.name.cmp(&b.name));
        out.push((
            path.file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default(),
            metas,
        ));
    }
    Ok(out)
}

/// Load every tensor onto `device`, casting to `dtype`.
pub fn load_tensors(files: &[PathBuf], dtype: DType, device: &Device) -> Result<Tensors> {
    let mut out: Tensors = HashMap::new();
    for path in files {
        let loaded = candle_core::safetensors::load(path, device)?;
        for (name, t) in loaded {
            out.insert(name, t.to_dtype(dtype)?);
        }
    }
    Ok(out)
}

pub fn get_weight<'a>(tensors: &'a Tensors, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| Error::MissingWeight(name.to_string()))
}

pub fn opt_weight<'a>(tensors: &'a Tensors, name: &str) -> Result<Option<&'a Tensor>> {
    Ok(tensors.get(name))
}

/// Count parameters and list dtypes from headers (for `inspect`).
pub fn summarize(headers: &[(String, Vec<TensorMeta>)]) -> (usize, Vec<String>, usize) {
    let mut total = 0usize;
    let mut dtypes = std::collections::BTreeSet::new();
    let mut count = 0usize;
    for (_, metas) in headers {
        for m in metas {
            total += m.num_elements;
            dtypes.insert(m.dtype.clone());
            count += 1;
        }
    }
    (total, dtypes.into_iter().collect(), count)
}

/// Load a complete Qwen2 model from a local directory.
pub fn load_model(
    dir: impl AsRef<Path>,
    dtype: DType,
    device: Device,
) -> Result<crate::qwen2::Qwen2> {
    if device.is_cpu() && dtype == DType::BF16 {
        return Err(Error::UnsupportedDtype(
            "BF16 on CPU: this Candle backend has no BF16 matmul; use F32 or F16".into(),
        ));
    }
    let dir = dir.as_ref();
    let t0 = std::time::Instant::now();
    let cfg = crate::config::ModelConfig::from_dir(dir)?;
    let files = discover_weight_files(dir)?;
    let headers = read_headers(&files)?;
    let (param_count, weight_dtypes, tensor_count) = summarize(&headers);

    tracing::info!(
        path = %dir.display(),
        architecture = cfg.architectures.join(","),
        layers = cfg.num_hidden_layers,
        hidden = cfg.hidden_size,
        heads = cfg.num_attention_heads,
        kv_heads = cfg.num_key_value_heads,
        vocab = cfg.vocab_size,
        parameters = param_count,
        weight_dtypes = weight_dtypes.join(","),
        "loading model weights"
    );

    let tensors = load_tensors(&files, dtype, &device)?;
    let model = crate::qwen2::Qwen2::load(cfg, &tensors, dtype, &device)?;
    tracing::info!(
        dtype = ?dtype,
        device = ?model.device(),
        tensors = tensor_count,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "model loaded"
    );
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_weights_error() {
        assert!(matches!(
            discover_weight_files("/nonexistent"),
            Err(Error::NoWeights)
        ));
    }
}
