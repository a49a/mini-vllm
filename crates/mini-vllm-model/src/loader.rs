//! Weight loading: Safetensors discovery (single file or sharded), header
//! inspection, and device-tensor loading.
//!
//! Flow: model directory → config.json → tokenizer files (owned by the
//! tokenizer crate) → Safetensors index → tensor mapping → device tensors.

use std::collections::{HashMap, HashSet};
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
        if index_path.metadata()?.len() > 16 * 1024 * 1024 {
            return Err(Error::Config("safetensors index exceeds 16 MiB".into()));
        }
        let json = std::fs::read_to_string(&index_path)?;
        let value: serde_json::Value = serde_json::from_str(&json)?;
        let weight_map = value
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| Error::Config("safetensors index missing `weight_map`".into()))?;
        if weight_map.is_empty() {
            return Err(Error::Config(
                "safetensors index has empty weight_map".into(),
            ));
        }
        let mut files = Vec::new();
        for (name, value) in weight_map {
            if name.is_empty() {
                return Err(Error::Config(
                    "safetensors index has empty tensor name".into(),
                ));
            }
            let file = value
                .as_str()
                .ok_or_else(|| Error::Config(format!("invalid shard for tensor {name}")))?;
            if Path::new(file).file_name().and_then(|n| n.to_str()) != Some(file)
                || !file.ends_with(".safetensors")
            {
                return Err(Error::Config(format!("invalid shard filename {file}")));
            }
            files.push(file.to_owned());
        }
        files.sort();
        files.dedup();
        let paths: Vec<PathBuf> = files.into_iter().map(|f| dir.join(f)).collect();
        for p in &paths {
            if !p.is_file() {
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

fn read_header(path: &Path) -> Result<(HashMap<String, serde_json::Value>, u64)> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    let file_len = file.metadata()?.len();
    if header_len == 0 || header_len > 64 * 1024 * 1024 || header_len > file_len.saturating_sub(8) {
        return Err(Error::Config(format!(
            "invalid safetensors header length in {}",
            path.display()
        )));
    }
    let header_len = usize::try_from(header_len)
        .map_err(|_| Error::Config("header length overflows usize".into()))?;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    Ok((
        serde_json::from_slice(&header_bytes)?,
        file_len - 8 - header_len as u64,
    ))
}

fn dtype_bytes(dtype: &str) -> Option<u64> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Some(1),
        "U16" | "I16" | "F16" | "BF16" => Some(2),
        "U32" | "I32" | "F32" => Some(4),
        "U64" | "I64" | "F64" => Some(8),
        _ => None,
    }
}

/// Read tensor metadata from every weight file without loading tensors.
pub fn read_headers(files: &[PathBuf]) -> Result<Vec<(String, Vec<TensorMeta>)>> {
    let mut out = Vec::with_capacity(files.len());
    let mut names = HashSet::new();
    for path in files {
        let (header, data_len) = read_header(path)?;
        let mut metas = Vec::new();
        let mut spans = Vec::new();
        for (name, meta) in header {
            if name == "__metadata__" {
                continue;
            }
            if !names.insert(name.clone()) {
                return Err(Error::Config(format!(
                    "duplicate tensor {name} across shards"
                )));
            }
            let invalid =
                || Error::Config(format!("invalid metadata for {name} in {}", path.display()));
            let dtype = meta
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(invalid)?
                .to_owned();
            let width = dtype_bytes(&dtype).ok_or_else(invalid)?;
            let dims = meta
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(invalid)?;
            let shape = dims
                .iter()
                .map(|d| {
                    d.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(invalid)
                })
                .collect::<Result<Vec<_>>>()?;
            let num_elements = shape
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .ok_or_else(invalid)?;
            let offsets = meta
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .filter(|v| v.len() == 2)
                .ok_or_else(invalid)?;
            let start = offsets[0].as_u64().ok_or_else(invalid)?;
            let end = offsets[1].as_u64().ok_or_else(invalid)?;
            if end < start
                || end > data_len
                || end - start
                    != (num_elements as u64)
                        .checked_mul(width)
                        .ok_or_else(invalid)?
            {
                return Err(invalid());
            }
            spans.push((start, end));
            metas.push(TensorMeta {
                name,
                dtype,
                shape,
                num_elements,
            });
        }
        spans.sort_unstable();
        if spans.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(Error::Config(format!(
                "overlapping tensor offsets in {}",
                path.display()
            )));
        }
        if metas.is_empty() {
            return Err(Error::Config(format!("no tensors in {}", path.display())));
        }
        metas.sort_by(|a, b| a.name.cmp(&b.name));
        out.push((
            path.file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default(),
            metas,
        ));
    }
    // A shard manifest must name every tensor exactly once, in the file where it lives.
    if let Some(dir) = files.first().and_then(|p| p.parent()) {
        let index = dir.join("model.safetensors.index.json");
        if index.exists() && !dir.join("model.safetensors").exists() {
            if index.metadata()?.len() > 16 * 1024 * 1024 {
                return Err(Error::Config("safetensors index exceeds 16 MiB".into()));
            }
            let value: serde_json::Value = serde_json::from_slice(&std::fs::read(index)?)?;
            let map = value
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| Error::Config("invalid weight_map".into()))?;
            let actual: HashMap<&str, &str> = out
                .iter()
                .flat_map(|(file, metas)| {
                    metas.iter().map(move |m| (m.name.as_str(), file.as_str()))
                })
                .collect();
            if map.len() != actual.len()
                || map
                    .iter()
                    .any(|(name, file)| file.as_str() != actual.get(name.as_str()).copied())
            {
                return Err(Error::Config(
                    "safetensors index does not match shard tensors".into(),
                ));
            }
        }
    }
    Ok(out)
}

/// Load every tensor onto `device`, casting to `dtype`.
pub fn load_tensors(files: &[PathBuf], dtype: DType, device: &Device) -> Result<Tensors> {
    read_headers(files)?;
    let mut out: Tensors = HashMap::new();
    for path in files {
        let loaded = candle_core::safetensors::load(path, device)?;
        for (name, t) in loaded {
            if out.insert(name.clone(), t.to_dtype(dtype)?).is_some() {
                return Err(Error::Config(format!("duplicate tensor {name}")));
            }
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
            total = total.saturating_add(m.num_elements);
            dtypes.insert(m.dtype.clone());
            count += 1;
        }
    }
    (total, dtypes.into_iter().collect(), count)
}

fn validate_qwen2_shapes(
    cfg: &crate::config::ModelConfig,
    headers: &[(String, Vec<TensorMeta>)],
) -> Result<()> {
    let by_name: HashMap<&str, &TensorMeta> = headers
        .iter()
        .flat_map(|(_, metas)| metas.iter().map(|m| (m.name.as_str(), m)))
        .collect();
    let h = cfg.hidden_size;
    let kv = cfg.num_key_value_heads * cfg.head_dim();
    let i = cfg.intermediate_size;
    let check = |name: &str, expected: &[usize], required: bool| -> Result<()> {
        match by_name.get(name) {
            Some(meta) if meta.shape == expected => Ok(()),
            Some(meta) => Err(Error::Config(format!(
                "{name} has shape {:?}; expected {expected:?}",
                meta.shape
            ))),
            None if required => Err(Error::MissingWeight(name.to_owned())),
            None => Ok(()),
        }
    };
    check("model.embed_tokens.weight", &[cfg.vocab_size, h], true)?;
    check("model.norm.weight", &[h], true)?;
    check(
        "lm_head.weight",
        &[cfg.vocab_size, h],
        !cfg.tie_word_embeddings,
    )?;
    for layer in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{layer}.");
        check(&format!("{p}input_layernorm.weight"), &[h], true)?;
        check(&format!("{p}post_attention_layernorm.weight"), &[h], true)?;
        for (name, shape) in [
            ("self_attn.q_proj.weight", vec![h, h]),
            ("self_attn.k_proj.weight", vec![kv, h]),
            ("self_attn.v_proj.weight", vec![kv, h]),
            ("self_attn.o_proj.weight", vec![h, h]),
            ("mlp.gate_proj.weight", vec![i, h]),
            ("mlp.up_proj.weight", vec![i, h]),
            ("mlp.down_proj.weight", vec![h, i]),
        ] {
            check(&format!("{p}{name}"), &shape, true)?;
        }
        for (name, width) in [("q_proj.bias", h), ("k_proj.bias", kv), ("v_proj.bias", kv)] {
            check(&format!("{p}self_attn.{name}"), &[width], false)?;
        }
    }
    Ok(())
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
    validate_qwen2_shapes(&cfg, &headers)?;
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

    fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mini-vllm-loader-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }

    fn write_shard(path: &Path, header: serde_json::Value, data_len: usize) {
        use std::io::Write;
        let json = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&json).unwrap();
        file.write_all(&vec![0u8; data_len]).unwrap();
    }

    #[test]
    fn missing_weights_error() {
        assert!(matches!(
            discover_weight_files("/nonexistent"),
            Err(Error::NoWeights)
        ));
    }

    #[test]
    fn rejects_oversized_header_and_corrupt_tensor_metadata() {
        use std::io::Write;
        let dir = temp_dir();
        let path = dir.join("model.safetensors");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&u64::MAX.to_le_bytes()).unwrap();
        drop(file);
        assert!(read_headers(std::slice::from_ref(&path)).is_err());
        for header in [
            serde_json::json!({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}),
            serde_json::json!({"x":{"dtype":"F32","shape":[u64::MAX,2],"data_offsets":[0,4]}}),
            serde_json::json!({"x":{"dtype":"F32","shape":[1,"bad"],"data_offsets":[0,4]}}),
        ] {
            write_shard(&path, header, 4);
            assert!(read_headers(std::slice::from_ref(&path)).is_err());
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_index_traversal_and_mismatched_shards() {
        let dir = temp_dir();
        let index = dir.join("model.safetensors.index.json");
        std::fs::write(&index, r#"{"weight_map":{"x":"../outside.safetensors"}}"#).unwrap();
        assert!(discover_weight_files(&dir).is_err());
        let shard = dir.join("part.safetensors");
        write_shard(
            &shard,
            serde_json::json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
            4,
        );
        std::fs::write(&index, r#"{"weight_map":{"y":"part.safetensors"}}"#).unwrap();
        let files = discover_weight_files(&dir).unwrap();
        assert!(read_headers(&files).is_err());
        std::fs::write(&index, r#"{"weight_map":{"x":"part.safetensors"}}"#).unwrap();
        assert_eq!(read_headers(&files).unwrap()[0].1[0].name, "x");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_duplicate_tensor_across_shards() {
        let dir = temp_dir();
        let header = serde_json::json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}});
        let a = dir.join("a.safetensors");
        let b = dir.join("b.safetensors");
        write_shard(&a, header.clone(), 4);
        write_shard(&b, header, 4);
        assert!(read_headers(&[a, b]).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_wrong_embedding_shape_before_loading_weights() {
        let cfg = crate::config::ModelConfig::from_json(
            r#"{
            "hidden_size": 4, "intermediate_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "vocab_size": 8, "max_position_embeddings": 16
        }"#,
        )
        .unwrap();
        let headers = vec![(
            "model.safetensors".to_owned(),
            vec![TensorMeta {
                name: "model.embed_tokens.weight".to_owned(),
                dtype: "F32".to_owned(),
                shape: vec![8, 5],
                num_elements: 40,
            }],
        )];
        assert!(
            matches!(validate_qwen2_shapes(&cfg, &headers), Err(Error::Config(message)) if message.contains("model.embed_tokens.weight"))
        );
    }

    #[test]
    fn malformed_headers_are_rejected_without_panicking() {
        let dir = temp_dir();
        let path = dir.join("model.safetensors");
        let mut seed = 0x517c_c1b7_2722_0a95u64;
        for len in 0..256 {
            let mut bytes = vec![0u8; len];
            for byte in &mut bytes {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *byte = seed as u8;
            }
            std::fs::write(&path, bytes).unwrap();
            assert!(read_headers(std::slice::from_ref(&path)).is_err());
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
