//! Independent Transformers fixtures. The tiny model runs offline in CI;
//! real Qwen weights are opt-in and never downloaded during cargo test.
use candle_core::{DType, Device};
use mini_vllm_kv::KvCache;
use mini_vllm_model::{loader, BatchTokens, CausalLm};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Row {
    argmax: usize,
    values: Vec<(usize, f32)>,
}
#[derive(Deserialize)]
struct Reference {
    token_ids: Vec<u32>,
    rows: Vec<Row>,
    atol: f32,
    config_sha256: String,
    weights_sha256: std::collections::BTreeMap<String, String>,
}
fn hash(path: &Path) -> String {
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    format!("{:x}", hash.finalize())
}
fn compare(actual: &[f32], expected: &Row, atol: f32) {
    assert!(actual.iter().all(|v| v.is_finite()), "nonfinite logits");
    let max = actual
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    assert_eq!(max, expected.argmax, "argmax differs from HF reference");
    let error = expected
        .values
        .iter()
        .map(|&(i, v)| (actual[i] - v).abs())
        .fold(0f32, f32::max);
    assert!(
        error <= atol,
        "max sampled-logit absolute error {error} > {atol}"
    );
}
fn check(directory: &Path, reference_path: &Path) {
    check_on(directory, reference_path, Device::Cpu, DType::F32, None);
}
fn check_on(
    directory: &Path,
    reference_path: &Path,
    device: Device,
    dtype: DType,
    tolerance: Option<f32>,
) {
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    assert_eq!(
        hash(&directory.join("config.json")),
        reference.config_sha256
    );
    for (file, expected) in &reference.weights_sha256 {
        assert_eq!(&hash(&directory.join(file)), expected);
    }
    let model = loader::load_model(directory, dtype, device.clone()).unwrap();
    let positions: Vec<_> = (0..reference.token_ids.len() as u32).collect();
    let full = model
        .forward_nocache(&reference.token_ids, &positions)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    for (actual, expected) in full.iter().zip(&reference.rows) {
        compare(actual, expected, tolerance.unwrap_or(reference.atol));
    }
    for (paged, selective) in [(false, false), (true, false), (false, true), (true, true)] {
        let cfg = model.config();
        let mut cache = if paged {
            KvCache::new_paged(cfg.num_hidden_layers, 16, 3).unwrap()
        } else {
            KvCache::new(
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim(),
                16,
                dtype,
                &device,
            )
            .unwrap()
        };
        let mut start = 0;
        for count in [3, 2, 1, 1, 1] {
            let end = start + count;
            let batch = BatchTokens {
                token_ids: reference.token_ids[start..end].to_vec(),
                positions: positions[start..end].to_vec(),
                seq_lens: vec![count],
            };
            let logits = if selective {
                model
                    .forward_cached_selected(
                        &batch,
                        std::slice::from_mut(&mut cache),
                        if start == 0 { &[] } else { &[0] },
                    )
                    .unwrap()
            } else {
                model
                    .forward_cached(&batch, std::slice::from_mut(&mut cache))
                    .unwrap()
            };
            if selective && start == 0 {
                assert_eq!(logits.dims(), &[0, model.vocab_size()]);
                assert_eq!(cache.seq_len(), end);
            } else {
                let logits = logits
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                compare(
                    &logits[0],
                    &reference.rows[end - 1],
                    tolerance.unwrap_or(reference.atol),
                );
            }
            if paged && end == 3 {
                cache = cache.fork_prefix(3, 16).unwrap();
            }
            start = end;
        }
    }
}
#[test]
fn hf_tiny_logits_match_full_chunked_and_paged_paths() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    check(
        &fixtures.join("tiny-qwen2"),
        &fixtures.join("tiny-reference.json"),
    );
}
#[test]
#[ignore = "requires local Qwen2.5-0.5B weights; set MINI_VLLM_REFERENCE_MODEL"]
fn hf_real_qwen_logits_match_all_paths() {
    let model = PathBuf::from(
        std::env::var("MINI_VLLM_REFERENCE_MODEL").expect("set MINI_VLLM_REFERENCE_MODEL"),
    );
    let model = if model.is_absolute() {
        model
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(model)
    };
    check(
        &model,
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen2.5-0.5b-reference.json"),
    );
}

/// Opt-in: unavailable backends fail explicitly instead of silently skipping.
#[test]
#[ignore = "set MINI_VLLM_TEST_DEVICE and MINI_VLLM_TEST_DTYPE; enable metal/cuda feature"]
fn backend_dtype_reference() {
    let name = std::env::var("MINI_VLLM_TEST_DEVICE").expect("set MINI_VLLM_TEST_DEVICE");
    let dtype = mini_vllm_model::resolve_dtype(
        &std::env::var("MINI_VLLM_TEST_DTYPE").expect("set MINI_VLLM_TEST_DTYPE"),
    )
    .unwrap();
    let (device, _) = mini_vllm_model::resolve_device(&name).unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let tolerance = match dtype {
        DType::F32 => 0.002,
        DType::F16 => 0.005,
        DType::BF16 => 0.04,
        _ => panic!("unsupported test dtype"),
    };
    check_on(
        &fixtures.join("tiny-qwen2"),
        &fixtures.join("tiny-reference.json"),
        device,
        dtype,
        Some(tolerance),
    );
}
