#![no_main]

use libfuzzer_sys::fuzz_target;
use std::{path::PathBuf, sync::OnceLock};

fn corpus_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let path = std::env::temp_dir().join(format!("mini-vllm-fuzz-{}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create fuzz scratch directory");
        path
    })
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > 64 * 1024 {
        return;
    }
    let dir = corpus_dir();
    let single = dir.join("model.safetensors");
    let index = dir.join("model.safetensors.index.json");
    if data[0] & 1 == 0 {
        let _ = std::fs::remove_file(&index);
        std::fs::write(&single, &data[1..]).expect("write fuzzed safetensors file");
        let files =
            mini_vllm_model::loader::discover_weight_files(dir).expect("single file exists");
        let _ = mini_vllm_model::loader::read_headers(&files);
    } else {
        let _ = std::fs::remove_file(&single);
        std::fs::write(&index, &data[1..]).expect("write fuzzed index");
        if let Ok(files) = mini_vllm_model::loader::discover_weight_files(dir) {
            let _ = mini_vllm_model::loader::read_headers(&files);
        }
    }
});
