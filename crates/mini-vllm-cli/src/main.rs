//! `mini-vllm` CLI: inspect / generate / serve.

mod runtime;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use mini_vllm_core::{
    EngineConfig, FinishReason, GenerationEvent, GenerationRequest, SamplingParams,
};
use mini_vllm_model::loader;
use mini_vllm_model::{resolve_device, resolve_dtype, CausalLm, ModelConfig};
use mini_vllm_tokenizer::{ModelChatTemplate, TokenizerWrapper};

#[derive(Parser)]
#[command(
    name = "mini-vllm",
    version,
    about = "A small, educational LLM inference engine in Rust (vLLM-style)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Clone)]
enum Command {
    /// Inspect a local model directory (config, tokenizer, weights).
    Inspect {
        /// Path to the model directory.
        #[arg(long)]
        model: PathBuf,
        /// cpu | metal | cuda | auto
        #[arg(long, default_value = "cpu")]
        device: String,
        /// auto | f32 | f16 | bf16
        #[arg(long, default_value = "auto")]
        dtype: String,
    },
    /// Run a single prompt through the model and stream tokens to stdout.
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        /// 0 = greedy (default; deterministic demos).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long)]
        top_p: Option<f32>,
        #[arg(long)]
        repetition_penalty: Option<f32>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value = "auto")]
        dtype: String,
    },
    /// Serve an OpenAI-compatible HTTP API.
    Serve {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        /// auto | cpu | metal | cuda
        #[arg(long, default_value = "auto")]
        device: String,
        /// auto | f32 | f16 | bf16
        #[arg(long, default_value = "auto")]
        dtype: String,
        /// Context limit; defaults to min(model max, 8192).
        #[arg(long)]
        max_model_len: Option<usize>,
        #[arg(long, default_value_t = 32)]
        max_num_seqs: usize,
        #[arg(long, default_value_t = 2048)]
        max_batch_tokens: usize,
        #[arg(long, default_value_t = 256)]
        max_prefill_chunk_tokens: usize,
        /// Candidates considered for admission; 1 is strict FIFO.
        #[arg(long, default_value_t = 1)]
        admission_lookahead: usize,
        /// Block further bypasses once a request has waited this long.
        #[arg(long, default_value_t = 1000)]
        admission_max_wait_ms: u64,
        /// Log batch membership, positions and page changes for teaching.
        #[arg(long)]
        trace_request: bool,
        #[arg(long)]
        trace_jsonl: Option<PathBuf>,
        #[arg(long, default_value_t = 512)]
        default_max_new_tokens: usize,
        #[arg(long, default_value_t = 0)]
        queue_timeout_ms: u64,
        #[arg(long, default_value_t = 0)]
        request_timeout_ms: u64,
        #[arg(long, default_value_t = 30)]
        shutdown_timeout_secs: u64,
        /// Concurrent CPU tokenization/template jobs.
        #[arg(long, default_value_t = 2)]
        preprocessing_workers: usize,
        /// Additional requests admitted before body read/tokenization.
        #[arg(long, default_value_t = 16)]
        preprocessing_waiting: usize,
        /// Total body-read, preprocessing queue and CPU-result budget.
        #[arg(long, default_value_t = 10000)]
        preprocessing_timeout_ms: u64,
        /// Total KV budget in tokens.
        #[arg(long, default_value_t = 32768)]
        max_kv_tokens: usize,
        /// Maximum requests waiting for a running slot.
        #[arg(long, default_value_t = 256)]
        max_waiting_requests: usize,
        /// Grace period for delivering completed output after KV release.
        #[arg(long, default_value_t = 1000)]
        output_drain_timeout_ms: u64,
        #[arg(long, default_value_t = 16)]
        kv_block_size: usize,
        /// Use the contiguous KV reference path instead of physical pages.
        #[arg(long)]
        contiguous_kv: bool,
        /// Additional token budget for LRU prefix snapshots (0 disables).
        #[arg(long, default_value_t = 0)]
        prefix_cache_tokens: usize,
        /// Base seed for sampling when a request has no seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Log filter (also configurable via RUST_LOG).
        #[arg(long, default_value = "info")]
        log_level: String,
    },
}

fn init_tracing(default_level: &str) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

fn resolve(device: &str, dtype: &str) -> Result<(candle_core::Device, candle_core::DType)> {
    let (device, name) = resolve_device(device).map_err(anyhow::Error::msg)?;
    let dtype = resolve_dtype(dtype).map_err(anyhow::Error::msg)?;
    tracing::debug!(device = name, dtype = ?dtype, "device/dtype resolved");
    Ok((device, dtype))
}

fn load_tokenizer_or_die(dir: &Path) -> Result<TokenizerWrapper> {
    TokenizerWrapper::from_model_dir(dir)
        .with_context(|| format!("loading tokenizer from {}", dir.display()))
}

fn cmd_inspect(model: PathBuf, device: String, dtype: String) -> Result<()> {
    let (dev, dev_name) = resolve_device(&device).map_err(anyhow::Error::msg)?;
    let dt = resolve_dtype(&dtype).map_err(anyhow::Error::msg)?;
    let cfg = ModelConfig::from_dir(&model).context("parsing config.json")?;
    let files = loader::discover_weight_files(&model).context("discovering weights")?;
    let headers = loader::read_headers(&files)?;
    let (params, weight_dtypes, tensor_count) = loader::summarize(&headers);
    let tokenizer = load_tokenizer_or_die(&model)?;

    println!("Architecture:       {}", cfg.architectures.join(","));
    println!("Layers:             {}", cfg.num_hidden_layers);
    println!("Hidden size:        {}", cfg.hidden_size);
    println!("Intermediate size:  {}", cfg.intermediate_size);
    println!("Attention heads:    {}", cfg.num_attention_heads);
    println!("KV heads (GQA):     {}", cfg.num_key_value_heads);
    println!("Head dim:           {}", cfg.head_dim());
    println!("Vocabulary size:    {}", cfg.vocab_size);
    println!(
        "Tokenizer vocab:    {} (with added)",
        tokenizer.vocab_size(true)
    );
    println!("Max context:        {}", cfg.max_position_embeddings);
    println!("RoPE theta:         {}", cfg.rope_theta);
    println!(
        "Tied embeddings:    {}",
        if cfg.tie_word_embeddings { "yes" } else { "no" }
    );
    println!("EOS token ids:      {:?}", cfg.eos_token_ids);
    println!("Weight files:       {}", files.len());
    println!("Weight tensors:     {tensor_count}");
    println!("Weight dtypes:      {}", weight_dtypes.join(","));
    println!("Parameters:         {params}");
    println!("Compute dtype:      {dt:?}");
    println!("Device:             {dev_name} ({dev:?})");
    Ok(())
}

fn cmd_generate(args: Command) -> Result<()> {
    // Destructure inside so clap stays the single source of truth.
    let Command::Generate {
        model,
        prompt,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
        device,
        dtype,
    } = args
    else {
        unreachable!("cmd_generate called with a different subcommand")
    };
    init_tracing("warn");

    let (dev, dt) = resolve(&device, &dtype)?;
    let tokenizer = Arc::new(load_tokenizer_or_die(&model)?);
    let loaded = loader::load_model(&model, dt, dev).context("loading model")?;
    let max_model_len = loaded
        .config()
        .max_position_embeddings
        .min(mini_vllm_core::EngineConfig::default().max_model_len);
    let prompt_token_ids = tokenizer.encode(&prompt, true)?;
    let sampling = SamplingParams {
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
    };
    let request = GenerationRequest {
        id: "cli-generate".into(),
        prompt_token_ids,
        sampling,
        max_new_tokens,
        stop_token_ids: vec![],
        stop_strings: vec![],
    };
    request
        .validate(max_model_len)
        .map_err(anyhow::Error::msg)?;

    let config = EngineConfig {
        max_model_len,
        ..EngineConfig::default()
    };
    let handle = mini_vllm_engine::spawn_engine(
        Arc::new(loaded),
        Some(Arc::clone(&tokenizer)),
        config,
        seed.unwrap_or(42),
    )?;

    use std::io::Write;
    let mut events = handle.generate(request).map_err(anyhow::Error::msg)?;
    while let Some(ev) = events.blocking_recv() {
        match ev {
            GenerationEvent::Token { text, .. } => {
                print!("{text}");
                let _ = std::io::stdout().flush();
            }
            GenerationEvent::Finished { reason, usage } => {
                let _ = std::io::stdout().flush();
                eprintln!();
                if matches!(reason, FinishReason::Error | FinishReason::Cancelled) {
                    anyhow::bail!("generation {}", reason.as_str());
                } else {
                    eprintln!(
                        "[finish: {} | prompt {} tok | generated {} tok]",
                        reason.as_str(),
                        usage.prompt_tokens,
                        usage.completion_tokens
                    );
                }
                return Ok(());
            }
            GenerationEvent::Error { message, .. } => {
                anyhow::bail!("generation error: {message}");
            }
        }
    }
    anyhow::bail!("generation ended without a terminal event")
}

fn cmd_serve(args: Command) -> Result<()> {
    let Command::Serve {
        model,
        host,
        port,
        device,
        dtype,
        max_model_len,
        max_num_seqs,
        max_batch_tokens,
        max_prefill_chunk_tokens,
        admission_lookahead,
        admission_max_wait_ms,
        trace_request,
        trace_jsonl,
        default_max_new_tokens,
        queue_timeout_ms,
        request_timeout_ms,
        shutdown_timeout_secs,
        preprocessing_workers,
        preprocessing_waiting,
        preprocessing_timeout_ms,
        max_kv_tokens,
        max_waiting_requests,
        output_drain_timeout_ms,
        kv_block_size,
        contiguous_kv,
        prefix_cache_tokens,
        seed,
        log_level,
    } = args
    else {
        unreachable!("cmd_serve called with a different subcommand")
    };
    init_tracing(&log_level);

    let preprocessing_config = mini_vllm_server::preprocessing::PreprocessConfig {
        workers: preprocessing_workers,
        waiting: preprocessing_waiting,
        timeout: std::time::Duration::from_millis(preprocessing_timeout_ms),
    };
    mini_vllm_server::preprocessing::Preprocessor::new(preprocessing_config)
        .map_err(anyhow::Error::msg)?;
    let (dev, dt) = resolve(&device, &dtype)?;
    let tokenizer = Arc::new(load_tokenizer_or_die(&model)?);
    let template = Arc::new(ModelChatTemplate::from_model_dir(&model)?);
    tracing::info!(path = %model.display(), "loading model");
    let loaded = loader::load_model(&model, dt, dev)?;
    let model_max = loaded.config().max_position_embeddings;
    let max_model_len = max_model_len.unwrap_or(model_max.min(8192)).min(model_max);
    let model_id = model
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "local-model".into());
    let vocab_size = tokenizer.vocab_size(true);

    let config = EngineConfig {
        max_model_len,
        max_num_seqs,
        max_batch_tokens,
        max_prefill_chunk_tokens,
        admission_lookahead,
        admission_max_wait_ms,
        trace_requests: trace_request,
        trace_jsonl,
        default_max_new_tokens,
        queue_timeout_ms,
        request_timeout_ms,
        max_kv_tokens,
        max_waiting_requests,
        output_drain_timeout_ms,
        kv_block_size,
        prefix_cache_tokens,
        paged_kv: !contiguous_kv,
        ..EngineConfig::default()
    };
    let runtime = runtime::BoundedRuntime::new().context("building tokio runtime")?;
    let handle = mini_vllm_engine::spawn_engine(
        Arc::new(loaded),
        Some(Arc::clone(&tokenizer)),
        config,
        seed,
    )?;

    let state: mini_vllm_server::SharedState = Arc::new(mini_vllm_server::AppState {
        engine: Arc::new(handle.clone()),
        tokenizer,
        template,
        model_id: model_id.clone(),
        max_model_len,
        vocab_size,
    });

    let cleanup_handle = handle.clone();
    let result = runtime.block_on(async move {
        let app = mini_vllm_server::routes::router_with_preprocessing(state, preprocessing_config)
            .map_err(anyhow::Error::msg)?;
        let addr: SocketAddr = format!("{host}:{port}")
            .parse()
            .context("parsing host:port")?;
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("binding {addr}"))?;
        tracing::info!(%addr, model = %model_id, max_model_len, "OpenAI-compatible server listening");
        let shutdown_handle = handle.clone();
        let grace = std::time::Duration::from_secs(shutdown_timeout_secs);
        let server = axum::serve(listener, app).with_graceful_shutdown(async move {
            shutdown_signal().await;
            shutdown_handle.request_shutdown(mini_vllm_engine::ShutdownMode::Drain, grace);
        });
        // Bound transport drain too: a peer may never read its response.
        let server_task = tokio::spawn(async move { server.await });
        while handle.is_accepting() && !server_task.is_finished() {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let abort = server_task.abort_handle();
        match tokio::time::timeout(grace, server_task).await {
            Ok(result) => result??,
            Err(_) => { abort.abort(); handle.request_shutdown(mini_vllm_engine::ShutdownMode::Cancel, std::time::Duration::ZERO); }
        }
        Ok::<(), anyhow::Error>(())
    });
    // Join directly outside Tokio: queued blocking jobs cannot delay cleanup.
    // Do this even if binding or transport handling returned an error.
    cleanup_handle.request_shutdown(
        mini_vllm_engine::ShutdownMode::Cancel,
        std::time::Duration::ZERO,
    );
    let joined = cleanup_handle
        .join(std::time::Duration::from_secs(5))
        .map_err(anyhow::Error::msg);
    drop(runtime);
    result?;
    joined?;
    tracing::info!("server drained; exiting");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received");
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Command::Inspect {
            model,
            device,
            dtype,
        } => cmd_inspect(model.clone(), device.clone(), dtype.clone()),
        gen @ Command::Generate { .. } => cmd_generate(gen.clone()),
        serve @ Command::Serve { .. } => cmd_serve(serve.clone()),
    }
}
