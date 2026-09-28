//! Launch the real CLI and real tiny model; no mock engine or network downloads.
#![cfg(unix)]
use serde_json::Value;
use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Server {
    child: Child,
    directory: PathBuf,
    port: u16,
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_dir_all(&self.directory);
    }
}
impl Server {
    fn start(occupied: Option<u16>) -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let directory = std::env::temp_dir().join(format!(
            "mini-vllm-process-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let source = root.join("../mini-vllm-model/tests/fixtures/tiny-qwen2");
        fs::copy(
            source.join("model.safetensors"),
            directory.join("model.safetensors"),
        )
        .unwrap();
        let mut config: Value =
            serde_json::from_slice(&fs::read(source.join("config.json")).unwrap()).unwrap();
        config["eos_token_id"] = serde_json::json!([]);
        config["max_position_embeddings"] = 4096.into();
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let tok = mini_vllm_tokenizer::testutil::char_tokenizer();
        let mut tokenizer: Value = serde_json::from_str(&tok.to_string(false).unwrap()).unwrap();
        for id in 12..64 {
            tokenizer["model"]["vocab"][format!("t{id}")] = id.into();
        }
        fs::write(
            directory.join("tokenizer.json"),
            serde_json::to_vec(&tokenizer).unwrap(),
        )
        .unwrap();
        fs::write(directory.join("tokenizer_config.json"), serde_json::to_vec(&serde_json::json!({
            "chat_template": include_str!("../../mini-vllm-tokenizer/src/templates/qwen2.5-instruct.jinja")
        })).unwrap()).unwrap();
        let port = occupied.unwrap_or_else(|| {
            TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .port()
        });
        let log = fs::File::create(directory.join("server.log")).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_mini-vllm"))
            .args([
                "serve",
                "--model",
                directory.to_str().unwrap(),
                "--device",
                "cpu",
                "--dtype",
                "f32",
                "--port",
                &port.to_string(),
                "--max-model-len",
                "4096",
                "--max-kv-tokens",
                "8192",
                "--max-batch-tokens",
                "16",
                "--shutdown-timeout-secs",
                "1",
                "--preprocessing-workers",
                "1",
                "--preprocessing-waiting",
                "1",
                "--preprocessing-timeout-ms",
                "500",
            ])
            .stdout(Stdio::from(log.try_clone().unwrap()))
            .stderr(Stdio::from(log))
            .spawn()
            .unwrap();
        let mut server = Self {
            child,
            directory,
            port,
        };
        if occupied.is_none() {
            let deadline = Instant::now() + Duration::from_secs(20);
            loop {
                assert!(
                    server.child.try_wait().unwrap().is_none(),
                    "{}",
                    server.log()
                );
                if server
                    .request("GET", "/health", "")
                    .is_ok_and(|s| s.starts_with("HTTP/1.1 200"))
                {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "startup timeout: {}",
                    server.log()
                );
                std::thread::sleep(Duration::from_millis(20));
            }
        }
        server
    }
    fn log(&self) -> String {
        fs::read_to_string(self.directory.join("server.log")).unwrap_or_default()
    }
    fn socket(&self) -> std::io::Result<TcpStream> {
        let socket = TcpStream::connect(("127.0.0.1", self.port))?;
        socket.set_read_timeout(Some(Duration::from_secs(2)))?;
        socket.set_write_timeout(Some(Duration::from_secs(2)))?;
        Ok(socket)
    }
    fn request(&self, method: &str, path: &str, body: &str) -> std::io::Result<String> {
        let mut socket = self.socket()?;
        write!(socket, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len())?;
        let mut response = String::new();
        socket.read_to_string(&mut response)?;
        Ok(response)
    }
    fn metrics(&self) -> Value {
        let response = self.request("GET", "/metrics", "").unwrap();
        serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap()
    }
    fn until(&self, check: impl Fn(&Value) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let metrics = self.metrics();
            if check(&metrics) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "metrics never converged: {metrics}"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    fn terminate(&mut self) {
        assert!(Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status()
            .unwrap()
            .success());
        self.wait(true);
    }
    fn wait(&mut self, success: bool) {
        // 1s transport grace + 5s engine join + 250ms runtime teardown, with CI margin.
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert_eq!(status.success(), success, "{}", self.log());
                return;
            }
            assert!(
                Instant::now() < deadline,
                "exit deadline exceeded: {}",
                self.log()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
#[test]
fn slow_upload_and_nonreading_peer_do_not_prevent_sigterm_exit() {
    let mut server = Server::start(None);
    let mut upload = server.socket().unwrap();
    write!(upload, "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 10000\r\n\r\n{{").unwrap();
    server.until(|m| m["preprocessing"]["body_read"]["in_flight"] == 1);
    assert!(server
        .request("GET", "/health", "")
        .unwrap()
        .starts_with("HTTP/1.1 200"));
    let mut response = String::new();
    upload.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 408"), "{response}");
    server.until(|m| {
        m["preprocessing"]["body_read"]["in_flight"] == 0
            && m["preprocessing"]["timeouts_total"] == 1
    });
    let mut unread = server.socket().unwrap();
    let body = r#"{"prompt":"a","max_tokens":4000,"stream":true,"temperature":0}"#;
    write!(unread, "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    server.until(|m| m["requests_running"].as_i64().unwrap_or(0) > 0);
    // Keep the socket open without reading through the entire shutdown.
    server.terminate();
    drop(unread);
}
#[test]
fn sse_disconnect_reclaims_real_engine_and_next_request_succeeds() {
    let mut server = Server::start(None);
    let mut socket = server.socket().unwrap();
    let body = r#"{"prompt":"a","max_tokens":4000,"stream":true,"temperature":0}"#;
    write!(socket, "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    let mut bytes = [0; 4096];
    let mut response = String::new();
    while !response.contains("data:") {
        let n = socket.read(&mut bytes).unwrap();
        assert!(n > 0);
        response.push_str(&String::from_utf8_lossy(&bytes[..n]));
    }
    drop(socket);
    server.until(|m| {
        m["requests_cancelled"].as_u64().unwrap_or(0) > 0
            && m["kv_blocks_used"] == 0
            && m["requests_running"] == 0
    });
    let response = server
        .request(
            "POST",
            "/v1/completions",
            r#"{"prompt":"a","max_tokens":2,"temperature":0}"#,
        )
        .unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    server.terminate();
}
#[test]
fn binding_error_exits_without_leaving_runtime_or_engine_waiting() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut server = Server::start(Some(listener.local_addr().unwrap().port()));
    server.wait(false);
    assert!(
        server.log().contains("binding host:port") || server.log().contains("binding 127.0.0.1"),
        "{}",
        server.log()
    );
}
