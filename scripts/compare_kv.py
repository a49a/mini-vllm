#!/usr/bin/env python3
"""Repeatable contiguous/paged/prefix HTTP experiment; no third-party Python packages.

Each trial starts a fresh server, warms the same prompt once, then measures the
same deterministic workload. RSS is process memory (not device VRAM), sampled
with ps. Allocation counts cover persistent KV tensors, not all allocations.
"""
import argparse
import concurrent.futures
import http.client
import hashlib
import json
import platform
from pathlib import Path
import socket
import statistics
import subprocess
import threading
import time
import uuid

from benchmark import stream_one


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def get_json(port, path, host="127.0.0.1"):
    conn = http.client.HTTPConnection(host, port, timeout=2)
    try:
        conn.request('GET', path)
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f'{path}: HTTP {response.status}')
        return json.loads(response.read())
    finally:
        conn.close()


def trial(args, mode, index, output):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(Path(args.binary).resolve()), 'serve', '--model', str(Path(args.model).resolve()),
               '--device', args.device, '--dtype', args.dtype, '--port', str(port),
               '--max-model-len', '512', '--max-kv-tokens', '4096',
               '--max-num-seqs', str(args.concurrency), '--max-batch-tokens', '32',
               '--max-prefill-chunk-tokens', '16', '--kv-block-size', '8',
               '--shutdown-timeout-secs', '2']
    if mode == 'contiguous':
        command += ['--contiguous-kv']
    if mode == 'prefix':
        command += ['--prefix-cache-tokens', '256']
    trace_path = None
    if args.trace:
        trace_path = output.parent / f'{output.stem}-{mode}-{index}-{uuid.uuid4().hex[:8]}.jsonl'
        command += ['--trace-jsonl', str(trace_path.resolve())]
    log_path = output.parent / f'{output.stem}-{mode}-{index}.log'
    samples, stop = [], threading.Event()
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        def sample_memory():
            while not stop.is_set():
                try:
                    value = subprocess.check_output(['ps', '-o', 'rss=', '-p', str(process.pid)], text=True)
                    samples.append(int(value.strip()) * 1024)
                except (ValueError, subprocess.CalledProcessError):
                    pass
                stop.wait(0.05)
        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited; see {log_path}')
                try:
                    get_json(port, '/health')
                    break
                except (OSError, RuntimeError):
                    if time.monotonic() > deadline:
                        raise RuntimeError(f'server startup timeout; see {log_path}')
                    time.sleep(0.1)
            stream_one('127.0.0.1', port, args.prompt, args.max_tokens)
            before = get_json(port, '/metrics')
            results, errors = [], []
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(stream_one, '127.0.0.1', port, args.prompt, args.max_tokens)
                           for _ in range(args.requests)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as error:
                        errors.append(str(error))
            elapsed = time.perf_counter() - start
            after = get_json(port, '/metrics')
            tokens = sum(row['tokens'] for row in results)
            ttft = [row['ttft'] * 1000 for row in results if row['ttft'] is not None]
            return {'mode': mode, 'trial': index, 'command': command, 'successful': len(results),
                    'failures': errors, 'tokens': tokens, 'wall_seconds': elapsed,
                    'tokens_per_second': tokens / elapsed,
                    'ttft_ms_median': statistics.median(ttft) if ttft else None,
                    'peak_process_rss_bytes': max(samples, default=None),
                    'kv_storage_allocations': after['kv_storage_allocations_total'] - before['kv_storage_allocations_total'],
                    'computed_prompt_tokens': after['prompt_tokens_total'] - before['prompt_tokens_total'],
                    'prefix_hit_tokens': after['prefix_cache_hit_tokens'] - before['prefix_cache_hit_tokens'],
                    'trace_events_dropped': after['trace_events_dropped'] - before['trace_events_dropped'],
                    'trace_path': str(trace_path) if trace_path else None,
                    'metrics_after': after}
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stop.set()
            sampler.join()
            if process.returncode != 0:
                raise RuntimeError(f'server shutdown exited {process.returncode}; see {log_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='target/release/mini-vllm')
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='f32')
    parser.add_argument('--requests', type=int, default=8)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--modes', nargs='+', choices=['contiguous', 'paged', 'prefix'],
                        default=['contiguous', 'paged', 'prefix'])
    parser.add_argument('--trace', action='store_true', help='record an opt-in JSONL trace during measured requests')
    parser.add_argument('--prompt-repeat', type=int, default=1,
                        help='repeat the prompt to create a deeper cached prefix')
    parser.add_argument('--prompt', default='Explain what a KV cache is, how it stores keys and values, and why prefix sharing reduces repeated work.')
    parser.add_argument('--output', default='artifacts/kv-comparison.json')
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens, args.trials, args.prompt_repeat) < 1:
        parser.error('counts must be positive')
    args.prompt = ' '.join([args.prompt] * args.prompt_repeat)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'platform': platform.platform(), 'settings': vars(args), 'rss_scope': 'process including load and warmup; excludes dedicated device VRAM', 'trials': [], 'binary_sha256': digest(args.binary), 'model_config_sha256': digest(Path(args.model) / 'config.json'), 'python_version': platform.python_version()}
    for index in range(args.trials):
        modes = args.modes.copy()
        # Rotate order to reduce systematic cold/thermal ordering bias.
        rotation = index % len(modes)
        modes = modes[rotation:] + modes[:rotation]
        for mode in modes:
            row = trial(args, mode, index, output)
            report['trials'].append(row)
            output.write_text(json.dumps(report, indent=2) + '\n')
            print(f'{mode} trial {index}: {row["tokens_per_second"]:.2f} tok/s, {len(row["failures"])} failures', flush=True)
    lines = ['# KV comparison / KV 对照实验', '',
             f'Platform: `{report["platform"]}`. Device: {args.device}; dtype: {args.dtype}.', '',
             'Warmup excluded from timing/allocations. RSS includes model loading and warmup; it is not VRAM. Allocation counts include persistent KV tensors only. Small samples are smoke measurements, not capacity claims.', '',
             '| Mode | Trace | Trial | Success | tok/s | TTFT median ms | Peak RSS MiB | KV allocations | Computed prompt tokens | Prefix hits | Trace drops |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['trials']:
        ttft = row['ttft_ms_median']
        rss = row['peak_process_rss_bytes']
        lines.append(f'| {row["mode"]} | {"on" if args.trace else "off"} | {row["trial"]} | {row["successful"]}/{args.requests} | {row["tokens_per_second"]:.2f} | {round(ttft, 2) if ttft is not None else "n/a"} | {round(rss / 2**20, 2) if rss else "n/a"} | {row["kv_storage_allocations"]} | {row["computed_prompt_tokens"]} | {row["prefix_hit_tokens"]} | {row["trace_events_dropped"]} |')
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    return 1 if any(row['failures'] for row in report['trials']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
