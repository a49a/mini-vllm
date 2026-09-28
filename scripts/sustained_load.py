#!/usr/bin/env python3
"""Fixed-arrival-rate mixed workload with bounded client concurrency.

Every scheduled arrival is accounted for: the client drops arrivals when all
workers are busy instead of silently queuing them. Latency from scheduled arrival
includes dispatch delay. RSS is optional local process memory, never VRAM.
"""
import argparse
import concurrent.futures
import datetime
import json
import math
import platform
import subprocess
import threading
import time
from pathlib import Path

from benchmark import stream_one
from compare_kv import distribution, get_json


def summarize(rows, elapsed, duration):
    successful = [r for r in rows if r['status'] == 'success']
    errors = sum(r['status'] == 'error' for r in rows)
    dropped = sum(r['status'] == 'client_dropped' for r in rows)
    tokens = sum(r['tokens'] for r in successful)
    return {
        'scheduled': len(rows), 'successful': len(successful), 'server_or_transport_errors': errors,
        'client_dropped': dropped, 'failure_fraction': (errors + dropped) / len(rows) if rows else 0,
        'offered_seconds': duration, 'wall_seconds_including_drain': elapsed,
        'successful_model_tokens': tokens, 'tokens_per_second_including_drain': tokens / elapsed,
        'latency_ms': distribution([r['latency'] * 1000 for r in successful]),
        'scheduled_latency_ms': distribution([r['scheduled_latency'] * 1000 for r in successful]),
        'scheduled_ttft_ms': distribution([r['scheduled_ttft'] * 1000 for r in successful
                                            if r['scheduled_ttft'] is not None]),
        'dispatch_lag_ms': distribution([r['dispatch_lag'] * 1000 for r in rows if 'dispatch_lag' in r]),
        'itl_ms': distribution([v * 1000 for r in successful for v in r['intervals']]),
    }


def run(args, request=stream_one, metrics=get_json):
    short = 'Explain KV cache in simple terms.'
    long = short + ' Explain keys, values, attention and shared prefixes.' * args.long_repeat
    for i in range(args.warmup):
        request(args.host, args.port, [short, long][i % 2], 1, timeout=args.timeout)
    start = time.perf_counter()
    stopped = threading.Event()
    samples, rows = [], []

    def sample():
        row = {'seconds': time.perf_counter() - start}
        try:
            row['metrics'] = metrics(args.port, '/metrics', args.host)
        except Exception as error:
            row['metrics_error'] = str(error)
        if args.server_pid is not None:
            try:
                row['rss_bytes'] = int(subprocess.check_output(
                    ['ps', '-o', 'rss=', '-p', str(args.server_pid)], text=True,
                    stderr=subprocess.PIPE, timeout=2).strip()) * 1024
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                row['rss_error'] = str(error)
        samples.append(row)

    def monitor():
        while not stopped.is_set():
            sample()
            stopped.wait(args.sample_interval)

    def one(i, due):
        dispatched = time.perf_counter()
        row = {'index': i, 'kind': 'short' if i % 2 == 0 else 'long',
               'scheduled_seconds': due - start, 'dispatch_lag': dispatched - due}
        try:
            result = request(args.host, args.port, short if i % 2 == 0 else long,
                             args.short_output if i % 2 == 0 else args.long_output,
                             timeout=args.timeout)
            row.update(result)
            row.update(status='success', scheduled_latency=time.perf_counter() - due,
                       scheduled_ttft=None if result['ttft'] is None else result['ttft'] + dispatched - due)
        except Exception as error:
            row.update(status='error', error=str(error), scheduled_latency=time.perf_counter() - due)
        return row

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            pending = set()
            for i in range(math.ceil(args.duration * args.arrival_rate)):
                due = start + i / args.arrival_rate
                time.sleep(max(0, due - time.perf_counter()))
                completed = {f for f in pending if f.done()}
                rows.extend(f.result() for f in completed)
                pending.difference_update(completed)
                if len(pending) >= args.concurrency:
                    rows.append({'index': i, 'kind': 'short' if i % 2 == 0 else 'long',
                                 'scheduled_seconds': due - start, 'status': 'client_dropped'})
                else:
                    pending.add(pool.submit(one, i, due))
            rows.extend(f.result() for f in concurrent.futures.as_completed(pending))
            # Keep the full offer window in the denominator, even at low rates.
            time.sleep(max(0, start + args.duration - time.perf_counter()))
        elapsed = time.perf_counter() - start
    finally:
        stopped.set()
        monitor_thread.join()
    sample()
    rows.sort(key=lambda r: r['index'])
    return {'settings': vars(args), 'platform': platform.platform(),
            'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'percentile_method': 'nearest rank; successful requests only, sample count included',
            'rss_scope': 'optional local server process RSS; not device VRAM',
            'summary': summarize(rows, elapsed, args.duration), 'requests': rows, 'samples': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--arrival-rate', type=float, default=1)
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--concurrency', type=int, default=16, help='client in-flight limit; excess arrivals are recorded as dropped')
    parser.add_argument('--long-repeat', type=int, default=16)
    parser.add_argument('--short-output', type=int, default=4)
    parser.add_argument('--long-output', type=int, default=32)
    parser.add_argument('--sample-interval', type=float, default=1)
    parser.add_argument('--server-pid', type=int, help='local server PID for RSS sampling')
    parser.add_argument('--timeout', type=float, default=60, help='HTTP socket operation timeout in seconds')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--output', default='artifacts/sustained-load.json')
    args = parser.parse_args()
    positive = [args.arrival_rate, args.duration, args.concurrency, args.long_repeat,
                args.short_output, args.long_output, args.sample_interval, args.timeout]
    if any(not math.isfinite(v) or v <= 0 for v in positive) or args.warmup < 0:
        parser.error('counts, rates and times must be finite and positive; warmup may be zero')
    if args.server_pid is not None and (args.server_pid <= 0 or args.host not in ('127.0.0.1', 'localhost')):
        parser.error('--server-pid requires a positive local PID and a loopback host')
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))
    return int(report['summary']['failure_fraction'] > 0 or any('metrics_error' in s or 'rss_error' in s for s in report['samples']))


if __name__ == '__main__':
    raise SystemExit(main())
