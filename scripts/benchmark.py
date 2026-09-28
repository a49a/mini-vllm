#!/usr/bin/env python3
"""Benchmark a mini-vllm server using terminal usage, not text chunk counts.

Reports successful model tokens, client event TTFT/ITL, wall-clock throughput,
and failures. Idle server history does not enter the scenario denominator.
"""
import argparse
import concurrent.futures
import http.client
import json
import statistics
import time


def consume_sse(response, start, clock=time.perf_counter):
    """Read the complete protocol; DONE without successful finish is an error."""
    ttft = None
    previous = None
    intervals = []
    usage = None
    finish = None
    done = False
    for line in response:
        line = line.strip()
        if not line.startswith(b'data: '):
            continue
        payload = line[6:]
        if payload == b'[DONE]':
            done = True
            break
        event = json.loads(payload)
        if 'error' in event:
            raise RuntimeError(f"SSE error: {event['error']}")
        choices = event.get('choices', [])
        if not choices:
            continue
        choice = choices[0]
        if choice.get('finish_reason') is not None:
            finish = choice['finish_reason']
            usage = event.get('usage')
            continue
        # Every token event counts for timing, including an empty text delta.
        now = clock()
        if previous is None:
            ttft = now - start
        else:
            intervals.append(now - previous)
        previous = now
    if not done or finish not in ('stop', 'length'):
        raise RuntimeError('incomplete or unsuccessful generation stream')
    if not usage or not isinstance(usage.get('completion_tokens'), int):
        raise RuntimeError('terminal usage.completion_tokens is missing')
    return {'ttft': ttft, 'intervals': intervals,
            'tokens': usage['completion_tokens'], 'finish': finish}


def stream_one(host, port, prompt, max_tokens):
    conn = http.client.HTTPConnection(host, port, timeout=600)
    start = time.perf_counter()
    try:
        conn.request('POST', '/v1/completions', body=json.dumps({
            'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0, 'stream': True,
        }), headers={'Content-Type': 'application/json'})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {response.read()[:200]!r}')
        result = consume_sse(response, start)
        result["latency"] = time.perf_counter() - start
        return result
    finally:
        conn.close()


def run_scenario(host, port, total, concurrency, prompt, max_tokens):
    results, failures = [], []
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(stream_one, host, port, prompt, max_tokens) for _ in range(total)]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(str(error))
    elapsed = time.perf_counter() - start
    tokens = sum(r['tokens'] for r in results)
    print(f'\nconcurrency={concurrency}: {len(results)} successful, {len(failures)} failed, {total} attempted')
    print(f'  successful model tokens={tokens}, wall={elapsed:.3f}s, throughput={tokens / elapsed:.2f} tok/s')
    for name, values in [
        ('client token-event TTFT', [r['ttft'] for r in results if r['ttft'] is not None]),
        ('client token-event ITL', [v for r in results for v in r['intervals']]),
    ]:
        if values:
            print(f'  {name}: mean={statistics.mean(values)*1000:.2f}ms median={statistics.median(values)*1000:.2f}ms')
    for error in failures[:3]:
        print(f'  failure: {error}')
    return len(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 4, 16])
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--prompt', default='Explain what a KV cache is and why it matters.')
    args = parser.parse_args()
    if min(args.requests, args.max_tokens, *args.concurrency) < 1 or args.warmup < 0:
        parser.error('counts must be positive; warmup may be zero')
    for _ in range(args.warmup):
        stream_one(args.host, args.port, args.prompt, args.max_tokens)
    failures = sum(run_scenario(args.host, args.port, args.requests, c, args.prompt, args.max_tokens)
                   for c in args.concurrency)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
