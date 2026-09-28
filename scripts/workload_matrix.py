#!/usr/bin/env python3
"""Run mixed-length/load/prefix-reuse scenarios against an existing server.

Use a fresh server per storage mode. Measured prefix hit fraction is reported
separately from the requested fraction of requests reusing a warmed prompt.
"""
import argparse
import concurrent.futures
import json
import math
import hashlib
import time
from pathlib import Path
from benchmark import stream_one
from compare_kv import get_json


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    at = (len(values) - 1) * p / 100
    low, high = math.floor(at), math.ceil(at)
    return values[low] + (values[high] - values[low]) * (at - low)


def scenario(args, concurrency, hit_rate):
    # Nonces isolate scenarios and put cold differences before any full block.
    nonce = hashlib.sha256(f"{args.seed}:{concurrency}:{hit_rate}".encode()).hexdigest()[:32]
    short = f'{nonce} Explain KV cache in simple terms. '
    long = short + ('Explain how keys, values, attention and shared prefixes work. ' * args.long_repeat)
    for prompt in [short, long]:
        stream_one(args.host, args.port, prompt, 1)
    before = get_json(args.port, '/metrics', args.host)
    results, failures = [], []
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i in range(args.requests):
            prompt = short if i % 2 == 0 else long
            if i >= round(args.requests * hit_rate):
                prompt = hashlib.sha256(f"{nonce}:{i}".encode()).hexdigest()[:32] + prompt[len(nonce):]
            limit = args.short_output if i % 2 == 0 else args.long_output
            futures.append(pool.submit(stream_one,args.host,args.port,prompt,limit))
        for future in concurrent.futures.as_completed(futures):
            try: results.append(future.result())
            except Exception as error: failures.append(str(error))
    wall = time.perf_counter() - start
    after = get_json(args.port, '/metrics', args.host)
    hits = after['prefix_cache_hit_tokens'] - before['prefix_cache_hit_tokens']
    computed = after['prompt_tokens_total'] - before['prompt_tokens_total']
    latencies = [r['latency'] * 1000 for r in results]
    ttfts = [r['ttft'] * 1000 for r in results if r['ttft'] is not None]
    return {'concurrency':concurrency,'requested_reuse_fraction':hit_rate,'successful':len(results),
            'failures':failures,'wall_seconds':wall,'tokens_per_second':sum(r['tokens'] for r in results)/wall,
            'latency_ms':{f'p{p}':percentile(latencies,p) for p in [50,95,99]},
            'ttft_ms':{f'p{p}':percentile(ttfts,p) for p in [50,95,99]},
            'itl_ms':{f'p{p}':percentile([v*1000 for r in results for v in r['intervals']],p) for p in [50,95,99]},
            'prefix_hit_tokens':hits,'computed_prompt_tokens':computed,
            'observed_prefix_token_fraction':hits/(hits+computed) if hits+computed else 0,
            'kv_storage_allocations':after['kv_storage_allocations_total']-before['kv_storage_allocations_total']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--requests',type=int,default=32)
    parser.add_argument('--concurrency',type=int,nargs='+',default=[1,2,4,8])
    parser.add_argument('--hit-rates',type=float,nargs='+',default=[0,.5,1])
    parser.add_argument('--long-repeat',type=int,default=16)
    parser.add_argument('--short-output',type=int,default=4)
    parser.add_argument('--long-output',type=int,default=32)
    parser.add_argument('--output',default='artifacts/workload-matrix.json')
    args=parser.parse_args()
    if min(args.requests,args.long_repeat,args.short_output,args.long_output,*args.concurrency)<1 or any(not 0<=v<=1 for v in args.hit_rates):
        parser.error('counts must be positive and hit rates in [0,1]')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report={'settings':vars(args),'percentile_method':'linear interpolation over successful requests; small samples do not establish tail latency','scenarios':[]}
    for concurrency in args.concurrency:
        for hit_rate in args.hit_rates:
            row=scenario(args,concurrency,hit_rate);report['scenarios'].append(row)
            output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    return int(any(row['failures'] for row in report['scenarios']))


if __name__=='__main__':
    raise SystemExit(main())
