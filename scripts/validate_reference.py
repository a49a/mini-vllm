#!/usr/bin/env python3
"""Verify pinned assets, then run real-tokenizer and real-model Rust oracles.

Only --download permits downloads. A corrupt/missing cache never counts as a
passing skip. Reports record hashes, commands, exit status and complete logs.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'scripts/reference-model.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def verify(path, expected):
    if not path.is_file() or path.stat().st_size != expected['bytes']:
        raise ValueError(f'{path.name}: missing file or wrong size')
    actual = digest(path)
    if actual != expected['sha256']:
        raise ValueError(f'{path.name}: SHA-256 mismatch')
    return actual


class ReferenceRedirects(urllib.request.HTTPRedirectHandler):
    # Python 3.10 does not handle permanent 308 redirects by default.
    http_error_308 = urllib.request.HTTPRedirectHandler.http_error_302

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, 307 if code == 308 else code,
                                        msg, headers, newurl)


def open_asset(request):
    return urllib.request.build_opener(ReferenceRedirects()).open(request, timeout=60)


def ensure_asset(directory, name, expected, url, download):
    path = directory / name
    try:
        return verify(path, expected)
    except ValueError:
        if not download:
            raise
    # Verify the temporary file before replacing a cache entry.
    temporary = path.with_suffix(path.suffix + '.part')
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'mini-vllm-reference-validation'})
        with open_asset(request) as response, temporary.open('wb') as output:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > expected['bytes']:
                    raise ValueError(f'{name}: download exceeds pinned size')
                output.write(chunk)
        actual = verify(temporary, expected)
        temporary.replace(path)
        return actual
    finally:
        temporary.unlink(missing_ok=True)


def run(command, env, timeout):
    start = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=os.name == 'posix')
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        output, _ = process.communicate()
    return dict(command=command, exit_code=process.returncode,
                passed=process.returncode == 0 and not timed_out, timed_out=timed_out,
                elapsed_seconds=time.monotonic() - start, output=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/reference-qwen')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--base-url', default='https://huggingface.co')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/reference-validation.json')
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error('timeout must be positive')
    manifest = json.loads(MANIFEST.read_text())
    args.model = args.model.resolve()
    args.model.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  manifest=manifest, model_directory=str(args.model), verified_files={}, tests=[], passed=False)
    env = os.environ.copy()
    try:
        report['revision'] = run(['git', 'rev-parse', 'HEAD'], env, 20)
        report['working_tree'] = run(['git', 'status', '--porcelain'], env, 20)
        report['rustc'] = run(['rustc', '-Vv'], env, 20)
        report['oracle_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in (
            ROOT / 'crates/mini-vllm-tokenizer/tests/fixtures/chat-reference.json',
            ROOT / 'crates/mini-vllm-model/tests/fixtures/qwen2.5-0.5b-reference.json')}
        for name, expected in manifest['files'].items():
            url = f"{args.base_url.rstrip('/')}/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
            report['verified_files'][name] = ensure_asset(args.model, name, expected, url, args.download)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
        env.update(MINI_VLLM_CHAT_MODEL=str(args.model), MINI_VLLM_REFERENCE_MODEL=str(args.model))
        for package, target, test in [
            ('mini-vllm-tokenizer', 'chat_reference', 'rendered_token_ids_match_transformers'),
            ('mini-vllm-model', 'reference', 'hf_real_qwen_logits_match_all_paths'),
        ]:
            result = run(['cargo', 'test', '--locked', '-p', package, '--test', target,
                          test, '--', '--ignored', '--exact', '--nocapture'], env, args.timeout)
            # An accidentally renamed/removed ignored test must not pass as zero tests.
            result['passed'] = result['passed'] and '1 passed; 0 failed' in result['output']
            report['tests'].append(result)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(f"{test}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
        report['passed'] = all(row['passed'] for row in report['tests'])
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report['error'] = str(error)
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
