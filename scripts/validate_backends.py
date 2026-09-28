#!/usr/bin/env python3
"""Run explicit backend/dtype numerical and lifecycle tests; never silently skip.

Missing hardware, build failures and unsupported operations remain failures in
the JSON report. Use --devices cpu for portable validation.
"""
import argparse
import datetime
import hashlib
import shutil
import time
import json
import os
import platform
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices',nargs='+',choices=['cpu','metal','cuda'],default=['cpu'])
    parser.add_argument('--dtypes',nargs='+',choices=['f32','f16','bf16'],default=['f32','f16'])
    parser.add_argument('--output',default='artifacts/backend-validation.json')
    args=parser.parse_args()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    def probe(command):
        try:
            p = subprocess.run(command, text=True, capture_output=True, timeout=20)
            return {'command': command, 'exit_code': p.returncode, 'output': p.stdout + p.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            return {'command': command, 'exit_code': -1, 'output': str(error)}
    inventory = []
    if platform.system() == 'Darwin':
        inventory.append(probe(['system_profiler', 'SPDisplaysDataType']))
    if shutil.which('nvidia-smi'):
        inventory.append(probe(['nvidia-smi']))
    else:
        inventory.append({'command': ['nvidia-smi'], 'exit_code': -1, 'output': 'nvidia-smi not found'})
    fixture = Path('crates/mini-vllm-model/tests/fixtures/tiny-qwen2/model.safetensors')
    report={'platform':platform.platform(), 'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'revision': probe(['git', 'rev-parse', 'HEAD']), 'working_tree': probe(['git', 'status', '--porcelain']),
            'rustc': probe(['rustc', '-Vv']), 'inventory': inventory,
            'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
            'test_source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                Path('crates/mini-vllm-model/tests/reference.rs'), Path('crates/mini-vllm-engine/tests/engine.rs'))},
            'results': []}
    output.write_text(json.dumps(report, indent=2) + '\n')
    for device in args.devices:
        for dtype in args.dtypes:
            for package,target,test in [('mini-vllm-model','reference','backend_dtype_reference'),('mini-vllm-engine','engine','backend_dtype_lifecycle')]:
                command=['cargo','test','--locked','-p',package,'--test',target,test]
                if device!='cpu': command+=['--features',f'mini-vllm-model/{device}' if package.endswith('engine') else device]
                command+=['--','--ignored','--nocapture']
                env=os.environ.copy();env.update(MINI_VLLM_TEST_DEVICE=device,MINI_VLLM_TEST_DTYPE=dtype)
                started = time.monotonic()
                try:
                    result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
                    code,log=result.returncode,result.stdout
                except (OSError, subprocess.TimeoutExpired) as error:
                    code,log=-1,str(error)
                entry={'device':device,'dtype':dtype,'test':test,'command':command,'passed':code==0,'exit_code':code,'output':log,'elapsed_seconds':time.monotonic()-started}
                report['results'].append(entry);output.write_text(json.dumps(report,indent=2)+'\n')
                print(f'{device}/{dtype} {test}: {"PASS" if code==0 else "FAIL"}',flush=True)
    return int(any(not row['passed'] for row in report['results']))


if __name__=='__main__':
    raise SystemExit(main())
