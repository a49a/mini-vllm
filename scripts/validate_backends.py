#!/usr/bin/env python3
"""Run explicit backend/dtype numerical and lifecycle tests; never silently skip.

Missing hardware, build failures and unsupported operations remain failures in
the JSON report. Use --devices cpu for portable validation.
"""
import argparse
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
    report={'platform':platform.platform(),'results':[]}
    for device in args.devices:
        for dtype in args.dtypes:
            for package,target,test in [('mini-vllm-model','reference','backend_dtype_reference'),('mini-vllm-engine','engine','backend_dtype_lifecycle')]:
                command=['cargo','test','--locked','-p',package,'--test',target,test]
                if device!='cpu': command+=['--features',f'mini-vllm-model/{device}' if package.endswith('engine') else device]
                command+=['--','--ignored','--nocapture']
                env=os.environ.copy();env.update(MINI_VLLM_TEST_DEVICE=device,MINI_VLLM_TEST_DTYPE=dtype)
                try:
                    result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
                    code,log=result.returncode,result.stdout
                except subprocess.TimeoutExpired as error:
                    code,log=-1,str(error)
                entry={'device':device,'dtype':dtype,'test':test,'command':command,'passed':code==0,'exit_code':code,'output':log}
                report['results'].append(entry);output.write_text(json.dumps(report,indent=2)+'\n')
                print(f'{device}/{dtype} {test}: {"PASS" if code==0 else "FAIL"}',flush=True)
    return int(any(not row['passed'] for row in report['results']))


if __name__=='__main__':
    raise SystemExit(main())
