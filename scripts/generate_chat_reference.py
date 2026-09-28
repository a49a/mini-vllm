#!/usr/bin/env python3
"""Generate offline chat text/token oracles from a local Hugging Face tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
import transformers
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    conversations = {
        'default_system': [('user', '你好，Rust!')],
        'explicit_system': [('system', 'Be concise.'), ('user', 'hi')],
        'multi_turn': [('user', 'one'), ('assistant', 'two'), ('user', 'three')],
        'special_tokens': [('system', ''), ('user', '<|im_start|> and <|im_end|>\n café 🙂')],
        'late_system': [('user', ''), ('system', 'updated'), ('assistant', '')],
    }
    cases = []
    for name, messages in conversations.items():
        messages = [dict(role=r, content=c) for r, c in messages]
        cases.append(dict(name=name, messages=messages,
                          rendered=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                          token_ids=tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)))
    result = dict(source='Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0)', transformers_version=transformers.__version__,
                  tokenizer_sha256=hashlib.sha256((args.model / 'tokenizer.json').read_bytes()).hexdigest(), cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
