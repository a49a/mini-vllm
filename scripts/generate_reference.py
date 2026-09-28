#!/usr/bin/env python3
"""Export independent HF Qwen2 CPU/F32 logits and reproducible model provenance.

Use --tiny for the small offline CI fixture, or --model for real local weights.
No remote code is executed and no model is downloaded by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--tiny', action='store_true')
    mode.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(1234)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tokens = [1, 4, 7, 12, 19, 23, 31, 40]
    if args.tiny:
        directory = args.output.parent / 'tiny-qwen2'
        cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128,
                         tie_word_embeddings=True, eos_token_id=2, bos_token_id=1,
                         rope_theta=10000.0, attention_dropout=0.0)
        cfg._attn_implementation = 'eager'
        model = Qwen2ForCausalLM(cfg).float().eval()
        model.save_pretrained(directory, safe_serialization=True)
    else:
        directory = args.model
        model = AutoModelForCausalLM.from_pretrained(
            directory, torch_dtype=torch.float32, attn_implementation='eager',
            local_files_only=True, trust_remote_code=False).eval()
    with torch.no_grad():
        logits = model(torch.tensor([tokens]), use_cache=False).logits[0].float()
    rows = []
    for row in logits:
        # Tiny fixture covers every logit. Real fixture covers a fixed vocabulary
        # grid plus each row's top 32, keeping checked-in data small.
        indices = set(range(row.numel())) if args.tiny else set(range(0, row.numel(), max(1, row.numel() // 128)))
        indices.update(torch.topk(row, min(32, row.numel())).indices.tolist())
        rows.append({'argmax': int(row.argmax()),
                     'values': [[i, float(row[i])] for i in sorted(indices)]})
    result = {'format_version': 1, 'source': 'Hugging Face Qwen2 eager CPU F32',
              'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
              'seed': 1234, 'token_ids': tokens, 'atol': 0.002,
              'config_sha256': sha256(directory / 'config.json'),
              'weights_sha256': {p.name: sha256(p) for p in sorted(directory.glob('*.safetensors'))},
              'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {args.output}: {len(rows)} positions, independently computed by Transformers')


if __name__ == '__main__':
    main()
