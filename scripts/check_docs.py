#!/usr/bin/env python3
"""Check portable local Markdown links and fenced code blocks."""
import re
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for path in [root / 'README.md', *sorted((root / 'docs').glob('*.md'))]:
    text = path.read_text()
    assert text.count('```') % 2 == 0, f'unbalanced fences: {path}'
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if target.startswith(('http:', 'https:', '#')):
            continue
        assert (path.parent / target.split('#')[0]).exists(), (path, target)
print('Documentation links and fences OK')
