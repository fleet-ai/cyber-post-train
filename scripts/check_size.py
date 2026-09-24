"""Fail if the tracked repository is not a small, plain-text checkout."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 10_000
result = subprocess.run(
    ['git', '-C', str(ROOT), 'ls-files', '-z'], check=True, capture_output=True
)
paths = [name.decode('utf-8') for name in result.stdout.split(b'\0') if name]
total = 0
for name in paths:
    path = ROOT / name
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f'not a regular file: {name}')
    data = path.read_bytes()
    if b'\0' in data:
        raise SystemExit(f'binary tracked file: {name}')
    try:
        data.decode('utf-8')
    except UnicodeDecodeError:
        raise SystemExit(f'non-UTF-8 tracked file: {name}') from None
    total += len(data.splitlines())
print(f'{len(paths)} tracked files, {total} physical text lines (limit: <{LIMIT})')
if total >= LIMIT:
    sys.exit(1)
