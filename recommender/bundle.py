"""Resolve and check a serving bundle without loading ML libraries."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'metadata.json', 'catalog.csv', 'documents.npz', 'tfidf.joblib',
            'job_embeddings.npy', 'job_skills.npy', 'dictionary.json',
            'encoder/model.safetensors', 'encoder/config.json',
            'encoder/modules.json', 'encoder/tokenizer.json'}


def project_path(path, root=ROOT):
    """Resolve relative project inputs independently of the terminal directory."""
    value = Path(path).expanduser()
    return (value if value.is_absolute() else Path(root) / value).resolve()


def resolve_bundle(path=None):
    return project_path(path or os.environ.get('MODEL_BUNDLE') or 'artifacts/deployment')


def check_bundle(path=None, deep=False):
    bundle = resolve_bundle(path)
    manifest_path = bundle / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'Model bundle missing at {bundle}. Restore the complete trusted bundle here, '
            'set MODEL_BUNDLE to its folder, or rebuild with '
            '`python -m recommender.pipeline run --run-dir artifacts/runs/run-01`. '
            'See docs/PROJECT_HANDBOOK.md. Starting Streamlit does not train a model.')
    try:
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict) or not REQUIRED.issubset(manifest):
            raise ValueError('Manifest is missing required model files')
        for relative, expected in manifest.items():
            file = (bundle / relative).resolve()
            if not file.is_relative_to(bundle) or not file.is_file():
                raise ValueError(f'Model artifact missing or unsafe: {relative}')
            if deep:
                with file.open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != expected:
                    raise ValueError(f'Model artifact integrity check failed: {relative}')
        metadata = json.loads((bundle / 'metadata.json').read_text())
        if metadata.get('kind') == 'learned' and 'reranker.joblib' not in manifest:
            raise ValueError('Learned model bundle lacks reranker.joblib')
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f'Invalid model bundle at {bundle}: {error}') from error
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check'])
    parser.add_argument('--path')
    parser.add_argument('--deep', action='store_true', help='Verify every SHA-256 checksum')
    args = parser.parse_args()
    try:
        bundle = check_bundle(args.path, args.deep)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')
    print(f'Model bundle ready: {bundle}')


if __name__ == '__main__':
    main()
