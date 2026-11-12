"""deploy.py may only ship the app, the model bundle and translation assets: never secrets or data."""
import json
import shutil

import pytest

from tests import fixtures
from tests.deploy_helpers import load_deploy

SECRETS = {'.env.aws': 'AWS_SECRET_ACCESS_KEY=do-not-ship', '.env.deploy': 'X=1', '.hf_token': 'hf_do_not_ship',
           '.env': 'X=1', 'datasets/HSE_dataset(D4).csv': 'a|b', 'datasets/ai_job_dataset.csv': 'a,b',
           '.venv/bin/python': '', '.venv/lib/site.py': '', 'clean/d4_clean.csv': 'a,b', '.git/config': '',
           '.aws/credentials': 'secret', 'results/final/per_applicant.csv': 'a', 'notebooks/x.ipynb': '{}'}


def fake_project(tmp_path, light_bundle):
    root = tmp_path / 'project'
    deploy = load_deploy()
    for rel in ['Dockerfile.deploy', '.dockerignore', 'requirements-app.txt', 'requirements-app.lock.txt', 'app.py',
                '.streamlit/config.toml', 'docker/entrypoint.sh', 'docker/install-runtime.sh',
                'recommender/__init__.py', 'recommender/bundle.py', 'm9_skills/extraction.py'] + deploy.HOST_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text('x')
    for rel, text in SECRETS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    shutil.copytree(light_bundle, root / 'artifacts' / 'deployment')
    fixtures.build_translation(root / 'artifacts' / 'translation')
    (root / 'artifacts' / 'ec2-status.json').write_text('{}')
    return deploy, root


def test_allowlist_excludes_secrets_and_data(tmp_path, light_bundle):
    deploy, root = fake_project(tmp_path, light_bundle)
    allowed = deploy.allowlist(root)
    for rel in allowed:
        assert not rel.startswith(('.env', '.hf_token', 'datasets/', '.venv', 'clean/', '.git/', '.aws/', 'results/', 'notebooks/')), rel
    assert 'artifacts/ec2-status.json' not in allowed
    assert 'app.py' in allowed and 'artifacts/deployment/manifest.json' in allowed
    assert 'artifacts/translation/manifest.json' in allowed
    assert set(deploy.HOST_FILES) <= set(allowed)


def test_package_contains_no_secret_bytes(tmp_path, light_bundle):
    import tarfile
    deploy, root = fake_project(tmp_path, light_bundle)
    deploy.package(root, tmp_path / 'package.tar.gz')
    with tarfile.open(tmp_path / 'package.tar.gz') as archive:
        names = archive.getnames()
        data = b''.join(archive.extractfile(m).read() for m in archive.getmembers() if m.isfile())
    assert not any(n.startswith(('.env', '.hf_token', 'datasets', '.venv', 'clean')) for n in names)
    assert b'do-not-ship' not in data and b'hf_do_not_ship' not in data


def test_forbidden_paths_are_refused_even_if_listed():
    deploy = load_deploy()
    for rel in ['.env.aws', 'deploy/.env', '.hf_token', 'datasets/x.csv', '.venv/bin/python', '.venv-1/x',
                'clean/d4_clean.csv', '.git/config', '.aws/credentials', 'key.pem']:
        assert deploy.forbidden(rel), rel
    for rel in ['app.py', 'artifacts/deployment/catalog.csv', '.dockerignore', '.streamlit/config.toml']:
        assert not deploy.forbidden(rel), rel


def test_tampered_bundle_blocks_packaging(tmp_path, light_bundle):
    deploy, root = fake_project(tmp_path, light_bundle)
    with (root / 'artifacts' / 'deployment' / 'catalog.csv').open('a') as stream:
        stream.write('x\n')
    with pytest.raises(ValueError, match='integrity'):
        deploy.allowlist(root)


def test_real_project_allowlist(real_bundle):
    from tests.conftest import ROOT
    if not (ROOT / 'artifacts' / 'translation' / 'manifest.json').exists():
        pytest.skip('Real translation assets not present.')
    deploy = load_deploy()
    allowed = deploy.allowlist(ROOT)
    assert not [rel for rel in allowed if deploy.forbidden(rel)]
    assert json.loads((ROOT / 'artifacts' / 'deployment' / 'manifest.json').read_text())
