"""deploy.py may only ship the app, the model bundle and translation assets: never secrets or data."""
import json
import shutil

import pytest

from tests import fixtures
from tests.deploy_helpers import load_deploy

SECRETS = {'.env.aws': 'AWS_SECRET_ACCESS_KEY=do-not-ship', '.env.deploy': 'X=1', '.hf_token': 'hf_do_not_ship',
           '.env': 'X=1', 'data/raw/HSE_dataset(D4).csv': 'a|b', 'data/raw/ai_job_dataset.csv': 'a,b',
           '.venv/bin/python': '', '.venv/lib/site.py': '', 'data/processed/d4_clean.csv': 'a,b', '.git/config': '',
           '.aws/credentials': 'secret', 'results/final/per_applicant.csv': 'a', 'notebooks/x.ipynb': '{}'}


def fake_project(tmp_path, light_bundle):
    root = tmp_path / 'project'
    deploy = load_deploy()
    for rel in ['deploy/Dockerfile', '.dockerignore', 'requirements/app.txt', 'requirements/app.lock.txt', 'app/streamlit_app.py',
                '.streamlit/config.toml', 'deploy/docker/entrypoint.sh', 'deploy/docker/install-runtime.sh',
                'recommender/__init__.py', 'recommender/bundle.py', 'recommender/skills/extraction.py'] + deploy.HOST_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text('x', encoding='utf-8')
    for rel, text in SECRETS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding='utf-8')
    shutil.copytree(light_bundle, root / 'models' / 'recommender')
    fixtures.build_translation(root / 'models' / 'translation')
    (root / 'deploy' / '.state').mkdir(parents=True, exist_ok=True)
    (root / 'deploy' / '.state' / 'ec2-status.json').write_text('{}', encoding='utf-8')
    return deploy, root


def test_allowlist_excludes_secrets_and_data(tmp_path, light_bundle):
    deploy, root = fake_project(tmp_path, light_bundle)
    allowed = deploy.allowlist(root)
    for rel in allowed:
        assert not rel.startswith(('.env', '.hf_token', 'data/raw/', '.venv', 'data/processed/', '.git/', '.aws/', 'results/', 'notebooks/')), rel
    assert 'deploy/.state/ec2-status.json' not in allowed
    assert 'app/streamlit_app.py' in allowed and 'models/recommender/manifest.json' in allowed
    assert 'models/translation/manifest.json' in allowed
    assert set(deploy.HOST_FILES) <= set(allowed)


def test_package_contains_no_secret_bytes(tmp_path, light_bundle):
    import tarfile
    deploy, root = fake_project(tmp_path, light_bundle)
    deploy.package(root, tmp_path / 'package.tar.gz')
    with tarfile.open(tmp_path / 'package.tar.gz') as archive:
        names = archive.getnames()
        data = b''.join(archive.extractfile(m).read() for m in archive.getmembers() if m.isfile())
    assert not any(n.startswith(('.env', '.hf_token', 'data/', '.venv', 'notebooks/', 'results/')) for n in names)
    assert b'do-not-ship' not in data and b'hf_do_not_ship' not in data


def test_forbidden_paths_are_refused_even_if_listed():
    deploy = load_deploy()
    for rel in ['.env.aws', 'deploy/.env', '.hf_token', 'data/raw/x.csv', '.venv/bin/python', '.venv-1/x',
                'data/processed/d4_clean.csv', '.git/config', '.aws/credentials', 'key.pem']:
        assert deploy.forbidden(rel), rel
    for rel in ['app/streamlit_app.py', 'models/recommender/catalog.csv', '.dockerignore', '.streamlit/config.toml']:
        assert not deploy.forbidden(rel), rel


def test_tampered_bundle_blocks_packaging(tmp_path, light_bundle):
    deploy, root = fake_project(tmp_path, light_bundle)
    with (root / 'models' / 'recommender' / 'catalog.csv').open('a', encoding='utf-8') as stream:
        stream.write('x\n')
    with pytest.raises(ValueError, match='integrity'):
        deploy.allowlist(root)


def test_real_project_allowlist(real_bundle):
    from tests.conftest import ROOT
    if not (ROOT / 'models' / 'translation' / 'manifest.json').exists():
        pytest.skip('Real translation assets not present.')
    deploy = load_deploy()
    allowed = deploy.allowlist(ROOT)
    assert not [rel for rel in allowed if deploy.forbidden(rel)]
    assert json.loads((ROOT / 'models' / 'recommender' / 'manifest.json').read_text(encoding='utf-8'))
