"""Shared fixtures. Nothing here needs the raw dataset, the real bundle or AWS."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests import fixtures  # noqa: E402


def has(*modules):
    return all(importlib.util.find_spec(m) is not None for m in modules)


ML_STACK = ('numpy', 'pandas', 'scipy', 'sklearn', 'joblib')
SERVING_STACK = ML_STACK + ('torch', 'transformers', 'sentence_transformers')


@pytest.fixture(scope='session')
def light_bundle(tmp_path_factory):
    """Integrity-check bundle without an encoder model (no torch needed)."""
    if not has(*ML_STACK):
        pytest.skip('numpy/pandas/scipy/scikit-learn not installed: run  make venv')
    return fixtures.build_bundle(tmp_path_factory.mktemp('light') / 'deployment', with_encoder=False)


@pytest.fixture(scope='session')
def serving_bundle(tmp_path_factory):
    """Loadable bundle with a tiny random encoder (needs torch and sentence-transformers)."""
    if not has(*SERVING_STACK):
        pytest.skip('torch / transformers / sentence-transformers not installed: run  make venv')
    return fixtures.build_bundle(tmp_path_factory.mktemp('serving') / 'deployment', with_encoder=True)


@pytest.fixture()
def translation_dir(tmp_path):
    return fixtures.build_translation(tmp_path / 'translation')


@pytest.fixture()
def real_bundle():
    """The private bundle, when this computer has it. Git and CI do not."""
    path = ROOT / 'models' / 'recommender'
    if not (path / 'manifest.json').exists():
        pytest.skip('Real model bundle not present (models/ is git-ignored); fixture tests cover the contract.')
    return path
