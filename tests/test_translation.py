"""Translation asset check: required files present and every checksum matches."""
import json

import pytest

from recommender.translation import check_translation
from tests.conftest import ROOT


def test_fixture_translation_passes(translation_dir):
    assert check_translation(translation_dir, deep=True) == translation_dir.resolve()


def test_tampered_translation_file_fails(translation_dir):
    (translation_dir / 'ru-en' / 'vocab.json').write_text('changed', encoding='utf-8')
    check_translation(translation_dir, deep=False)
    with pytest.raises(ValueError, match='checksum mismatch: ru-en/vocab.json'):
        check_translation(translation_dir, deep=True)


def test_missing_required_translation_file_fails(translation_dir):
    manifest = json.loads((translation_dir / 'manifest.json').read_text(encoding='utf-8'))
    del manifest['en-ru/model.safetensors']
    (translation_dir / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='lacks required files'):
        check_translation(translation_dir)


def test_missing_translation_folder_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match='translation assets missing'):
        check_translation(tmp_path / 'absent')


def test_real_translation_assets():
    path = ROOT / 'artifacts' / 'translation'
    if not (path / 'manifest.json').exists():
        pytest.skip('Real translation assets not present (git-ignored).')
    assert check_translation(path, deep=True) == path.resolve()
