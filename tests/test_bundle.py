"""Model bundle integrity: the manifest SHA-256 check must reject any changed or missing file."""
import json
import shutil

import pytest

from recommender.bundle import check_bundle


def copy(bundle, tmp_path):
    target = tmp_path / 'bundle'
    shutil.copytree(bundle, target)
    return target


def test_fixture_bundle_passes_deep_check(light_bundle):
    assert check_bundle(light_bundle, deep=True) == light_bundle.resolve()


def test_tampered_file_fails_deep_check(light_bundle, tmp_path):
    bundle = copy(light_bundle, tmp_path)
    with (bundle / 'catalog.csv').open('a') as stream:
        stream.write('tampered,row\n')
    check_bundle(bundle, deep=False)  # shallow check only looks at presence
    with pytest.raises(ValueError, match='integrity check failed: catalog.csv'):
        check_bundle(bundle, deep=True)


def test_missing_file_fails(light_bundle, tmp_path):
    bundle = copy(light_bundle, tmp_path)
    (bundle / 'job_embeddings.npy').unlink()
    with pytest.raises(ValueError, match='missing or unsafe: job_embeddings.npy'):
        check_bundle(bundle)


def test_manifest_cannot_point_outside_bundle(light_bundle, tmp_path):
    bundle = copy(light_bundle, tmp_path)
    (tmp_path / 'outside.txt').write_text('x')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    manifest['../outside.txt'] = '0' * 64
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unsafe'):
        check_bundle(bundle)


def test_manifest_missing_required_entry(light_bundle, tmp_path):
    bundle = copy(light_bundle, tmp_path)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    del manifest['tfidf.joblib']
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='missing required'):
        check_bundle(bundle)


def test_missing_bundle_gives_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match='Model bundle missing'):
        check_bundle(tmp_path / 'nothing-here')


def test_real_bundle_deep_check(real_bundle):
    assert check_bundle(real_bundle, deep=True) == real_bundle.resolve()
