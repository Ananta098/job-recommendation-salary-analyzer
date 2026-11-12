"""Serving smoke test: load a bundle through the production loader and rank a sample profile."""
import pytest

from tests.conftest import SERVING_STACK, has

pytestmark = pytest.mark.skipif(not has(*SERVING_STACK),
                                reason='torch / sentence-transformers not installed: run  make venv')


def test_fixture_bundle_returns_top_n(serving_bundle):
    from recommender.serving import Recommender
    model = Recommender(serving_bundle)
    result, info = model.recommend('Бухгалтер', 'ведение бухгалтерского учета 1с', '', '1с; excel',
                                   'Москва', limit=5)
    assert len(result) == 5
    assert list(result['rank']) == [1, 2, 3, 4, 5]
    assert result.idVacancy.is_unique
    assert info['selected_model'] == 'M8' and info['known_region'] is True
    # Region priority: vacancies in the chosen region come first.
    moscow = (result.regionName == 'Москва').tolist()
    assert moscow == sorted(moscow, reverse=True)


def test_limit_is_validated(serving_bundle):
    from recommender.serving import Recommender
    model = Recommender(serving_bundle)
    with pytest.raises(ValueError):
        model.recommend('Бухгалтер', limit=0)
    with pytest.raises(ValueError):
        model.recommend('', '', limit=5)


def test_smoke_module_passes_on_fixture(serving_bundle):
    from recommender import smoke
    assert smoke.run(serving_bundle, limit=3) == 3


def test_real_bundle_smoke(real_bundle):
    from recommender import smoke
    assert smoke.run(real_bundle, limit=10) == 10
