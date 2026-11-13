"""Skills must reach the model once, as the separate cv_skills field, as in the evaluated data."""
import pytest

from recommender.profile_form import build_profile
from tests.conftest import ML_STACK, has


def test_build_profile_has_no_skill_text():
    title, history, qualification = build_profile('Accountant', '', '1–3 years', 'College diploma',
                                                  ['Payroll and tax reporting'], 'Worked at a bank.')
    assert title == 'Accountant' and qualification == 'College diploma'
    assert 'Skills' not in history and 'Excel' not in history
    assert 'Payroll and tax reporting' in history and 'Worked at a bank.' in history


def test_build_profile_requires_a_title():
    with pytest.raises(ValueError):
        build_profile('Choose a job title', '', 'Not specified', 'Not specified', [], '')


@pytest.mark.skipif(not has(*ML_STACK), reason='pandas not installed')
def test_skills_only_in_cv_skills_field():
    from recommender.core import profile
    row = profile('Бухгалтер', 'Ведение учета', '', 'Excel; 1С', 'Москва').iloc[0]
    assert 'excel' in row.cv_skills and 'excel' not in row.cv_text
