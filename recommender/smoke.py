"""Release smoke test: load the bundle with the production loader and rank one sample profile.

    python -m recommender.smoke            # exit 0 when recommendations come back, 1 otherwise

Used by the EC2 release script before a new image replaces the running one. It reads the
bundle only; nothing is written and the model is not changed.
"""
import sys
from .bundle import check_bundle

SAMPLE = {'title': 'Бухгалтер', 'history': 'Ведение бухгалтерского учета, 1С, Excel, налоговая отчетность',
          'education': '', 'skills': '1С; Excel', 'region': ''}


def run(bundle=None, limit=5):
    from .serving import Recommender
    model = Recommender(check_bundle(bundle))
    result, info = model.recommend(SAMPLE['title'], SAMPLE['history'], SAMPLE['education'],
                                   SAMPLE['skills'], SAMPLE['region'], limit)
    if len(result) != limit or not result.idVacancy.is_unique:
        raise RuntimeError(f'expected {limit} distinct recommendations, got {len(result)}')
    print(f"Smoke test passed: {len(result)} recommendations from {info['selected_model']}", flush=True)
    return len(result)


def main():
    try:
        run()
    except Exception as error:  # report any failure as a non-zero exit for the release script
        print('Smoke test FAILED:', error, file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
