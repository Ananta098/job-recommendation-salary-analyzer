"""Professor's research question: does skill-aware matching help when CV and vacancy use different words?

Recomputes the M8 and M9 + region rankings for the test applicants from the saved inputs, checks that
they reproduce results/final/metrics_grouped.csv, then compares the two models within subgroups of
applicants defined by how much vocabulary their CV shares with the vacancies they were invited to.
No model is retrained and the test set is not used for any choice: the M9 skill weight is the one
selected on the development set (results/final/development_skill_weights.csv).

    python -m recommender.terminology --skills DIR     # DIR holds cv_skills.jsonl and vacancy_skills.jsonl

Writes results/final/terminology_subgroups.csv (aggregates only, no applicant-level rows).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .bundle import ROOT, project_path
from .shared import MODEL, split, text, gold_for, evaluate, evidence_vectors, clean_title, region_key
from .core import candidates, rerank
from .analysis import paired, holm

WORD = re.compile(r'[a-zа-яё0-9]{3,}')


def words(value):
    return set(WORD.findall(str(value).lower()))


def overlap_table(data, test_ids, profiles):
    """Per test applicant: share of each invited vacancy's distinct words found in the CV (mean over
    invited vacancies), and whether the CV title shares any word with an invited vacancy title."""
    invited = data[(data.is_positive == 1) & data.idCv.isin(test_ids)]
    rows = []
    for cv_id, part in invited.groupby('idCv'):
        profile = profiles.loc[cv_id]
        cv_words = words(f"{profile.positionName} {profile.cv_text} {profile.cv_skills}")
        title_words = words(profile.positionName)
        shares, title_match = [], False
        for _, job in part.drop_duplicates('idVacancy').iterrows():
            job_words = words(f"{job.vacancyName} {job.job_text} {job.job_skills}")
            if job_words:
                shares.append(len(job_words & cv_words) / len(job_words))
            title_match |= bool(title_words & words(job.vacancyName))
        rows.append({'idCv': cv_id, 'overlap': float(np.mean(shares)) if shares else 0.0, 'title_match': title_match})
    return pd.DataFrame(rows).set_index('idCv').loc[test_ids]


def run(root, skills_dir, tolerance=1e-6):
    from recommender.skills.extraction import Extractor
    from recommender.skills.compatibility import align_sidecar
    out = root / 'results/final'
    data = pd.read_csv(root / 'data/processed/d4_clean.csv', encoding='utf-8-sig', keep_default_na=False, low_memory=False)
    data['is_positive'] = data.is_positive.astype(int)
    jobs = data.drop_duplicates('idVacancy').reset_index(drop=True)
    groups = pd.factorize(clean_title(jobs.vacancyName) + ' | ' + region_key(jobs.regionName))[0]
    rowmap = pd.Series(jobs.index, index=jobs.idVacancy)
    _, _, test_ids = split(data)
    profiles = data.drop_duplicates('idCv').set_index('idCv')
    test = profiles.loc[test_ids]

    extractor = Extractor(root / 'recommender/skills/dictionary.v2.json')
    cv_skills = evidence_vectors(align_sidecar(list(test_ids), Path(skills_dir) / 'cv_skills.jsonl', 'cv', extractor.fingerprint),
                                 extractor.dictionary, 'cv')
    job_skills = evidence_vectors(align_sidecar(jobs.idVacancy.tolist(), Path(skills_dir) / 'vacancy_skills.jsonl', 'vacancy',
                                                extractor.fingerprint), extractor.dictionary, 'vacancy')
    tfidf = TfidfVectorizer(min_df=3, max_features=120_000, sublinear_tf=True, dtype=np.float32)
    documents = tfidf.fit_transform(text(jobs, 'vacancy'))
    job_emb = np.load(root / f'data/processed/sbert_jobs_{MODEL}.npy').astype(np.float32)
    test_emb = np.load(root / f'data/processed/sbert_test_{MODEL}.npy').astype(np.float32)

    rows, features = candidates(tfidf.transform(text(test, 'cv')), test_emb, documents, job_emb, test, jobs, cv_skills, job_skills)
    weights = pd.read_csv(out / 'development_skill_weights.csv')
    m9 = weights[weights.model == 'M9 + region']
    weight = float(m9.sort_values(['grouped_Recall@10', 'skill_weight'], ascending=[False, True]).iloc[0].skill_weight)
    gold = gold_for(data, test_ids, rowmap, groups)
    recall = {'M8': evaluate(rerank(rows, features, 'M8'), *gold, groups)['grouped']['Recall@10'],
              'M9 + region': evaluate(rerank(rows, features, 'skill', weight, use_region=True), *gold, groups)['grouped']['Recall@10']}

    published = pd.read_csv(out / 'metrics_grouped.csv', index_col=0)
    for name, values in recall.items():
        gap = abs(values.mean() - published.loc[name, 'Recall@10'])
        print(f'{name}: recomputed grouped Recall@10 {values.mean():.6f}, published {published.loc[name, "Recall@10"]:.6f}')
        if gap > tolerance:
            raise ValueError(f'{name} does not reproduce the published result (difference {gap:.2e}); not writing subgroups')

    table = overlap_table(data, test_ids, profiles)
    table['usable_skills'] = (cv_skills.sum(axis=1) > 0)
    low, high = np.quantile(table.overlap, [1 / 3, 2 / 3])
    table['band'] = np.where(table.overlap <= low, 'low', np.where(table.overlap <= high, 'medium', 'high'))
    subsets = [('All test applicants', np.ones(len(table), bool)),
               (f'Low vocabulary overlap (<= {low:.2f})', (table.band == 'low').to_numpy()),
               (f'Medium overlap ({low:.2f}-{high:.2f})', (table.band == 'medium').to_numpy()),
               (f'High overlap (> {high:.2f})', (table.band == 'high').to_numpy()),
               ('CV title shares no word with invited vacancy titles', (~table.title_match).to_numpy()),
               ('Low overlap AND usable CV skills', ((table.band == 'low') & table.usable_skills).to_numpy())]
    result = []
    for name, mask in subsets:
        delta = recall['M9 + region'][mask] - recall['M8'][mask]
        lower, upper, p = paired(delta)
        result.append({'subgroup': name, 'applicants': int(mask.sum()),
                       'share_with_usable_cv_skills': float(table.usable_skills[mask].mean()),
                       'M8_grouped_Recall@10': float(recall['M8'][mask].mean()),
                       'M9_region_grouped_Recall@10': float(recall['M9 + region'][mask].mean()),
                       'difference_M9_minus_M8': float(delta.mean()), 'ci95_lower': lower, 'ci95_upper': upper,
                       'paired_permutation_p': p, 'improved': int((delta > 0).sum()), 'worse': int((delta < 0).sum())})
    result = pd.DataFrame(result)
    result['holm_adjusted_p'] = holm(result.paired_permutation_p.to_numpy())
    result['m9_skill_weight'] = weight
    result.to_csv(out / 'terminology_subgroups.csv', index=False)
    print(result.to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--skills', type=Path, default=Path('recommender/skills/outputs/v2_validation_run'),
                        help='Folder with cv_skills.jsonl and vacancy_skills.jsonl (make bundle regenerates them)')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                        help='Allowed gap to the published Recall@10 (keep the default with the pinned environment)')
    args = parser.parse_args()
    root = project_path(args.root)
    run(root, project_path(args.skills, root), args.tolerance)


if __name__ == '__main__':
    main()
