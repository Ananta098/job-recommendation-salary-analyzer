"""Shared normalization, split, skill evidence and ranking metrics. No experiment CLI."""
import hashlib
import numpy as np
import pandas as pd

MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
ENCODER_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
KS = (5, 10, 20)
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def clean_title(s):
    return s.str.lower().str.replace(r'[^a-zа-яё0-9\s]', ' ', regex=True).str.split().str.join(' ')

def region_key(s):
    return (s.astype(str).str.split(',').str[0].str.lower().str.replace('-', ' ')
            .str.replace(r'^(город|г)\s+', '', regex=True).str.split().str.join(' '))

def text(frame, side):
    title, body, skills = ('positionName', 'cv_text', 'cv_skills') if side == 'cv' else ('vacancyName', 'job_text', 'job_skills')
    return ((clean_title(frame[title]) + ' ') * 3 + frame[body] + ' ' + frame[skills]).str.strip()

def gold_for(frame, ids, row_map, groups):
    invited = frame[(frame.is_positive == 1) & frame.idCv.isin(ids)].groupby('idCv').idVacancy.apply(list)
    strict = [[int(row_map[v]) for v in invited[c]] for c in ids]
    return strict, [sorted(set(groups[r] for r in rows)) for rows in strict]

def grouped_rank(rows, groups):
    kept, seen = [], set()
    for r in rows:
        g = int(groups[r])
        if g not in seen:
            seen.add(g)
            kept.append(g)
        if len(kept) == max(KS):
            break
    return kept

def metric_rows(ranked, gold):
    values = {f'{m}@{k}': [] for k in KS for m in ('Recall', 'Precision', 'NDCG')}
    for rows, answers in zip(ranked, gold):
        answers = set(answers)
        if not answers:
            raise ValueError('Evaluation applicant has no positive label')
        for k in KS:
            hits = np.array([x in answers for x in rows[:k]], dtype=float)
            values[f'Recall@{k}'].append(float(hits.sum() / len(answers)))
            values[f'Precision@{k}'].append(float(hits.sum() / k))
            ideal = np.sum(1 / np.log2(np.arange(min(k, len(answers))) + 2))
            values[f'NDCG@{k}'].append(float(np.sum(hits / np.log2(np.arange(len(hits)) + 2)) / ideal))
    return {name: np.asarray(v) for name, v in values.items()}

def evaluate(rows, strict_gold, grouped_gold, groups):
    return {'strict': metric_rows(rows, strict_gold),
            'grouped': metric_rows([grouped_rank(r, groups) for r in rows], grouped_gold)}

def zscore(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)

def evidence_vectors(records, dictionary, side):
    """Soft traits, negation and non-requirements never affect the skill score.

    CV title/education mentions are excluded: they do not establish competence.
    Vacancy title/additional-requirement mentions are excluded conservatively.
    Duplicate evidence uses max weight per canonical skill, never mention counts.
    """
    skills = [s['id'] for s in dictionary['skills'] if s['domain'] != 'soft']
    index = {sid: i for i, sid in enumerate(skills)}
    vectors = np.zeros((len(records), len(skills)), dtype=np.float32)
    allowed = ({'workExperienceList', 'skills_cv', 'hardSkills_cv'} if side == 'cv'
               else {'responsibilities', 'positionRequirements', 'skills_vacancy', 'hardSkills_vacancy'})
    for i, record in enumerate(records):
        for e in record[f'm9_{side}_evidence']:
            sid = e['skill_id']
            if sid not in index or e['assertion'] != 'mentioned' or e['field'] not in allowed:
                continue
            requirement = e['requirement']
            if requirement == 'not_required':
                continue
            weight = 1.0 if side == 'cv' else {'required': 2.0, 'optional': 0.5, 'unspecified': 1.0}[requirement]
            vectors[i, index[sid]] = max(vectors[i, index[sid]], weight)
    return vectors

def coverage(cv_vectors, vacancy_vectors, rows):
    out = np.zeros(rows.shape, dtype=np.float32)
    for i, candidates in enumerate(rows):
        weights = vacancy_vectors[candidates]
        denominator = weights.sum(axis=1)
        np.divide(weights @ cv_vectors[i], denominator, out=out[i], where=denominator > 0)
    return out

def split(data):
    eligible=np.sort(data.loc[data.in_eval_set.astype(int)==1,'idCv'].unique())
    shuffled=np.random.default_rng(42).permutation(eligible)
    test=np.sort(shuffled[:int(len(shuffled)*.2)])
    rest=np.sort(shuffled[int(len(shuffled)*.2):])
    dev=np.sort(np.random.default_rng(42).choice(rest,min(1000,len(rest)),replace=False))
    train=np.sort(np.setdiff1d(rest,dev))
    assert not (set(train)&set(dev) or set(train)&set(test) or set(dev)&set(test))
    return train,dev,test
