"""One feature/scoring implementation shared by evaluation and serving."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .shared import clean_title, region_key, text, zscore, evidence_vectors, coverage

FEATURE_NAMES = ['tfidf_cosine', 'semantic_cosine', 'tfidf_z', 'semantic_z', 'hybrid_score',
                 'skill_coverage', 'same_region', 'cv_skill_evidence', 'job_skill_evidence', 'exact_title']


def normalize(value):
    return ' '.join(re.sub(r'[^a-zа-яё0-9\s]', ' ', str(value).lower()).split())


def profile(title, history, education, skills, region):
    if not str(title).strip() and not str(history).strip():
        raise ValueError('Enter a desired title or work history.')
    if any(len(str(x)) > 50000 for x in [title, history, education, skills]):
        raise ValueError('Each input must contain fewer than 50,000 characters.')
    return pd.DataFrame([{'positionName':str(title).strip(),
                          'cv_text':normalize(' '.join([title, history, education])),
                          'cv_skills':normalize(skills), 'localityName':str(region).strip()}])


def online_skill_vector(extractor, history, skills):
    record = extractor.extract_fields({'workExperienceList':history, 'skills_cv':skills}, ['skills_cv'])
    return evidence_vectors([{'m9_cv_evidence':record['evidence']}], extractor.dictionary, 'cv')


def candidates(queries, embeddings, documents, job_embeddings, profiles, jobs, cv_skills, job_skills,
               use_region=True, keep=300, batch_size=100):
    """Full-catalog standardization, followed by region priority and top-300 retrieval."""
    rows_out, features_out = [], []
    job_regions = region_key(jobs.regionName).to_numpy()
    query_regions = region_key(profiles.localityName).to_numpy()
    job_titles = clean_title(jobs.vacancyName).to_numpy()
    cv_titles = clean_title(profiles.positionName).to_numpy()
    for start in range(0, queries.shape[0], batch_size):
        end = start + batch_size
        lexical = (queries[start:end] @ documents.T).toarray()
        semantic = embeddings[start:end] @ job_embeddings.T
        wz, sz = zscore(lexical), zscore(semantic)
        hybrid = 0.5*wz + 0.5*sz
        same = query_regions[start:end, None] == job_regions[None, :]
        ranked_scores = hybrid.copy()
        if use_region:
            ranked_scores[~same] -= 1_000_000
        rows = np.argsort(-ranked_scores, axis=1)[:, :keep]
        gather = lambda scores: np.take_along_axis(scores, rows, axis=1)
        sk = coverage(cv_skills[start:end], job_skills, rows)
        cvhas = np.broadcast_to((cv_skills[start:end].sum(axis=1)>0)[:,None], rows.shape)
        jobhas = job_skills.sum(axis=1)[rows]>0
        exact = cv_titles[start:end,None] == job_titles[rows]
        features = np.stack([gather(lexical), gather(semantic), gather(wz), gather(sz), gather(hybrid),
                             sk, gather(same), cvhas, jobhas, exact],axis=-1).astype(np.float32)
        rows_out.append(rows);features_out.append(features)
    return np.vstack(rows_out), np.vstack(features_out)


def rerank(rows, features, kind, skill_weight=0.0, estimator=None, use_region=True):
    if kind == 'M8':
        return rows.copy()
    if kind == 'M5':
        scores = features[:,:,0]
    elif kind == 'skill':
        if skill_weight == 0:
            return rows.copy()
        scores = features[:,:,4] + skill_weight*features[:,:,5]
    elif kind == 'learned':
        if estimator is None:
            raise ValueError('Missing trained reranker')
        scores = estimator.predict_proba(features.reshape(-1, features.shape[-1]))[:,1].reshape(rows.shape)
    else:
        raise ValueError('Unknown scoring kind')
    out=[]
    for i, row in enumerate(rows):
        same=features[i,:,6].astype(bool)
        keys=(np.arange(len(row)), -scores[i], ~same) if use_region else (np.arange(len(row)), -scores[i])
        out.append(row[np.lexsort(keys)])
    return np.asarray(out)
