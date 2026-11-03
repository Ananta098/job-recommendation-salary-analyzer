"""Validated selected-model bundle loader and online inference."""
import hashlib,json,threading
from pathlib import Path
import torch
torch.set_num_threads(4)
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from m9_skills.extraction import Extractor
from .bundle import check_bundle
from .shared import region_key,grouped_rank,clean_title,text
from .core import candidates,rerank,profile,online_skill_vector,FEATURE_NAMES


class Recommender:
    def __init__(self,bundle):
        self.bundle=check_bundle(bundle, deep=True)
        self.meta=json.loads((self.bundle/'metadata.json').read_text())
        if self.meta['format_version']!=1 or self.meta['features']!=FEATURE_NAMES:raise ValueError('Unsupported bundle contract')
        self.catalog=pd.read_csv(self.bundle/'catalog.csv',keep_default_na=False)
        ordered_hash=hashlib.sha256('\n'.join(self.catalog.idVacancy).encode()).hexdigest()
        if ordered_hash!=self.meta['catalog_ids_sha256']:raise ValueError('Vacancy order mismatch')
        self.tfidf=joblib.load(self.bundle/'tfidf.joblib')
        self.documents=sparse.load_npz(self.bundle/'documents.npz')
        self.embeddings=np.load(self.bundle/'job_embeddings.npy')
        self.skills=np.load(self.bundle/'job_skills.npy')
        if self.documents.shape[0]!=len(self.catalog) or self.embeddings.shape!=(len(self.catalog),384) or len(self.skills)!=len(self.catalog):raise ValueError('Artifact dimensions do not align')
        self.extractor=Extractor(self.bundle/'dictionary.json')
        if self.extractor.fingerprint!=self.meta['dictionary_sha256']:raise ValueError('Skill dictionary mismatch')
        from sentence_transformers import SentenceTransformer
        # This BERT encoder must not receive Transformers' unrelated Mistral regex patch.
        self.encoder=SentenceTransformer(str(self.bundle/'encoder'),device='cpu',local_files_only=True,
                                         tokenizer_kwargs={'fix_mistral_regex':False})
        self.estimator=joblib.load(self.bundle/'reranker.joblib') if self.meta['kind']=='learned' else None
        self.lock=threading.Lock()
        self.regions=sorted(self.catalog.regionName.unique().tolist())
        self.groups=pd.factorize(clean_title(self.catalog.vacancyName)+' | '+region_key(self.catalog.regionName))[0]

    def rank_profile(self,profiles,embeddings,skill_vectors):
        rr,ff=candidates(self.tfidf.transform(text(profiles,'cv')),embeddings,self.documents,self.embeddings,
                         profiles,self.catalog,skill_vectors,self.skills,self.meta['use_region'])
        return rerank(rr,ff,self.meta['kind'],self.meta.get('skill_weight',0),self.estimator,self.meta['use_region'])

    def recommend(self,title,history='',education='',skills='',region='',limit=10,group_titles=False):
        if not 1<=limit<=20:raise ValueError('Choose between 1 and 20 recommendations')
        pp=profile(title,history,education,skills,region)
        region_normalized=region_key(pp.localityName).iloc[0]
        known=bool((region_key(self.catalog.regionName)==region_normalized).any())
        vectors=online_skill_vector(self.extractor,history,skills)
        with self.lock:
            embedding=self.encoder.encode(pp.cv_text.str[:400].tolist(),normalize_embeddings=True,convert_to_numpy=True)
            ranked=self.rank_profile(pp,embedding,vectors)[0]
        if group_titles:
            seen=set();kept=[]
            for row in ranked:
                key=self.groups[row]
                if key not in seen:seen.add(key);kept.append(row)
                if len(kept)==limit:break
            rows=kept
        else:rows=ranked[:limit]
        result=self.catalog.iloc[rows].copy().reset_index(drop=True)
        result.insert(0,'rank',np.arange(1,len(result)+1))
        return result,{'selected_model':self.meta['selected_model'],'known_region':known,
                       'usable_skill_evidence':bool(vectors.sum()),'region_priority':self.meta['use_region']}
