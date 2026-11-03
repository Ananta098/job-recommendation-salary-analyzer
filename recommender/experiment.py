"""Train on disjoint applicants, select on development recall, then evaluate once.

python -m recommender.experiment --encoder /path/to/cached/encoder
"""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from .bundle import ROOT, project_path
from importlib.metadata import version
import torch
torch.set_num_threads(4)
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from m9_skills.extraction import Extractor
from m9_skills.compatibility import align_sidecar
from .shared import gold_for, evaluate, grouped_rank, digest, MODEL, WEIGHTS, split
from .core import clean_title, region_key, text, candidates, rerank, evidence_vectors, FEATURE_NAMES




def aggregate(metrics):
    return {mode:{k:float(v.mean()) for k,v in values.items()} for mode,values in metrics.items()}


def train_examples(rows, features, gold, groups):
    rng=np.random.default_rng(42);xs=[];ys=[];weights=[]
    # Use only retrieved candidates: no label-driven positive injection into the pool.
    for i,row in enumerate(rows):
        positive=np.isin(groups[row],gold[i])
        pos=np.flatnonzero(positive);neg=np.flatnonzero(~positive)
        # Include queries without retrieved positives too; give every query equal total weight.
        chosen=np.concatenate([pos,rng.choice(neg,min(30,len(neg)),replace=False)])
        xs.append(features[i,chosen]);ys.append(positive[chosen].astype(int))
        weights.extend([1/len(chosen)]*len(chosen))
    x=np.vstack(xs);y=np.concatenate(ys);w=np.asarray(weights)
    if len(np.unique(y))<2:raise ValueError('Training requires both label classes')
    return x,y,w


def run(root,output,encoder_path,bundle_dir):
    for path in [output,bundle_dir]:
        if path.exists() and any(path.iterdir()):raise ValueError(f'Choose an empty output directory: {path}')
        path.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    data=pd.read_csv(root/'clean/d4_clean.csv',encoding='utf-8-sig',keep_default_na=False,low_memory=False)
    data['is_positive']=data.is_positive.astype(int)
    jobs=data.drop_duplicates('idVacancy').reset_index(drop=True)
    groups=pd.factorize(clean_title(jobs.vacancyName)+' | '+region_key(jobs.regionName))[0]
    rowmap=pd.Series(jobs.index,index=jobs.idVacancy)
    train_ids,dev_ids,test_ids=split(data)
    ids=np.concatenate([train_ids,dev_ids,test_ids]);ntrain=len(train_ids);ndev=len(dev_ids)
    profiles=data.drop_duplicates('idCv').set_index('idCv').loc[ids]
    print(f'Train {ntrain}, development {ndev}, test {len(test_ids)}; {len(jobs)} jobs',flush=True)
    dictionary=root/'m9_skills/dictionary.v2.json';extractor=Extractor(dictionary)
    sidecars=root/'m9_skills/outputs/v2_validation_run'
    cv_records=align_sidecar(ids.tolist(),sidecars/'cv_skills.jsonl','cv',extractor.fingerprint)
    cvskills=evidence_vectors(cv_records,extractor.dictionary,'cv');del cv_records
    job_records=align_sidecar(jobs.idVacancy.tolist(),sidecars/'vacancy_skills.jsonl','vacancy',extractor.fingerprint)
    jobskills=evidence_vectors(job_records,extractor.dictionary,'vacancy');del job_records
    print('Skill evidence aligned',flush=True)
    tfidf=TfidfVectorizer(min_df=3,max_features=120_000,sublinear_tf=True,dtype=np.float32)
    documents=tfidf.fit_transform(text(jobs,'vacancy'))
    jobemb=np.load(root/f'clean/sbert_jobs_{MODEL}.npy').astype(np.float32)
    devemb=np.load(root/f'clean/sbert_val_{MODEL}.npy').astype(np.float32)
    testemb=np.load(root/f'clean/sbert_test_{MODEL}.npy').astype(np.float32)
    if jobemb.shape!=(len(jobs),384) or devemb.shape!=(ndev,384) or testemb.shape!=(len(test_ids),384):
        raise ValueError('Embedding cache shape mismatch')
    print('TF-IDF and embedding caches loaded',flush=True)
    from sentence_transformers import SentenceTransformer
    encoder=SentenceTransformer(str(encoder_path),device='cpu',local_files_only=True)
    check=encoder.encode(profiles.iloc[ntrain+ndev:ntrain+ndev+16].cv_text.str[:400].tolist(),normalize_embeddings=True,convert_to_numpy=True)
    encoder_max_difference=float(np.abs(check-testemb[:16]).max())
    print('Encoder/cache max difference:',encoder_max_difference,flush=True)
    if encoder_max_difference>.001:raise ValueError('Encoder does not match cached test inputs; rebuild all caches consistently')
    cache=root/'artifacts/training_embeddings.npy';cache_meta=cache.with_suffix('.json')
    signature=hashlib.sha256(('\n'.join(train_ids)+'\n'+'\n'.join(profiles.iloc[:ntrain].cv_text.str[:400])+'\n'+str(encoder_path)).encode()).hexdigest()
    if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text()).get('signature')==signature:
        trainemb=np.load(cache)
    else:
        print('Encoding training applicants',flush=True)
        trainemb=encoder.encode(profiles.iloc[:ntrain].cv_text.str[:400].tolist(),batch_size=64,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
        cache.parent.mkdir(exist_ok=True);np.save(cache,trainemb)
        cache_meta.write_text(json.dumps({'signature':signature,'shape':list(trainemb.shape)}))
    all_embeddings=np.vstack([trainemb,devemb,testemb])
    portions={'train':slice(0,ntrain),'development':slice(ntrain,ntrain+ndev),'test':slice(ntrain+ndev,None)}
    pools={};golds={}
    for name,sl in portions.items():
        print('Retrieving '+name,flush=True)
        pp=profiles.iloc[sl]
        rr,ff=candidates(tfidf.transform(text(pp,'cv')),all_embeddings[sl],documents,jobemb,pp,jobs,cvskills[sl],jobskills)
        pools[name]=(rr,ff)
        golds[name]=gold_for(data,ids[sl],rowmap,groups)
    x,y,sample_weights=train_examples(*pools['train'],golds['train'][1],groups)
    positive_weight=float(sample_weights[y==1].sum());negative_weight=float(sample_weights[y==0].sum())
    estimators={
      'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=500,class_weight='balanced',random_state=42)),
      'Random Forest':RandomForestClassifier(n_estimators=150,max_depth=10,min_samples_leaf=20,class_weight='balanced_subsample',n_jobs=4,random_state=42),
      'XGBoost':XGBClassifier(n_estimators=200,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,reg_lambda=10,scale_pos_weight=negative_weight/positive_weight,tree_method='hist',eval_metric='logloss',n_jobs=4,random_state=42)}
    for name,estimator in estimators.items():
        print('Fitting '+name,flush=True)
        if name=='Logistic Regression':estimator.fit(x,y,logisticregression__sample_weight=sample_weights)
        else:estimator.fit(x,y,sample_weight=sample_weights)
    model_specs={'M8':{'kind':'M8','use_region':True},'M9 without region':{'kind':'skill','use_region':False},'M9 + region':{'kind':'skill','use_region':True}}
    global_pools={}
    for name in ['development','test']:
        sl=portions[name];pp=profiles.iloc[sl]
        global_pools[name]=candidates(tfidf.transform(text(pp,'cv')),all_embeddings[sl],documents,jobemb,pp,jobs,cvskills[sl],jobskills,use_region=False)
    tuning=[]
    for model in ['M9 without region','M9 + region']:
        rr,ff=pools['development'] if model_specs[model]['use_region'] else global_pools['development']
        curve=[]
        for weight in WEIGHTS:
            ranked=rerank(rr,ff,'skill',weight,use_region=model_specs[model]['use_region'])
            result=evaluate(ranked,*golds['development'],groups)
            recall=float(result['grouped']['Recall@10'].mean())
            curve.append((weight,recall));tuning.append({'model':model,'skill_weight':weight,'grouped_Recall@10':recall})
        model_specs[model]['skill_weight']=max(curve,key=lambda z:(z[1],-z[0]))[0]
    for name in estimators:model_specs[name]={'kind':'learned','use_region':True}
    pd.DataFrame(tuning).to_csv(output/'development_skill_weights.csv',index=False)
    ranking={};metrics={};dev_scores=[];dev_recall={}
    def ranking_for(name,portion):
        spec=model_specs[name];rr,ff=(pools[portion] if spec['use_region'] else global_pools[portion])
        return rerank(rr,ff,spec['kind'],spec.get('skill_weight',0),estimators.get(name),spec['use_region'])
    for name in model_specs:
        rr=ranking_for(name,'development');result=evaluate(rr,*golds['development'],groups)
        dev_scores.append({'model':name,**aggregate(result)['grouped']})
        dev_recall[name]=result['grouped']['Recall@10']
    # Conservative deployment promotion, based exclusively on paired development outcomes.
    # This safeguard was added after exploratory comparisons; do not call it preregistered.
    dev_table=pd.DataFrame(dev_scores).set_index('model')
    development_max=max(model_specs,key=lambda name:dev_table.loc[name,'Recall@10'])
    promotion=[];eligible=['M8'];family_size=len(model_specs)-1
    for name in model_specs:
        if name=='M8':continue
        delta=dev_recall[name]-dev_recall['M8'];rng=np.random.default_rng(42)
        means=[delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(10000)]
        lower,upper=np.quantile(means,[.025/family_size,1-.025/family_size])
        promote=bool(lower>0)
        if promote:eligible.append(name)
        promotion.append({'model':name,'mean_recall_gain':float(delta.mean()),'simultaneous_ci_lower':float(lower),'simultaneous_ci_upper':float(upper),'promote':promote})
    selected=max(eligible,key=lambda name:dev_table.loc[name,'Recall@10'])
    pd.DataFrame(promotion).to_csv(output/'development_promotion.csv',index=False)
    pd.DataFrame({'idCv':dev_ids,**dev_recall}).to_csv(output/'development_per_applicant.csv',index=False)
    selection={'selected_model':selected,'criterion':'retain M8 unless a candidate has positive lower paired-development bootstrap bound after Bonferroni family correction; then maximize development grouped Recall@10, with fixed-order ties',
               'development_max_without_promotion_gate':development_max,
               'promotion_protocol':'10,000 paired bootstrap resamples, seed 42; 95% family coverage across five candidate comparisons (99% per comparison). Added after exploratory analysis; not preregistered.',
               'model_order':list(model_specs),'specification':model_specs[selected],'development_scores':dev_table['Recall@10'].to_dict(),
               'training_applicants':ntrain,'development_applicants':ndev,'test_applicants':len(test_ids),
               'features':FEATURE_NAMES,'labels':'positive if candidate title-region group matches a recorded invitation; other retrieved groups are weak negatives',
               'negative_sampling':'up to 30 non-invited candidate groups per training applicant; all retrieved positive groups kept; no gold injection',
               'test_selection_used':False}
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    dev_table.to_csv(output/'development_metrics_grouped.csv')
    print('Selected on development:',selected,flush=True)
    for name in model_specs:
        ranked=ranking_for(name,'test');ranking[name]=ranked
        metrics[name]=evaluate(ranked,*golds['test'],groups)
    # A true full-catalog M5 comparator (not a top-300 M8 approximation).
    pp=profiles.iloc[portions['test']];q=tfidf.transform(text(pp,'cv'));m5=[]
    jr=region_key(jobs.regionName).to_numpy();cr=region_key(pp.localityName).to_numpy()
    for i in range(0,len(pp),100):
        scores=(q[i:i+100]@documents.T).toarray();scores[cr[i:i+100,None]!=jr[None,:]]-=10
        m5.append(np.argsort(-scores,axis=1)[:,:300])
    ranking['M5']=np.vstack(m5);metrics['M5']=evaluate(ranking['M5'],*golds['test'],groups)
    for mode in ['strict','grouped']:
        pd.DataFrame({name:aggregate(value)[mode] for name,value in metrics.items()}).T.to_csv(output/f'metrics_{mode}.csv',index_label='model')
    per=pd.DataFrame({'idCv':test_ids,'region':region_key(pp.localityName).to_numpy(),'has_usable_skills':cvskills[portions['test']].sum(axis=1)>0})
    for name,result in metrics.items():
        for mode in ['strict','grouped']:
            for metric in ['Recall@10','NDCG@10']:
                per[f'{name}|{mode}|{metric}']=result[mode][metric]
    per.to_csv(output/'per_applicant.csv',index=False)
    # Store only first 20 row indices for local reference checks; no model refit on dev/test.
    np.savez_compressed(output/'reference_rankings.npz',**{name:rows[:,:20] for name,rows in ranking.items()})
    # Export an inference-ready bundle using the same fitted objects and candidate policy.
    print('Exporting selected-model bundle',flush=True)
    joblib.dump(tfidf,bundle_dir/'tfidf.joblib');sparse.save_npz(bundle_dir/'documents.npz',documents)
    np.save(bundle_dir/'job_embeddings.npy',jobemb);np.save(bundle_dir/'job_skills.npy',jobskills)
    catalog_columns=[c for c in ['idVacancy','vacancyName','vacancyName_en','regionName','job_text','job_skills','salaryMin_vacancy','experienceRequirements','datePublished','professionalSphereName'] if c in jobs]
    jobs[catalog_columns].to_csv(bundle_dir/'catalog.csv',index=False)
    (bundle_dir/'dictionary.json').write_text(dictionary.read_text())
    if selected in estimators:joblib.dump(estimators[selected],bundle_dir/'reranker.joblib')
    encoder.save(str(bundle_dir/'encoder'))
    # No training profiles, applicant IDs or application outcomes enter the bundle.
    metadata={'format_version':1,'selected_model':selected,**model_specs[selected], 'features':FEATURE_NAMES,
              'candidate_count':300,'semantic_max_chars':400,'lexical_weight':.5,'seed':42,
              'vacancies':len(jobs),'embedding_dimension':384,'encoder_revision':encoder_path.name,
              'catalog_ids_sha256':hashlib.sha256('\n'.join(jobs.idVacancy).encode()).hexdigest(),
              'dictionary_sha256':extractor.fingerprint,'encoder_cache_max_difference':encoder_max_difference,
              'versions':{p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','sentence-transformers','torch','joblib','streamlit']}}
    (bundle_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    bundle_manifest={str(p.relative_to(bundle_dir)):digest(p) for p in bundle_dir.rglob('*') if p.is_file()}
    (bundle_dir/'manifest.json').write_text(json.dumps(bundle_manifest,indent=2)+'\n')
    manifest={'python':platform.python_version(),'versions':metadata['versions'],'encoder_revision':encoder_path.name,
              'input_hashes':{str(p.relative_to(root)):digest(p) for p in [root/'clean/d4_clean.csv',dictionary,*[(root/f'clean/sbert_{x}_{MODEL}.npy') for x in ['jobs','val','test']],sidecars/'cv_skills.jsonl',sidecars/'vacancy_skills.jsonl']},
              'split_hashes':{name:hashlib.sha256('\n'.join(ids[sl]).encode()).hexdigest() for name,sl in portions.items()},
              'source_hashes':{'recommender/'+p.name:digest(p) for p in Path(__file__).resolve().parent.glob('*.py')},
              'encoder_cache_max_difference':encoder_max_difference,'train_examples':len(y),'train_positive_rows':int(y.sum()),
              'train_parameters':{name:repr(est) for name,est in estimators.items()},'total_seconds':time.perf_counter()-started,
              'historical_M8_grouped_recall10':float(pd.read_csv(root/'results/baselines/metrics_grouped_all.csv',index_col=0).loc['M8 Hybrid + region','Recall@10']),
              'current_M8_grouped_recall10':float(metrics['M8']['grouped']['Recall@10'].mean()),
              'limitations':['Historical test applicants have been examined in earlier project iterations; this is not a newly collected untouched external test.',
                             'Invitation-group labels and sampled weak negatives do not establish true job relevance or calibrated probabilities.',
                             'M9 evidence uses first raw snapshots; baseline cleaning is outcome-dependent. No causal isolation or temporal guarantee.',
                             'Dictionary human review remains pending.']}
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(selection,indent=2),flush=True)
    print(pd.read_csv(output/'metrics_grouped.csv').to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=Path('results/final'))
    p.add_argument('--bundle',type=Path,default=Path('artifacts/deployment'));p.add_argument('--encoder',type=Path,required=True)
    a=p.parse_args();run(project_path(a.root),project_path(a.output,project_path(a.root)),project_path(a.encoder,project_path(a.root)),project_path(a.bundle,project_path(a.root)))

if __name__=='__main__':main()
