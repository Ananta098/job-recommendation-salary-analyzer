"""Paired significance diagnostics and applicant/job counts per region."""
import argparse,json
from pathlib import Path
from .bundle import ROOT, project_path
import numpy as np
import pandas as pd
from .shared import region_key
from .shared import split


def paired(delta,resamples=10000):
    delta=np.asarray(delta,dtype=float);rng=np.random.default_rng(42)
    if not np.any(delta):return 0.,0.,1.
    bootstrap=[];extreme=0;observed=abs(delta.mean())
    for start in range(0,resamples,100):
        n=min(100,resamples-start)
        bootstrap.extend(delta[rng.integers(0,len(delta),size=(n,len(delta)))].mean(axis=1))
        null=(delta*rng.choice([-1,1],size=(n,len(delta)))).mean(axis=1)
        extreme+=int((np.abs(null)>=observed-1e-15).sum())
    low,high=np.quantile(bootstrap,[.025,.975])
    return float(low),float(high),float((extreme+1)/(resamples+1))


def holm(pvalues):
    p=np.asarray(pvalues);order=np.argsort(p);adjusted=np.empty_like(p);running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(p)-rank)*p[index]);adjusted[index]=min(1.,running)
    return adjusted


def run(root, results=None):
    out=project_path(results,root) if results is not None else root/'results/final';per=pd.read_csv(out/'per_applicant.csv',keep_default_na=False)
    selected=json.loads((out/'selection.json').read_text())['selected_model']
    models=[c.split('|')[0] for c in per if c.endswith('|grouped|Recall@10')]
    rows=[]
    for name in models:
        if name=='M8':continue
        for mode in ['grouped','strict']:
            delta=per[f'{name}|{mode}|Recall@10'].to_numpy()-per[f'M8|{mode}|Recall@10'].to_numpy()
            low,high,p=paired(delta)
            rows.append({'model':name,'reference':'M8','mode':mode,'metric':'Recall@10','applicants':len(per),
                         'mean_difference':float(delta.mean()),'ci95_lower':low,'ci95_upper':high,
                         'paired_permutation_p':p,'improved':int((delta>0).sum()),'worse':int((delta<0).sum())})
    result=pd.DataFrame(rows)
    # Primary family: grouped Recall@10 comparisons versus M8; strict is a separate family.
    for mode in ['grouped','strict']:
        mask=result['mode']==mode
        result.loc[mask,'holm_adjusted_permutation_p']=holm(result.loc[mask,'paired_permutation_p'])
    result.to_csv(out/'significance.csv',index=False)
    df=pd.read_csv(root/'clean/d4_clean.csv',encoding='utf-8-sig',keep_default_na=False,low_memory=False)
    df['region']=region_key(df.localityName);cv=df.drop_duplicates('idCv');jobs=df.drop_duplicates('idVacancy').copy();jobs['region']=region_key(jobs.regionName)
    train,dev,test=split(df)
    table=pd.DataFrame({'applicants':cv.groupby('region').size(),'vacancies':jobs.groupby('region').size(),
                        'applications':df.groupby('region').size(),'positive_applications':df[df.is_positive.astype(int)==1].groupby('region').size(),
                        'training_applicants':cv[cv.idCv.isin(train)].groupby('region').size(),
                        'development_applicants':cv[cv.idCv.isin(dev)].groupby('region').size(),
                        'test_applicants':cv[cv.idCv.isin(test)].groupby('region').size()}).fillna(0).astype(int)
    table['invitation_rate']=table.positive_applications/table.applications.replace(0,np.nan)
    table.sort_values('applicants',ascending=False).to_csv(out/'region_counts.csv',index_label='region')
    regional=[]
    for region,part in per.groupby('region'):
        for model in models:
            regional.append({'region':region,'test_applicants':len(part),'model':model,
                             'grouped_Recall@10':part[f'{model}|grouped|Recall@10'].mean(),
                             'grouped_NDCG@10':part[f'{model}|grouped|NDCG@10'].mean(),
                             'small_sample':len(part)<20})
    pd.DataFrame(regional).to_csv(out/'metrics_by_region.csv',index=False)
    summary={'primary':'paired permutation test on applicant-level grouped Recall@10 differences versus same-run M8',
             'paired_bootstrap_resamples':10000,'seed':42,'multiple_comparisons':'Holm adjustment across models within each strict/grouped family',
             'regional_analysis':'Descriptive; fewer than 20 test applicants flagged. Counts use applicant region for applications and vacancy region for job counts.',
             'selected_model':selected,'limitations':'Observed invitation groups are incomplete relevance labels; historical test set has been inspected previously. No temporal or causal guarantee.'}
    (out/'analysis_protocol.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(result.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--results',type=Path)
    a=p.parse_args();run(project_path(a.root),a.results)
