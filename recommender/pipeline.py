"""Single execution flow: prepare -> skills -> compare/export -> analyze -> review.

python -m recommender.pipeline run --run-dir artifacts/runs/run-01 --reuse-prepared
python -m recommender.pipeline review
"""
import argparse
import ast
import json
import shutil
from pathlib import Path
from .bundle import ROOT, project_path
from .shared import MODEL, ENCODER_REVISION, digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def review(root, results=None, bundle=None):
    import pandas as pd
    root=project_path(root)
    results=project_path(results,root) if results else root/'results/final'
    bundle=project_path(bundle,root) if bundle else root/'artifacts/deployment'
    source_root=Path(__file__).resolve().parent.parent
    for file in list((source_root/'recommender').glob('*.py'))+[source_root/'app.py',source_root/'deploy/aws/deploy.py']:
        ast.parse(file.read_text())
    selection=json.loads((results/'selection.json').read_text())
    grouped=pd.read_csv(results/'metrics_grouped.csv',index_col=0)
    strict=pd.read_csv(results/'metrics_strict.csv',index_col=0)
    required={'M8','M9 without region','M9 + region','Logistic Regression','Random Forest','XGBoost','M5'}
    if not required.issubset(grouped.index) or not required.issubset(strict.index):
        raise ValueError('Model comparison is incomplete')
    if selection['selected_model'] not in grouped.index or selection.get('test_selection_used') is not False:
        raise ValueError('Selection contract failed')
    for filename in ['development_promotion.csv','significance.csv','region_counts.csv','metrics_by_region.csv','analysis_protocol.json','run_manifest.json']:
        if not (results/filename).is_file():raise ValueError('Missing final output: '+filename)
    regional=pd.read_csv(results/'region_counts.csv')
    if int(regional.test_applicants.sum())!=selection['test_applicants']:
        raise ValueError('Region counts do not cover the test set')
    private=results/'per_applicant.csv'
    if private.exists():
        per=pd.read_csv(private)
        if len(per)!=selection['test_applicants'] or per.idCv.nunique()!=len(per):
            raise ValueError('Applicant-level results are misaligned')
        for name in required:
            for mode, table in [('strict',strict),('grouped',grouped)]:
                for metric in ['Recall@10','NDCG@10']:
                    if abs(per[f'{name}|{mode}|{metric}'].mean()-table.loc[name,metric])>1e-10:
                        raise ValueError('Aggregate mismatch: '+name)
    manifest_path=bundle/'manifest.json'
    bundle_verified=False
    if manifest_path.exists():
        for rel,expected in json.loads(manifest_path.read_text()).items():
            path=(bundle/rel).resolve()
            if not path.is_relative_to(bundle) or not path.is_file() or digest(path)!=expected:
                raise ValueError('Bundle integrity failure: '+rel)
        metadata=json.loads((bundle/'metadata.json').read_text())
        if metadata['selected_model']!=selection['selected_model']:
            raise ValueError('Serving model does not match selected model')
        bundle_verified=True
    report={'code_and_aggregate_results_complete':True,'selected_model':selection['selected_model'],
            'model_comparisons':len(grouped),'test_applicants':selection['test_applicants'],
            'applicant_aggregate_consistency_checked':private.exists(),'bundle_verified':bundle_verified,
            'deployment_status':'Local Streamlit verified separately; Docker engine unavailable; AWS is a template only.',
            'scientific_limits':['Invitation-only evaluation; weak negative labels.',
                                 'Promotion rule added after exploratory comparisons.',
                                 'Outcome-dependent cleaned snapshots; no temporal guarantee.',
                                 'Skill dictionary human review remains pending.']}
    return report


def run(root, run_dir, encoder=None, raw=None, reuse=False):
    root=project_path(root);workspace=project_path(run_dir,root)
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError('Choose a NEW empty --run-dir; existing runs are never overwritten')
    if workspace==root or root.is_relative_to(workspace):
        raise ValueError('Run directory must not contain the source repository')
    raw=project_path(raw,root) if raw else root/'datasets/HSE_dataset(D4).csv'
    if not raw.is_file():raise FileNotFoundError('Supply D4 at '+str(raw))
    workspace.mkdir(parents=True,exist_ok=True)
    state={'status':'running','steps':[],'root':str(workspace),'reuse_prepared':reuse}
    def step(name, function):
        state['active_step']=name;write(workspace/'pipeline_status.json',state)
        print('\nSTEP: '+name,flush=True)
        value=function();state['steps'].append(name);write(workspace/'pipeline_status.json',state)
        return value
    try:
        for rel in ['lookup/job_titles_ru_en.csv','m9_skills/dictionary.v2.json','results/baselines/metrics_grouped_all.csv']:
            copy_file(root/rel,workspace/rel)
        target=workspace/'datasets/HSE_dataset(D4).csv';target.parent.mkdir(exist_ok=True)
        try:target.symlink_to(raw)
        except OSError:copy_file(raw,target)
        if reuse:
            def reuse_inputs():
                previous=json.loads((root/'results/final/run_manifest.json').read_text())
                inputs=previous['input_hashes']
                for rel,expected in inputs.items():
                    source=root/rel
                    if not source.is_file() or digest(source)!=expected:
                        raise ValueError('Prepared input changed; use a full rebuild: '+rel)
                    copy_file(source,workspace/rel)
                coverage_path=root/'m9_skills/outputs/v2_validation_run/coverage.json'
                coverage=json.loads(coverage_path.read_text())
                if digest(raw)!=coverage['source_sha256']:
                    raise ValueError('Raw D4 differs from the recorded extraction input')
                copy_file(coverage_path,workspace/'m9_skills/outputs/v2_validation_run/coverage.json')
                for name in ['training_embeddings.npy','training_embeddings.json']:
                    if (root/'artifacts'/name).exists():copy_file(root/'artifacts'/name,workspace/'artifacts'/name)
                write(workspace/'prepared_inputs.json',{'policy':'Reused only after checking every recorded input hash and the raw D4 hash','input_hashes':inputs})
            step('verify_and_reuse_prepared_inputs',reuse_inputs)
        else:
            from .cleaning import clean_data
            step('clean_D4',lambda:clean_data(workspace))
        from huggingface_hub import snapshot_download
        if encoder is None:
            encoder=step('resolve_pinned_encoder',lambda:Path(snapshot_download('sentence-transformers/'+MODEL,revision=ENCODER_REVISION)))
        else:encoder=project_path(encoder,root)
        if not encoder.is_dir():raise FileNotFoundError('Encoder directory missing: '+str(encoder))
        if not reuse:
            def embeddings():
                import numpy as np
                import pandas as pd
                from sentence_transformers import SentenceTransformer
                from .shared import split
                import torch
                torch.set_num_threads(4)
                data=pd.read_csv(workspace/'clean/d4_clean.csv',encoding='utf-8-sig',keep_default_na=False,low_memory=False)
                jobs=data.drop_duplicates('idVacancy');_,dev,test=split(data)
                profiles=data.drop_duplicates('idCv').set_index('idCv')
                model=SentenceTransformer(str(encoder),device='cpu',local_files_only=True,tokenizer_kwargs={'fix_mistral_regex':False})
                provenance={'encoder_revision':ENCODER_REVISION,'clean_sha256':digest(workspace/'clean/d4_clean.csv'),'parts':{}}
                import hashlib
                for side,ids,texts in [('jobs',jobs.idVacancy.tolist(),jobs.job_text.str[:400].tolist()),
                                       ('val',list(dev),profiles.loc[dev].cv_text.str[:400].tolist()),
                                       ('test',list(test),profiles.loc[test].cv_text.str[:400].tolist())]:
                    array=model.encode(texts,batch_size=64,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True).astype(np.float32)
                    path=workspace/f'clean/sbert_{side}_{MODEL}.npy';np.save(path,array)
                    provenance['parts'][side]={'ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),'file_sha256':digest(path),'rows':len(ids)}
                write(workspace/'prepared_inputs.json',provenance)
            step('encode_catalog_and_evaluation_profiles',embeddings)
            from m9_skills.workflow import extract
            step('extract_and_normalize_skills',lambda:extract(raw,workspace/'m9_skills/outputs/v2_validation_run',dictionary=workspace/'m9_skills/dictionary.v2.json'))
        from .experiment import run as experiment
        results=workspace/'results/final';bundle=workspace/'artifacts/deployment'
        step('train_compare_select_and_export',lambda:experiment(workspace,results,encoder,bundle))
        from .analysis import run as analyze
        step('paired_significance_and_region_counts',lambda:analyze(workspace,results))
        result=step('final_consistency_review',lambda:review(workspace,results,bundle))
        write(workspace/'final_review.json',result)
        state['status']='complete';state.pop('active_step',None)
        state['results']=str(results);state['bundle']=str(bundle)
        write(workspace/'pipeline_status.json',state)
        print('\nComplete. To serve this run from the source repository:\nMODEL_BUNDLE="'+str(bundle)+'" streamlit run app.py',flush=True)
    except Exception as error:
        state['status']='failed';state['error']=str(error);write(workspace/'pipeline_status.json',state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('run');build.add_argument('--root',type=Path,default=ROOT)
    build.add_argument('--run-dir',type=Path,required=True);build.add_argument('--encoder',type=Path)
    build.add_argument('--raw',type=Path);build.add_argument('--reuse-prepared',action='store_true')
    check=sub.add_parser('review');check.add_argument('--root',type=Path,default=ROOT)
    check.add_argument('--results',type=Path);check.add_argument('--bundle',type=Path)
    args=parser.parse_args()
    if args.command=='run':run(args.root,args.run_dir,args.encoder,args.raw,args.reuse_prepared)
    else:print(json.dumps(review(args.root,args.results,args.bundle),indent=2))

if __name__=='__main__':main()
