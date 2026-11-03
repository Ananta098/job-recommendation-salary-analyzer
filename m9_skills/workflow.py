"""Offline CLI for dictionary proposals and extraction diagnostics, never rankings."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from .data import entities, SKILL_FIELDS
from .extraction import Extractor, normalize

def ensure_empty(path):
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output must be empty/new')
    path.mkdir(parents=True,exist_ok=True)

def write_csv(path, rows, fields):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def mine(raw, output):
    # Vacancy catalog only: never inspect held-out resume text or outcome labels
    # for dictionary discovery. Catalog-wide text fitting matches the M8 protocol.
    output=Path(output); ensure_empty(output)
    counts=Counter(); examples={}; errors=Counter(); n=0
    for entity in entities(raw,'vacancy'):
        n+=1
        for field,status in entity['parse_errors'].items(): errors[field+':'+status]+=1
        terms={normalize(s).strip() for values in entity['structured'].values() for s in values}
        for term in terms:
            counts[term]+=1
            examples.setdefault(term,entity['entity_id'])
    rows=[{'phrase':p,'vacancy_document_count':c,'example_id':examples[p],
           'proposed_skill_id':'','decision':'pending','reviewer':'','reviewed_at':'','source':'D4 structured vacancy fields; first raw snapshot',
           'reason':''} for p,c in sorted(counts.items(),key=lambda x:(-x[1],x[0]))]
    fields=['phrase','vacancy_document_count','example_id','proposed_skill_id','decision','reviewer','reviewed_at','source','reason']
    write_csv(output/'manual_review.csv',rows,fields)
    extractor=Extractor()
    att=[]
    for skill in extractor.dictionary['skills']:
        for alias in skill['aliases']:
            att.append({'skill_id':skill['id'],'alias':alias,'exact_structured_vacancy_count':counts[normalize(alias)],
                        'review_status':skill['review']['status']})
    write_csv(output/'seed_attestation.csv',att,['skill_id','alias','exact_structured_vacancy_count','review_status'])
    (output/'mining_manifest.json').write_text(json.dumps({'source':str(Path(raw).resolve()),'entities':n,
        'candidate_phrases':len(counts),'scope':'unique vacancy catalog only; no resumes or outcomes',
        'snapshot':'first raw occurrence, independent of outcomes; not a chronological snapshot',
        'parse_errors':dict(errors)},ensure_ascii=False,indent=2))
    return {'vacancies':n,'candidate_phrases':len(counts)}

def extract(raw, output, limit=None, dictionary=None):
    out=Path(output);ensure_empty(out)
    engine=Extractor(dictionary); diagnostics={}; unknown=Counter()
    for side in ('cv','vacancy'):
        total=covered=listed=both=negated=errors=0; counts=Counter(); domains=Counter()
        domain_by_id={s['id']:s['domain'] for s in engine.dictionary['skills']}
        with (out/(side+'_skills.jsonl')).open('w',encoding='utf-8') as f:
            for e in entities(raw,side):
                if limit is not None and total>=limit:break
                total+=1
                listed+=bool(any(e['structured'].values()))
                result=engine.extract_fields(e['fields'],SKILL_FIELDS[side])
                covered+=bool(result['skill_ids'])
                both+=bool(result['skill_ids']) and bool(any(e['structured'].values()))
                negated+=sum(x['assertion']!='mentioned' for x in result['evidence'])
                errors+=len(e['parse_errors'])
                counts.update(result['skill_ids'])
                domains.update(set(domain_by_id[sid] for sid in result['skill_ids']))
                for terms in e['structured'].values():
                    for term in terms:
                        if not any(x['assertion']=='mentioned' for x in engine.extract(term)):
                            unknown[(side,normalize(term))]+=1
                f.write(json.dumps({'entity_id':e['entity_id'],'side':side,**result,
                                     'parse_errors':e['parse_errors']},ensure_ascii=False)+'\n')
        diagnostics[side]={'entities':total,'entities_with_listed_skills':listed,'entities_with_extracted_skills':covered,
                           'entities_with_both':both,'mention_coverage':covered/total if total else 0,
                           'nonaffirmative_mentions':negated,'malformed_structured_fields':errors,
                           'skill_entity_counts':dict(counts),'domain_entity_counts':dict(domains)}
    write_csv(out/'unmatched_structured.csv',({'side':s,'phrase':p,'occurrence_count':c} for (s,p),c in unknown.most_common()),
              ['side','phrase','occurrence_count'])
    manifest={'dictionary_sha256':engine.fingerprint,'dictionary_version':engine.version,
              'source':str(Path(raw).resolve()),'source_size':Path(raw).stat().st_size,
              'source_sha256':file_hash(raw), 'limit_per_side':limit,
              'snapshot_policy':'first raw occurrence per entity, independent of labels; no chronological claim',
              'coverage_is_accuracy':False,'esco_verified_skills':sum(s['esco'] is not None for s in engine.dictionary['skills']),
              'human_review_complete':False,'diagnostics':diagnostics}
    (out/'coverage.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['mine','extract'])
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--dictionary',type=Path)
    args=parser.parse_args()
    if args.limit is not None and args.limit<=0:parser.error('--limit must be positive')
    result=mine(args.raw,args.output) if args.stage=='mine' else extract(args.raw,args.output,args.limit,args.dictionary)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
