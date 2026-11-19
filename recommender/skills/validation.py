"""Evaluate skill extraction against explicit Russian diagnostic annotations.

Not a recommendation evaluator. Agent annotations remain distinguishable from human
approval; authored and corpus cases are always reported separately.
"""
import argparse
import csv
import json
from pathlib import Path
from .extraction import Extractor

def read_cases(path):
    cases=[];seen=set()
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['case_id'] in seen:raise ValueError('Duplicate case ID')
            seen.add(row['case_id'])
            row['expected_skill_ids']=json.loads(row['expected_skill_ids'])
            row['expected_assertions']=json.loads(row['expected_assertions'])
            row['unknown_skill_labels']=json.loads(row['unknown_skill_labels'])
            decision=row['human_decision'].strip()
            if decision not in ('pending','approve','correct','exclude'):raise ValueError('Invalid human decision')
            if decision!='pending' and not row['human_reviewer'].strip():raise ValueError('Human decisions require a named reviewer')
            if decision=='correct':
                row['expected_skill_ids']=json.loads(row['human_expected_skill_ids'])
                if not isinstance(row['expected_skill_ids'],list):raise ValueError('Corrected IDs must be a JSON list')
            if not isinstance(row['expected_skill_ids'],list) or not all(isinstance(x,str) for x in row['expected_skill_ids']):
                raise ValueError('Expected IDs must be a list of strings')
            cases.append(row)
    return cases

def metric(rows):
    tp=sum(len(r['true_positive']) for r in rows);fp=sum(len(r['false_positive']) for r in rows);fn=sum(len(r['false_negative']) for r in rows)
    return {'cases':len(rows),'true_positive':tp,'false_positive':fp,'false_negative':fn,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'exact_match_cases':sum(r['predicted_ids']==r['expected_ids'] for r in rows),
            'assertion_error_cases':sum(bool(r['assertion_errors']) for r in rows),
            'cases_with_out_of_inventory_gaps':sum(bool(r['unknown_skill_labels']) for r in rows)}

def evaluate(cases, dictionary=None):
    engine=Extractor(dictionary);known={s['id'] for s in engine.dictionary['skills']}
    details=[];excluded=[]
    for c in cases:
        if c['human_decision']=='exclude' or (c['annotation_status']=='ambiguous_excluded' and c['human_decision']=='pending'):
            excluded.append(c['case_id']);continue
        # Gold may include skills absent from the baseline. These are genuine misses.
        result=engine.extract_fields({c['source_field']:c['text']})
        gold=set(c['expected_skill_ids']);pred=set(result['skill_ids'])
        assertions={}
        for e in result['evidence']:assertions.setdefault(e['skill_id'],set()).add(e['assertion'])
        errors=[sid+': expected '+v+', got '+','.join(sorted(assertions.get(sid,{'unrecognized'})))
                for sid,v in c['expected_assertions'].items() if assertions.get(sid,set())!={v}]
        details.append({'case_id':c['case_id'],'suite':c['suite'],'text':c['text'],
            'expected_ids':sorted(gold),'predicted_ids':sorted(pred),
            'true_positive':sorted(gold&pred),'false_positive':sorted(pred-gold),'false_negative':sorted(gold-pred),
            'assertion_errors':errors,'unknown_skill_labels':c['unknown_skill_labels'],
            'human_reviewed':c['human_decision'] in ('approve','correct')})
    suites={s:metric([r for r in details if r['suite']==s]) for s in sorted({r['suite'] for r in details})}
    human=[r for r in details if r['human_reviewed']]
    return {'dictionary_version':engine.version,'dictionary_sha256':engine.fingerprint,'canonical_skills':len(known),
            'annotation_basis':'Agent-reviewed diagnostics; human decisions separately counted. Not independent or representative evaluation.',
            'excluded_case_ids':excluded,'by_suite':suites,'human_reviewed_only':metric(human) if human else None,
            'human_reviewed_cases':len(human),'details':details}

def compare(cases_path, baseline_dictionary, expanded_dictionary, output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new/empty validation output directory')
    cases=read_cases(cases_path)
    reports={'baseline':evaluate(cases,baseline_dictionary),'expanded':evaluate(cases,expanded_dictionary)}
    output.mkdir(parents=True,exist_ok=True)
    for name,report in reports.items():
        (output/(name+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return reports

def coverage_by_sphere(clean_path, baseline_sidecar, expanded_sidecar):
    """Unique-vacancy coverage, not interaction-weighted counts."""
    spheres={}
    with Path(clean_path).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            spheres.setdefault(row['idVacancy'], row.get('professionalSphereName') or 'Unknown')
    coverage={}
    for name,path in (('baseline',baseline_sidecar),('expanded',expanded_sidecar)):
        coverage[name]={}
        with Path(path).open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);eid=row['entity_id']
                if row['side']!='vacancy' or eid in coverage[name]:raise ValueError('Invalid vacancy sidecar')
                coverage[name][eid]=bool(row['skill_ids'])
        if set(spheres)!=set(coverage[name]):raise ValueError('Vacancy catalog mismatch')
    result={}
    for eid,sphere in spheres.items():
        r=result.setdefault(sphere,{'sphere':sphere,'vacancies':0,'baseline_recognized':0,'expanded_recognized':0})
        r['vacancies']+=1
        for name in coverage:r[name+'_recognized']+=coverage[name][eid]
    for r in result.values():
        for name in coverage:r[name+'_coverage']=r[name+'_recognized']/r['vacancies']
    return sorted(result.values(),key=lambda r:(-r['vacancies'],r['sphere']))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',required=True);p.add_argument('--baseline',required=True);p.add_argument('--expanded',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();reports=compare(a.cases,a.baseline,a.expanded,a.output)
    print(json.dumps({k:v['by_suite'] for k,v in reports.items()},ensure_ascii=False,indent=2))
