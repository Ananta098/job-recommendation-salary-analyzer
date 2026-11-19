"""Compile explicitly reviewed alias decisions into a NEW dictionary; never auto-approve."""
import argparse
import csv
import copy
import json
from pathlib import Path
from .extraction import load_dictionary

def compile_review(dictionary, review_csv, destination):
    destination=Path(destination)
    if destination.exists():raise ValueError('Destination exists; use a new version')
    data=copy.deepcopy(load_dictionary(dictionary))
    index={s['id']:s for s in data['skills']}
    decisions=[]
    with open(review_csv,encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['decision'] not in ('approve','reject','pending'):raise ValueError('Unknown decision')
            if row['decision']=='pending':continue
            if not all(row.get(k,'').strip() for k in ('reviewer','reviewed_at','source','reason')):
                raise ValueError('A decided alias needs reviewer, date, source and reason')
            sid=row['proposed_skill_id']
            if sid not in index:raise ValueError('Unknown canonical ID: '+sid)
            phrase=row['phrase'].strip()
            if not phrase:raise ValueError('Empty phrase')
            aliases=index[sid]['aliases']
            if row['decision']=='approve' and phrase not in aliases:aliases.append(phrase)
            if row['decision']=='reject':index[sid]['aliases']=[a for a in aliases if a!=phrase]
            decisions.append(row)
    # Each decision is recorded without pretending the entire seed was human-reviewed.
    data['version']=data['version']+'+review'
    data.setdefault('manual_alias_decisions',[]).extend(decisions)
    for s in data['skills']:
        if not s['aliases']:raise ValueError('Canonical skill has no aliases')
    # Validate using a temporary peer; remove it on any failure.
    temp=destination.with_suffix(destination.suffix+'.tmp')
    if temp.exists():raise ValueError('Temporary destination exists')
    try:
        temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        load_dictionary(temp)
        temp.rename(destination)
    finally:
        if temp.exists():temp.unlink()
    return len(decisions)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dictionary',required=True);p.add_argument('--review',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(compile_review(a.dictionary,a.review,a.output),'decisions compiled')
