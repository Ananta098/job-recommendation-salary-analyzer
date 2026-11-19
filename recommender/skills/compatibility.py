"""Attach sidecar evidence by ID without changing existing model inputs or order."""
import json
from pathlib import Path

def align_sidecar(ordered_ids, sidecar_path, side, expected_dictionary_sha256=None):
    if side not in ('cv','vacancy'):
        raise ValueError('side must be cv or vacancy')
    ordered_ids=list(ordered_ids)
    if len(ordered_ids)!=len(set(ordered_ids)):
        raise ValueError('Requested entity IDs must be unique')
    index={};fingerprints=set()
    with Path(sidecar_path).open(encoding='utf-8') as f:
        for line in f:
            record=json.loads(line);eid=record['entity_id']
            if record['side']!=side:raise ValueError('Side mismatch')
            if eid in index:raise ValueError('Duplicate sidecar entity ID: '+eid)
            if record['skill_ids']!=sorted(set(record['skill_ids'])):raise ValueError('Noncanonical skill ID list')
            fingerprint=record['dictionary_sha256'];fingerprints.add(fingerprint)
            if expected_dictionary_sha256 and fingerprint!=expected_dictionary_sha256:raise ValueError('Dictionary mismatch')
            index[eid]=record
    if len(fingerprints)>1:raise ValueError('Mixed dictionary versions in sidecar')
    missing=set(ordered_ids)-index.keys()
    if missing:raise ValueError('Missing sidecar IDs: '+str(sorted(missing)[:5]))
    idfield='idCv' if side=='cv' else 'idVacancy'
    prefix='m9_'+side+'_'
    return [{idfield:eid,prefix+'skill_ids':index[eid]['skill_ids'],prefix+'evidence':index[eid]['evidence'],
             prefix+'dictionary_sha256':index[eid]['dictionary_sha256'],
             prefix+'snapshot_policy':'first_raw_occurrence; separate from cleaned M8 text'} for eid in ordered_ids]
