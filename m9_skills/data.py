"""D4 readers: preserve raw punctuation, ignore labels, first raw entity snapshot."""
import csv
import json
from pathlib import Path

SKILL_FIELDS = {
    'cv': ('skills_cv', 'hardSkills_cv', 'softSkills_cv'),
    'vacancy': ('skills_vacancy', 'hardSkills_vacancy', 'softSkills_vacancy')}

def parse_skills(value):
    try:
        items = json.loads(value)
    except (TypeError, ValueError):
        return [], 'invalid_json'
    if not isinstance(items, list):
        return [], 'not_list'
    valid = [x.strip() for x in items if isinstance(x, str) and x.strip()]
    return valid, 'mixed_types' if any(not isinstance(x, str) for x in items) else 'ok'

def records(value, keys):
    try:
        items = json.loads(value)
    except (TypeError, ValueError):
        return ''
    if not isinstance(items, list):
        return ''
    return '\n'.join(str(x[k]) for x in items if isinstance(x, dict) for k in keys if x.get(k))

def entities(raw_path, side, allowed_ids=None):
    if side not in SKILL_FIELDS:
        raise ValueError(side)
    seen = set()
    with Path(raw_path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='|')
        idfield = 'idCv' if side=='cv' else 'idVacancy'
        required = {idfield, *SKILL_FIELDS[side]}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing raw D4 columns: ' + str(required-set(reader.fieldnames or [])))
        for row in reader:
            eid = row[idfield]
            if not eid or eid in seen or (allowed_ids is not None and eid not in allowed_ids):
                continue
            seen.add(eid)
            structured, errors = {}, {}
            for field in SKILL_FIELDS[side]:
                structured[field], status = parse_skills(row[field])
                if status != 'ok': errors[field] = status
            if side=='cv':
                fields = {'positionName':row.get('positionName',''),
                          'workExperienceList':records(row.get('workExperienceList'), ('jobTitle','demands')),
                          'educationList':records(row.get('educationList'), ('speciality','qualification'))}
            else:
                fields = {k:row.get(k,'') for k in ('vacancyName','responsibilities','positionRequirements','additionalRequirements')}
            fields.update({k:'; '.join(v) for k,v in structured.items()})
            yield {'entity_id':eid, 'side':side, 'fields':fields, 'structured':structured, 'parse_errors':errors}
