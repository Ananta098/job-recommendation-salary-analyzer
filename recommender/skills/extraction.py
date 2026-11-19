"""Conservative Russian alias and phrase matching with auditable evidence."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def normalize(text):
    # One-to-one character changes preserve source offsets. No punctuation deletion:
    # C++, C#, .NET and 1С must not collapse into unrelated terms.
    lowered = ''.join(c.lower() if len(c.lower()) == 1 else c for c in str(text))
    return lowered.replace('ё', 'е').replace('\u00a0', ' ')

def load_dictionary(path=None):
    data = json.loads(Path(path or ROOT / 'dictionary.json').read_text(encoding='utf-8'))
    ids, aliases = set(), {}
    for skill in data['skills']:
        sid = skill['id']
        if sid in ids:
            raise ValueError('Duplicate canonical ID: ' + sid)
        ids.add(sid)
        if not skill.get('source') or not skill.get('review'):
            raise ValueError('Missing provenance/review: ' + sid)
        for alias in skill['aliases']:
            key = ' '.join(normalize(alias).split())
            if not key:
                raise ValueError('Empty alias')
            if key in aliases and aliases[key] != sid:
                raise ValueError('Ambiguous alias: ' + alias)
            aliases[key] = sid
        mapping = skill.get('esco')
        if mapping is not None:
            required = ['uri', 'preferred_label', 'version', 'source_url', 'reviewer', 'reviewed_at', 'evidence']
            if mapping.get('status') != 'verified' or not all(mapping.get(k) for k in required):
                raise ValueError('Unverified ESCO mapping: ' + sid)
            if not mapping['uri'].startswith('http://data.europa.eu/esco/skill/'):
                raise ValueError('Not an ESCO skill URI')
    return data

class Extractor:
    def __init__(self, path=None):
        self.dictionary = load_dictionary(path)
        self.version = self.dictionary['version']
        self.fingerprint = hashlib.sha256(json.dumps(self.dictionary, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.patterns = []
        for s in self.dictionary['skills']:
            for alias in s['aliases']:
                phrase = r'\s+'.join(re.escape(x) for x in normalize(alias).split())
                # Also block punctuation immediately adjacent to short technical tokens.
                self.patterns.append((re.compile(r'(?<![\w+#.])' + phrase + r'(?![\w+#])'), s, alias, ' '.join(normalize(alias).split())))

    def extract(self, text, field='text', structured=False):
        if not isinstance(text, str) or not text.strip():
            return []
        norm = normalize(text)
        candidates = []
        compact = ' '.join(norm.split())
        for pattern, skill, alias, alias_key in self.patterns:
            if alias_key not in compact:
                continue
            for m in pattern.finditer(norm):
                # Do not match across sentence/semicolon boundaries. Keep mention and
                # assertion separate: absence of a skill mention is never incompetence.
                left = re.split(r'[.!?;\n]', norm[:m.start()])[-1][-110:]
                right = re.split(r'[.!?;\n]', norm[m.end():])[0][:65]
                negative = bool(re.search(r'(?:не\s+(?:владею|владеет|знаю|знает|умею)|без\s+(?:знания|знаний|опыта)(?:\s+работы)?(?:\s+с)?|нет\s+(?:знаний|опыта)(?:\s+работы)?(?:\s+с)?)\s*$', left))
                not_required = bool(re.search(r'(?:не\s+требуется|не\s+обязательно)\s+(?:знание\s+|владение\s+)?$', left) or re.match(r'\s*(?:не\s+(?:требуется|требуются|обязателен|обязательна|обязательно))', right))
                optional = bool(re.search(r'(?:желательно|приветствуется|будет\s+плюсом)\s*$', left) or re.match(r'\s*(?:желательно|приветствуется|будет\s+плюсом)', right))
                required = bool(re.search(r'(?:обязательно|требуется|необходимо|знание|владение|умение)\s*$', left))
                status = 'negated' if negative else 'not_required' if not_required else 'mentioned'
                requirement = ('not_required' if not_required else 'unspecified' if negative
                               else 'optional' if optional else 'required' if required else 'unspecified')
                candidates.append(dict(skill_id=skill['id'], canonical=skill['canonical'], alias=alias,
                    matched=text[m.start():m.end()], start=m.start(), end=m.end(), field=field,
                    source='structured' if structured else 'phrase_rule', assertion=status,
                    requirement=requirement, esco=skill.get('esco')))
        # Prefer the longest phrase where aliases overlap; retain distinct occurrences.
        accepted = []
        for item in sorted(candidates, key=lambda x: (-(x['end']-x['start']), x['start'], x['skill_id'])):
            if not any(item['start'] < a['end'] and a['start'] < item['end'] for a in accepted):
                accepted.append(item)
        return sorted(accepted, key=lambda x: (x['start'], x['skill_id']))

    def extract_fields(self, fields, structured_fields=()):
        evidence = [e for field, value in fields.items() for e in self.extract(value, field, field in structured_fields)]
        return {'skill_ids': sorted({e['skill_id'] for e in evidence if e['assertion']=='mentioned'}),
                'evidence': evidence, 'dictionary_version': self.version,
                'dictionary_sha256': self.fingerprint}
