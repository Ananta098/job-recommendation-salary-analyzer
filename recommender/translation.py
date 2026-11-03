"""Offline English interface adapter. Never changes the evaluated M8 model."""
import argparse
import hashlib
import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from .bundle import project_path

MODELS = {
    'en-ru': ('Helsinki-NLP/opus-mt-en-ru', 'bb09c99d180016eac6819df3dae68edb1690fdee'),
    'ru-en': ('Helsinki-NLP/opus-mt-ru-en', 'fbd6dc73284f95536648512cc21d57f19191961a'),
}
CYRILLIC = re.compile('[А-Яа-яЁё]')


def translation_path(path=None):
    return project_path(path or os.environ.get('TRANSLATION_BUNDLE') or 'artifacts/translation')


def check_translation(path=None, deep=False):
    root = translation_path(path)
    if not (root/'manifest.json').is_file():
        raise FileNotFoundError(f'English translation assets missing at {root}. Run '
                                '`python -m recommender.translation prepare` once with Internet access, '
                                'or restore the complete artifacts/translation folder.')
    manifest = json.loads((root/'manifest.json').read_text())
    required = {'regions.json'}
    for direction in MODELS:
        required.update(f'{direction}/{name}' for name in
                        ['model.safetensors', 'config.json', 'source.spm', 'target.spm', 'vocab.json', 'tokenizer_config.json'])
    if not required.issubset(manifest):
        raise ValueError('Translation manifest lacks required files')
    for relative, expected in manifest.items():
        file = (root/relative).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError('Translation artifact missing or unsafe: '+relative)
        if deep:
            with file.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest() != expected:
                    raise ValueError('Translation checksum mismatch: '+relative)
    return root


class EnglishAdapter:
    def __init__(self, path=None, verify=True):
        self.root = check_translation(path, deep=True) if verify else translation_path(path)
        self.lock = threading.Lock()
        self.models = {}
        self.regions = json.loads((self.root/'regions.json').read_text()) if (self.root/'regions.json').exists() else {}

    def _load(self, direction):
        if direction not in self.models:
            import torch
            torch.set_num_threads(4)
            from transformers import MarianMTModel, MarianTokenizer
            folder = self.root/direction
            tokenizer = MarianTokenizer.from_pretrained(str(folder),local_files_only=True)
            model = MarianMTModel.from_pretrained(str(folder),local_files_only=True).eval()
            self.models[direction] = (tokenizer, model)
        return self.models[direction]

    def translate(self, texts, direction):
        """Token chunks preserve long inputs instead of silently truncating them."""
        import torch
        values = [str(value or '').strip() for value in texts]
        output = values.copy()
        with self.lock:
            tokenizer, model = self._load(direction)
            chunks, owners = [], []
            for i,value in enumerate(values):
                if not value or (direction == 'ru-en' and not CYRILLIC.search(value)):
                    continue
                tokens = tokenizer.encode(value,add_special_tokens=False, truncation=False)
                # Smaller chunks stay safely inside Marian's positional limit.
                for start in range(0,len(tokens),240):
                    chunks.append(tokenizer.decode(tokens[start:start+240],skip_special_tokens=True))
                    owners.append(i)
            translated = {i:[] for i in owners}
            for start in range(0,len(chunks),8):
                inputs = tokenizer(chunks[start:start+8],return_tensors='pt',padding=True,truncation=False)
                with torch.inference_mode():
                    generated = model.generate(**inputs,max_new_tokens=480,num_beams=2)
                for i,value in zip(owners[start:start+8],tokenizer.batch_decode(generated,skip_special_tokens=True)):
                    translated[i].append(value)
            for i,parts in translated.items():
                output[i] = ' '.join(parts)
        return output

    def profile(self, title, history, education, skills):
        # Profile content is not cached or written to disk.
        if not title.strip() and not history.strip():
            raise ValueError('Enter a desired title or work history.')
        return self.translate([title,history,education,skills],'en-ru')

    def region_labels(self, regions):
        missing = [r for r in regions if r not in self.regions]
        if missing:
            self.regions.update(zip(missing,self.translate(missing,'ru-en')))
        return {r:self.regions[r] for r in regions}

    @lru_cache(maxsize=256)
    def _vacancy_text(self, text):
        # Only public catalog text is cached, never an applicant profile.
        return self.translate([text],'ru-en')[0]

    def results(self, frame):
        """Return an English display/download table without Russian source columns."""
        import pandas as pd
        fields = [('vacancyName','Job title'),('regionName','Region'),
                  ('job_text','Description'),('job_skills','Skills'),
                  ('professionalSphereName','Professional field')]
        out = pd.DataFrame({'Rank':frame['rank'].tolist()})
        # Batch distinct uncached strings to keep output translation practical on CPU.
        texts = list(dict.fromkeys(str(v) for source,_ in fields if source in frame for v in frame[source].fillna('')))
        translated = dict(zip(texts,self.translate(texts,'ru-en')))
        for source,label in fields:
            out[label] = [translated[str(v)] for v in frame[source].fillna('')] if source in frame else ''
        out['Listed minimum salary (RUB)'] = frame['salaryMin_vacancy'].tolist()
        def experience(value):
            try:
                number=float(value)
                return f'{number:g} years' if number>0 else 'Not specified'
            except (ValueError,TypeError):
                return self._vacancy_text(str(value)) if value else 'Not specified'
        out['Required experience'] = [experience(v) for v in frame['experienceRequirements']]
        out['Published'] = frame['datePublished'].tolist() if 'datePublished' in frame else ''
        out['Vacancy reference'] = frame['idVacancy'].tolist()
        return out


def prepare(path=None):
    """One-time pinned downloads; app inference is strictly local/offline."""
    from huggingface_hub import snapshot_download
    import torch
    torch.set_num_threads(4)
    from transformers import MarianMTModel, MarianTokenizer
    root = translation_path(path)
    root.mkdir(parents=True,exist_ok=True)
    for direction,(repo,revision) in MODELS.items():
        folder=root/direction
        if not (folder/'model.safetensors').is_file():
            snapshot_download(repo,revision=revision,local_dir=folder,
                              allow_patterns=['*.json','*.spm','pytorch_model.bin','README.md'])
            model=MarianMTModel.from_pretrained(str(folder),local_files_only=True)
            tokenizer=MarianTokenizer.from_pretrained(str(folder),local_files_only=True)
            model.save_pretrained(folder,safe_serialization=True)
            tokenizer.save_pretrained(folder)
            (folder/'pytorch_model.bin').unlink(missing_ok=True)
            del model,tokenizer
    # Resolve available catalog regions once; selected original region keys stay intact.
    import pandas as pd
    from .bundle import resolve_bundle
    adapter=EnglishAdapter(root,verify=False)
    catalog=resolve_bundle()/'catalog.csv'
    regions=sorted(pd.read_csv(catalog,usecols=['regionName']).regionName.dropna().unique()) if catalog.exists() else []
    (root/'regions.json').write_text(json.dumps(adapter.region_labels(regions),ensure_ascii=False,indent=2)+'\n')
    (root/'provenance.json').write_text(json.dumps({'models':MODELS,'purpose':'English UI adapter; M8 scoring and evaluation unchanged'},indent=2)+'\n')
    manifest={}
    for file in sorted(root.rglob('*')):
        if not file.is_file() or '.cache' in file.parts or file.name=='manifest.json':continue
        with file.open('rb') as stream:manifest[str(file.relative_to(root))]=hashlib.file_digest(stream,'sha256').hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    check_translation(root,deep=True)
    print('English translation bundle ready:',root,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','check'])
    parser.add_argument('--path');parser.add_argument('--deep',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.path)
    else:print('English translation bundle ready:',check_translation(args.path,args.deep))

if __name__=='__main__':main()
