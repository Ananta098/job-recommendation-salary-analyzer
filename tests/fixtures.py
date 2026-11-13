"""Tiny, self-consistent stand-ins for the private model bundle and translation assets.

The real bundle (about 1.2 GB) is git-ignored, so CI builds these instead:

    python -m tests.fixtures --out artifacts      # writes artifacts/deployment and artifacts/translation

The fixture bundle follows the same file contract and manifest as the real one, so the
integrity checks, the serving code and the Docker image build run unchanged. Its encoder is
a one-layer BERT with random weights: rankings are meaningless, only the plumbing is tested.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ['Москва', 'Тульская область', 'Санкт-Петербург']
JOBS = [
    ('Бухгалтер', 'ведение бухгалтерского учета 1с excel отчетность', '1с excel'),
    ('Главный бухгалтер', 'налоговая отчетность бухгалтерский учет 1с', '1с налоговая отчетность'),
    ('Программист Python', 'разработка на python sql базы данных', 'python sql'),
    ('Водитель', 'перевозка грузов категория в знание правил дорожного движения', 'вождение'),
    ('Повар', 'приготовление блюд санитарные нормы', 'приготовление пищи'),
    ('Продавец-кассир', 'работа на кассе обслуживание покупателей', 'работа с кассой'),
    ('Учитель', 'преподавание планирование уроков работа с родителями', 'преподавание'),
    ('Электромонтер', 'обслуживание электрооборудования техника безопасности', 'электромонтаж'),
]


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_manifest(folder, exclude=('manifest.json',)):
    folder = Path(folder)
    manifest = {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob('*'))
                if p.is_file() and p.name not in exclude}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def build_encoder(target):
    """One-layer BERT (384-dim output, as the serving contract requires), saved as Sentence-Transformers."""
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from sentence_transformers import SentenceTransformer, models
    letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz0123456789'
    words = sorted({w for _, text, skills in JOBS for w in (text + ' ' + skills).split()})
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + list(letters) + ['##' + c for c in letters] + words
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'vocab.txt').write_text('\n'.join(dict.fromkeys(vocab)) + '\n', encoding='utf-8')
        tokenizer = BertTokenizerFast(vocab_file=str(tmp / 'vocab.txt'), do_lower_case=True)
        torch.manual_seed(0)
        config = BertConfig(vocab_size=len(tokenizer), hidden_size=384, num_hidden_layers=1,
                            num_attention_heads=4, intermediate_size=128, max_position_embeddings=128)
        BertModel(config).save_pretrained(tmp / 'hf', safe_serialization=True)
        tokenizer.save_pretrained(tmp / 'hf')
        word = models.Transformer(str(tmp / 'hf'), max_seq_length=128)
        pooling = models.Pooling(word.get_word_embedding_dimension(), 'mean')
        model = SentenceTransformer(modules=[word, pooling], device='cpu')
        model.save(str(target))
    return SentenceTransformer(str(target), device='cpu', local_files_only=True)


def build_bundle(out, with_encoder=True):
    """Write a complete fixture bundle to `out`. Without the encoder only the integrity checks apply."""
    import numpy as np
    import pandas as pd
    import joblib
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from m9_skills.extraction import Extractor
    from recommender.shared import text, evidence_vectors
    from recommender.core import FEATURE_NAMES
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = []
    for i, (title, body, skills) in enumerate(JOBS):
        rows.append({'idVacancy': f'fixture-{i:03d}', 'vacancyName': title, 'vacancyName_en': title,
                     'regionName': REGIONS[i % len(REGIONS)], 'job_text': body, 'job_skills': skills,
                     'salaryMin_vacancy': 30000 + 5000 * i, 'experienceRequirements': i % 4,
                     'datePublished': '2023-04-0%d' % (1 + i % 9), 'professionalSphereName': 'Тест'})
    jobs = pd.DataFrame(rows)
    jobs.to_csv(out / 'catalog.csv', index=False)
    dictionary = ROOT / 'm9_skills' / 'dictionary.v2.json'
    shutil.copyfile(dictionary, out / 'dictionary.json')
    extractor = Extractor(out / 'dictionary.json')
    records = [{'m9_vacancy_evidence': extractor.extract_fields(
        {'responsibilities': r['job_text'], 'skills_vacancy': r['job_skills']},
        ['skills_vacancy'])['evidence']} for r in rows]
    np.save(out / 'job_skills.npy', evidence_vectors(records, extractor.dictionary, 'vacancy'))
    tfidf = TfidfVectorizer(min_df=1, sublinear_tf=True, dtype=np.float32)
    sparse.save_npz(out / 'documents.npz', tfidf.fit_transform(text(jobs, 'vacancy')))
    joblib.dump(tfidf, out / 'tfidf.joblib')
    if with_encoder:
        encoder = build_encoder(out / 'encoder')
        embeddings = encoder.encode(jobs.job_text.str[:400].tolist(), normalize_embeddings=True,
                                    convert_to_numpy=True).astype(np.float32)
    else:
        (out / 'encoder').mkdir()
        for name in ['model.safetensors', 'config.json', 'modules.json', 'tokenizer.json']:
            (out / 'encoder' / name).write_text('{}' if name.endswith('.json') else 'placeholder', encoding='utf-8')
        rng = np.random.default_rng(0)
        embeddings = rng.normal(size=(len(jobs), 384)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.save(out / 'job_embeddings.npy', embeddings)
    metadata = {'format_version': 1, 'selected_model': 'M8', 'kind': 'M8', 'use_region': True,
                'features': FEATURE_NAMES, 'candidate_count': 300, 'semantic_max_chars': 400,
                'lexical_weight': .5, 'seed': 42, 'vacancies': len(jobs), 'embedding_dimension': 384,
                'encoder_revision': 'fixture',
                'catalog_ids_sha256': hashlib.sha256('\n'.join(jobs.idVacancy).encode()).hexdigest(),
                'dictionary_sha256': extractor.fingerprint, 'fixture': True}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    write_manifest(out)
    return out


def build_translation(out):
    """Files with the names the translation check requires. They are not loadable Marian models."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    for direction in ['en-ru', 'ru-en']:
        folder = out / direction
        folder.mkdir(parents=True)
        for name in ['model.safetensors', 'config.json', 'source.spm', 'target.spm', 'vocab.json', 'tokenizer_config.json']:
            (folder / name).write_text('fixture ' + direction + ' ' + name, encoding='utf-8')
    (out / 'regions.json').write_text(json.dumps({r: r for r in REGIONS}, ensure_ascii=False) + '\n', encoding='utf-8')
    write_manifest(out)
    return out


def is_real(folder):
    """True for a real (non-fixture) bundle or translation folder, which must never be overwritten."""
    folder = Path(folder)
    if not (folder / 'manifest.json').exists():
        return False
    if (folder / 'metadata.json').exists():
        return not json.loads((folder / 'metadata.json').read_text(encoding='utf-8')).get('fixture')
    return (folder / 'provenance.json').exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True, help='Folder that receives deployment/ and translation/')
    args = parser.parse_args()
    for name in ['deployment', 'translation']:
        if is_real(args.out / name):
            raise SystemExit(f'{args.out / name} holds a real bundle; refusing to overwrite it with a fixture.')
    build_bundle(args.out / 'deployment')
    build_translation(args.out / 'translation')
    print('Fixture bundle and translation assets written to', args.out)


if __name__ == '__main__':
    main()
