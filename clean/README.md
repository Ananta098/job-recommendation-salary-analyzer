# Prepared data

Everything here stays local and is ignored by Git.

- `d4_clean.csv` is written by notebook 02 (Stage 1) or by `recommender.pipeline` (inside its run folder). The shared cleaning implementation is `recommender/cleaning.py`.
- `sbert_{jobs,val,test}_paraphrase-multilingual-MiniLM-L12-v2.npy` are the original Sentence-BERT embedding caches (43,756 job vectors, 1,000 validation and 2,806 test resume vectors, 384 dimensions). Notebook 04 reuses them when their row counts match the clean file. Their SHA-256 values match `results/final/run_manifest.json`, which lets `python -m recommender.pipeline run --reuse-prepared` use them.

Fresh pipeline runs record encoder revision, input hashes and ordered entity hashes. The original caches have no historical ID manifest; reuse is permitted only after verifying the recorded input hashes and raw-data hash. Never reuse them after changing text or entity order. See `docs/PROJECT_HANDBOOK.md`.
