# Prepared data

`recommender.pipeline` creates the clean CSV and the three evaluation embedding caches. They stay local and are ignored by Git. The shared cleaning implementation is in `recommender/cleaning.py`; notebook 02 displays its report. Notebook 04 documents the fixed encoder configuration without producing a second cache.

Fresh pipeline runs record encoder revision, input hashes and ordered entity hashes. The original caches have no historical ID manifest; reuse is permitted only after verifying the recorded input hashes and raw-data hash. Never reuse them after changing text or entity order. See `docs/IMPLEMENTATION.md`.
