# Canonical skill evidence

This package contains the single reused extractor, dictionary provenance, ID-safe alignment and linguistic validation. Ranking is implemented once in `recommender/experiment.py`, using shared helpers in `recommender/shared.py`; the old separate M9 runner was removed.

Both variants remain in `results/final/`: M9 without region uses global retrieval plus a positive hard-skill bonus; M9 + region preserves regional priority. Their weights are selected using development applicants. Missing evidence is unknown, not lack of competence. Soft traits, negation, nonrequirements and resume title/education-only mentions do not contribute to the bonus.

Dictionary v2 has 47 canonical skills and still awaits human review. The original 29-skill dictionary remains only because it is the baseline for the retained linguistic diagnostics. These compare different dictionary versions, not a second recommendation pipeline. See `VALIDATION_GUIDE.md` for that distinct review process.

The complete run command is documented in `docs/IMPLEMENTATION.md`; it handles extraction in the correct place and reuses verified completed outputs when requested. Notebook 06 displays the evidence and diagnostics; notebook 07 displays model comparisons.
