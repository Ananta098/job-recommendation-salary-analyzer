# Final model verdict

Authoritative results: `results/final/`, rerun through the consolidated pipeline under the locked Python 3.12 environment. Historical original baselines are separated in `results/baselines/`; they are not mixed into this comparison.

## Decision

Serve **M8 Hybrid + region**. None of the five candidates passes the conservative development-only promotion check on the primary metric, grouped Recall@10. The existing M8 remains a defensible prototype choice. This does not mean M8 wins every ranking measure.

| Model | Development grouped R@10 | Test grouped R@10 | Test strict R@10 |
|---|---:|---:|---:|
| M8 | 0.4425 | 0.466031 | 0.274941 |
| M9 without region | 0.1135 | 0.131213 | 0.044449 |
| M9 + region | 0.4445 | 0.464071 | 0.273634 |
| Logistic Regression | 0.4345 | 0.465615 | 0.284254 |
| Random Forest | 0.4400 | 0.464962 | 0.280167 |
| XGBoost | 0.4395 | 0.465972 | 0.284284 |
| M5 | comparator | 0.455803 | 0.266221 |

## Interpretation and limitations by method

- **M8:** combines standardized lexical and semantic similarity with regional priority. It is the established reference, but has encoder/caching complexity, first-400-character semantic inputs and a region policy that does not model relocation/remote preferences.
- **M9 without region:** skill-aware global retrieval performs much worse. Recognized skill overlap cannot compensate for losing the strong location signal. Its development-selected bonus weight is 2.0.
- **M9 + region:** the bonus weight is 1.0. The small development gain (0.4445 versus 0.4425) has a simultaneous paired-bootstrap interval covering zero, and the test primary metric is lower than M8. This attempt is retained without claiming statistically established inferiority.
- **Logistic Regression:** interpretable linear reranking improves strict Recall@10 and NDCG in this test comparison, but does not improve development grouped recall. It is a useful future candidate if exact-posting retrieval becomes the main objective.
- **Random Forest:** models feature interactions but adds model complexity and is sensitive to incomplete labels, sampled negatives and candidate coverage. It does not demonstrate a development primary-metric gain.
- **XGBoost:** also improves some rank/strict measures but has no demonstrated primary-metric improvement sufficient for promotion. Small runtime-dependent score changes are not evidence of general superiority.
- **M5:** simpler lexical fallback candidate; its primary recall is below M8. Historical title/popularity/BM25 baselines in notebook 03 explain the progression to M8 without adding a second training pipeline.

## Protocol

10,227 fitting applicants, 1,000 development applicants and 2,806 test applicants are disjoint under seed 42. The catalog has 43,756 vacancies. The learned rerankers retain all retrieved invitation-group positives and up to 30 sampled other candidates per training query, with equal query weight. Other groups are weak negatives, not verified irrelevant jobs. No gold vacancies are injected into candidate pools. Hyperparameters and input/source hashes are recorded in the run manifest.

The deployment safeguard promotes a candidate only if the lower paired-development bootstrap bound is positive after correction across five comparisons; otherwise M8 is retained. It was added after exploratory analysis and is not preregistered. Historical test applicants have been inspected in earlier project iterations, so these are exploratory project results, not an untouched external evaluation.

The final analysis uses **one paired permutation significance procedure**, with Holm correction, plus paired-bootstrap confidence intervals for uncertainty. It does not repeat a second Wilcoxon significance test. Regional counts/performance describe coverage and variation; regions with fewer than 20 test applicants are flagged.

## Limits on claims

Invitation-group labels are incomplete. Grouped title/region matches can differ in employer, salary and requirements. Evaluation covers applicants with invitations and does not establish future-vacancy quality. The cleaned snapshot policy is outcome-dependent and is not chronological. Dictionary v2 has 47 agent-reviewed skills with human review pending; negation and aspiration errors remain. Missing skill evidence is unknown, not missing competence, and learned scores are not hiring probabilities.

The final rerun uses the same evaluated M8 configuration and reproduces its reference primary score. Small learned-reranker variations between numerical environments are why the latest locked-runtime results, rather than mixed historical tables, are authoritative.

See `docs/IMPLEMENTATION.md` for the single execution procedure and `docs/FINAL_REVIEW.md` for verification and external deployment limits.
