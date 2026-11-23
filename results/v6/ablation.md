# Ablation: `free_seq_only` vs `free_seq_atac` (split: test)

114 held-out windows. Deltas for per-window metrics use a paired bootstrap (1000 resamples, 95% CI); pooled metrics (boundary F1, E-P AUPRC/AUROC) are single numbers over all pairs and carry no interval.

| metric | free_seq_only | free_seq_atac | Δ | 95% CI |
|---|---|---|---|---|
| map Pearson | 0.2598 | 0.5607 | 0.3009 | [0.2696, 0.3318]** |
| map Spearman | 0.2629 | 0.5233 | 0.2604 | [0.2315, 0.2899]** |
| insulation Pearson | 0.2247 | 0.6538 | 0.4291 | [0.3784, 0.4747]** |
| boundary F1 | 0.0096 | 0.3478 | 0.3382 | — |
| E-P loop AUPRC | 0.1378 | 0.5475 | 0.4096 | — |
| E-P loop AUROC | 0.5845 | 0.8973 | 0.3128 | — |

### Reference points for E–P AUPRC

- distance-only baseline: **0.0890** — a model scoring at or below this has learned distance decay, not looping.
- observed-map ceiling: **1.0000** — scoring pairs with the *true* map, i.e. the best achievable given how labels were defined.
- positive rate: **0.0909** — the AUPRC of random guessing.

*Rows marked ** have a 95% CI excluding zero.*
