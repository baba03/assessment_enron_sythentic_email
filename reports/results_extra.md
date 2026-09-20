| Arm | Embedding cos (pair / random) | Own-source top-1 / top-5 | Form-classifier AUC raw → **styled** ↓ | Source names surviving (e-mails) ↓ | Sender in sign-off | Diversity distinct-2 |
|---|---|---|---|---|---|---|
| *Original Enron (reference)* | - | - | 0.50 by definition | - | 46% | 0.765 |
| Hybrid v1 | 0.79 / 0.61 | 88% / 96% | 0.86 → **0.68** | 5.8% (46 names) | 41% | 0.801 |
| Hybrid v2 | 0.80 / 0.61 | 89% / 98% | 0.89 → **0.69** | 4.2% (21 names) | 44% | 0.807 |
| Rule-based, scrubber as of v2 | 0.90 / 0.62 | 100% / 100% | 0.77 | 5.0% (22 names) | 44% | 0.777 |
| Rule-based, final scrubber | 0.90 / 0.62 | 100% / 100% | 0.77 | 2.5% (3 names) | 41% | 0.778 |
| LLM only (naive) | 0.83 / 0.61 | 92% / 97% | 0.96 → **0.96** | 3.3% (4 names) | 0% | 0.723 |
| Llama-3.3-70B | 0.80 / 0.61 | 92% / 98% | 0.83 → **0.71** | 5.0% (44 names) | 32% | 0.742 |
| Qwen3-235B | 0.83 / 0.61 | 89% / 95% | 0.83 → **0.64** | 4.2% (44 names) | 50% | 0.811 |
| Gemini-2.5-Flash | 0.82 / 0.61 | 92% / 96% | 0.87 → **0.67** | 5.8% (46 names) | 37% | 0.787 |
| Claude Haiku 4.5 | 0.82 / 0.61 | 90% / 98% | 0.82 → **0.65** | 5.8% (46 names) | 30% | 0.786 |
