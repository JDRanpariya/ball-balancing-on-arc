# Paper

This directory contains the manuscript source:

| Directory | Publication | Status |
|-----------|-------------|--------|
| `ram/` | arXiv preprint | Public preprint |

---

## arXiv Preprint

**A Ball-on-Arc Benchmark: Classical and Learning-Based Control on Realistic Deployment Hardware**

```bash
cd paper/ram
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Output: `paper/ram/main.pdf` (8 pages) and the supplementary PDF.

### What's inside `ram/`

```
ram/
+-- main.tex                      # Manuscript
+-- bibliography.bib              # 53 entries (20 cited in main text)
+-- sections/                     # 7 section .tex files
+-- figures/                      # 20 figures (4 in main paper, 16 in supplementary)
|   +-- system_setup.jpg          #   High-quality color photograph of platform
|   +-- exp1_boxplot.png
|   +-- hw_sr_ranking.png
|   +-- hw_difficulty_breakdown.png
+-- supplementary/                # Separate supplementary PDF
|   +-- supplementary.tex / supplementary.pdf
|   +-- sections/                 # Appendices A-I
|   +-- figures/                  # Supplementary figures (shared with paper/ram/figures/)
+-- regenerate_all_figures.py     # Canonical entry: rebuilds ALL figures
+-- regenerate_figures.py         # Main-paper figure generation
+-- regenerate_extra_figures.py   # Supplementary figure generation
+-- scripts/                      # verify_supplementary_claims.py, etc.
```

See [`ram/README.md`](ram/README.md) for the complete structure, controller
taxonomy, reproducibility guide, and data-to-paper-section mapping.

---

## Shared Experimental Data

`data/` contains canonical JSON files backing every table and figure in the
arXiv preprint:

```
data/
+-- README.md                         # Data directory map
+-- exp1_hardware/                    # 50-trial HW results (all 13 controllers)
|   +-- consolidated.json             #   144 MB; the master dataset
|   +-- ppo_wm_v1.json / _ft.json     #   PPO-WM hardware evaluation
|   +-- sources/                      #   Per-controller raw JSON
+-- exp1_sim/                         # Simulation predictions (2 sim models)
+-- rl_ablation/                      # Blind-zone, DR, multi-algo, checkpoint sweeps
+-- rl_new4/                          # SAC/TD3/TQC/TRPO (1M/3M, base+DR)
+-- rl_5m_ablation/                   # Extended 5M training ablation
+-- iql_offline_rl/                   # Offline RL results
+-- sim_validation/                   # Sim-to-real validation (.npz)
+-- tuned_params/                     # CMA-ES best parameters per controller
+-- world_model/                      # PPO-WM sim evaluation results
+-- world_model_models/               # Trained PPO-WM checkpoints (.zip)
+-- wm_data_sweep/                    # World-model accuracy vs. dataset size
+-- wm_hardware_results/              # Per-data-size hardware evaluation
```

All code to regenerate figures from these data files lives in
`ram/regenerate_figures.py`, `ram/regenerate_extra_figures.py`, and
`ram/scripts/verify_supplementary_claims.py`.

---

## Other Directories

| Directory | Purpose |
|-----------|---------|
| `scripts/` | Utility scripts (statistical tests, claim verification) |

---

## Full Dataset

The complete 650-trial experimental record, 1M-transition random-exploration
dataset, and >1M-transition demonstration archive are included in the public
repository: <https://github.com/JDRanpariya/ball-balancing-on-arc>.
