# XGBoost_FX

This repository contains a single Jupyter notebook (`XGBoost_Main.ipynb`) with the full set of **XGBoost FX forecasting experiments** used in a master’s thesis. The notebook implements **monthly and quarterly walk-forward evaluation** for **EUR/NOK level forecasting**, including:
- price-only baselines,
- macro-financial variants (incl. exogenous extensions),
- robust benchmark comparisons (e.g., driftless random walk),
- regime-split performance tables,
- bootstrap robustness checks.

---

## Contents
- `XGBoost_Main.ipynb` — end-to-end workflow: data handling, feature variants, walk-forward evaluation, comparisons, and robustness runs.
- `data/` — raw/processed datasets used by the notebook (if applicable).
- `requirements.txt` — pinned dependencies.
- `LICENSE` — MIT License.

---

## Quick start
### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\Activate.ps1   # Windows (PowerShell)
