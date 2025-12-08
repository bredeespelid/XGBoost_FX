# XGBoost_FX

This repository contains code and Jupyter notebooks supporting a master’s thesis on forecasting foreign-exchange (FX) levels—specifically EUR/NOK—using machine learning, with an emphasis on Gradient Boosting (XGBoost) and macro-financial covariates.

## Purpose
- Establish robust benchmark models for **EUR/NOK level forecasting**.
- Compare XGBoost specifications using lagged FX history, macro covariates, and panel/exogenous feature sets.
- Provide a fully reproducible workflow for model training, tuning, and evaluation.

## Repository Structure
- `src/` — Jupyter notebooks (feature engineering, modeling, evaluation).
- `data/` — Data files used in the thesis (raw and processed).
- `LICENSE` — MIT License.

## Notebooks
All notebooks are located in `src/`. The list below maps directly to the model families used in the thesis.

### Price-Only Models (PO)
- **XGBPriceMonthly.ipynb** — XGBoost using FX/price history, monthly frequency.  
  Link: [`src/XGBPriceMonthly.ipynb`](src/XGBPriceMonthly.ipynb)

- **XGBPriceQuarterly.ipynb** — XGBoost using FX/price history, quarterly frequency.  
  Link: [`src/XGBPriceQuarterly.ipynb`](src/XGBPriceQuarterly.ipynb)

### Macro-Enriched Models (EN)
- **XGBMacroMonthly.ipynb** — XGBoost with selected macro-financial covariates, monthly frequency.  
  Link: [`src/XGBMacroMonthly.ipynb`](src/XGBMacroMonthly.ipynb)

- **XGBMacroQuarterly.ipynb** — XGBoost with selected macro-financial covariates, quarterly frequency.  
  Link: [`src/XGBMacroQuarterly.ipynb`](src/XGBMacroQuarterly.ipynb)

### Panel / Exogenous-Extended Models (EN-EX)
- **XGBPanelMonthly.ipynb** — XGBoost with panel/exogenous extensions, monthly frequency.  
  Link: [`src/XGBPanelMonthly.ipynb`](src/XGBPanelMonthly.ipynb)

- **XGBPanelQuarterly.ipynb** — XGBoost with panel/exogenous extensions, quarterly frequency.  
  Link: [`src/XGBPanelQuarterly.ipynb`](src/XGBPanelQuarterly.ipynb)

### Variable Importance (LOFO)
- **XGBoostVariableImportance.ipynb** — LOFO-style variable-importance analysis for covariate selection.  
  Link: [`src/XGBoostVariableImportance.ipynb`](src/XGBoostVariableImportance.ipynb)

## Status
Supporting research material for a master’s thesis. The repository is intended for academic and benchmarking use.

## License
Released under the MIT License. See `LICENSE` for details.

## Citation
If you use this repository in academic work, please cite the author and the associated thesis appropriately.
