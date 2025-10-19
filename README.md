# Tool Window Usage Analysis

Analyze how long an IDE tool window stays open depending on how it was **opened**: `manual` vs `auto`.

## Project Layout
```
toolwindow-analysis/
├─ data/
│  └─ toolwindow_data.csv                # place your dataset here
├─ figures/                              # generated plots
├─ reports/
│  └─ report.pdf                         # generated PDF report
├─ notebooks/
│  └─ toolwindow_analysis.ipynb          # interactive notebook
├─ tests/
│  └─ synthetic_toolwindow_data.csv      # small synthetic dataset for smoke tests
├─ toolwindow_analysis.py                # main analysis script
├─ requirements.txt                      # dependencies
└─ run.sh                                # example run command
```

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset
CSV columns:
- `user_id` (string)
- `timestamp` (epoch **milliseconds**)
- `event_id` ∈ {`open`, `close`}
- `open_type` ∈ {`manual`, `auto`} — present **only for `open` events** (may be empty otherwise)

Real-world messiness is expected: orphan `close`, consecutive `open` events, and `open` without a matching `close` before the log ends (right-censored).

## Run
```bash
python toolwindow_analysis.py   --data data/toolwindow_data.csv   --double-open-policy close_previous   --min-duration-ms 100   --iqr-multiplier 3   --per-user   --survival
```
Artifacts will appear in `figures/` and `reports/report.pdf`.

## Policies
- `--double-open-policy`:
  - `close_previous`: **close the previous episode at the moment of the new `open`** (recommended)
  - `drop_previous`: drop the previous conflicting `open`
  - `keep_first`: ignore additional `open` until a `close` arrives

## What gets generated
- `figures/*.png`: histograms, ECDF, boxplot, and (optionally) KM curves
- `reports/report.pdf`: consolidated PDF with methods, summary stats, and visuals
- `episodes_clean.csv`: cleaned episodes
- `summary_stats.json`: aggregate statistics
- `per_user_stats.csv`: per-user statistics (if `--per-user`)

## Methods Summary
- Normalize fields and values; enforce `event_id` ∈ {open, close}
- Match `open→close` per user, with a selected policy for consecutive `open`
- Treat `open` without `close` as **right-censored** (kept for survival analysis)
- Cleaning: minimum duration filter + **group-wise** IQR filtering
- Group comparison: Mann–Whitney U; bootstrap 95% CI for mean/median; Cliff’s delta
- Optional per-user analysis: per-user medians + paired Wilcoxon
- Survival analysis: Kaplan–Meier + log-rank test
