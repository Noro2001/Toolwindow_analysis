#!/usr/bin/env bash
set -e
python toolwindow_analysis.py --data data/toolwindow_data.csv --double-open-policy close_previous --min-duration-ms 100 --iqr-multiplier 3 --per-user --survival
