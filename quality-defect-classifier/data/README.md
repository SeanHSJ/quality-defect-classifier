# Data

The SECOM dataset isn't bundled in this repo — keep raw data out of git.

## How to get it
1. Go to the UCI Machine Learning Repository SECOM page:
   https://archive.ics.uci.edu/dataset/179/secom
2. Click "Download" — you'll get a zip with two files:
   - `secom.data` — the 590 sensor feature values (space-delimited, no header)
   - `secom_labels.data` — pass/fail label (-1 = pass, 1 = fail) plus a timestamp column
3. Extract both files into this `data/` folder.

## Format note
- `secom.data`: 1567 rows × 590 columns, space-delimited, no header row. Contains missing values (represented as `NaN`).
- `secom_labels.data`: 1567 rows × 2 columns — label and timestamp, space-delimited.

Add a `.gitignore` entry so raw data files aren't committed to GitHub — only processed/summary outputs should be.
