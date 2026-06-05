# Forecast Report Runner

This branch contains only the files needed to run `report/forecast_report.ipynb`.

## Contents

- `report/forecast_report.ipynb` - method report and rolling forecast notebook
- `outputs/cache/daily_metric_coverage_panel.parquet` - daily input panel used by the notebook
- `src/clean_candidate_variables.py` - minimal panel loader
- `src/cv_yoy_eval.py` - minimal model factory
- `requirements.txt` - Python package dependencies

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
jupyter notebook report/forecast_report.ipynb
```

The notebook writes generated forecast outputs to `submission/`.
