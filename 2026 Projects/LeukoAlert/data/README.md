# LeukoAlert: combined cohort data

`leukoalert_all_cohorts.csv.gz` is a gzip-compressed CSV containing **446,663 blood-test records** for leukemia classification. Each row represents a test record, not necessarily a unique patient.

**Source:** [LeukoAlert, Mendeley Data v1](https://doi.org/10.17632/vc7kwnyppz.1), Shilong Liu (2026). **License:** CC BY 4.0. [Study methods](https://doi.org/10.1038/s41746-026-03016-3).

## Columns

- **Target:** `leukemia_label`, with values `leukemia` and `healthy`. The latter is the source label for non-leukemia controls, not a guarantee of overall health.
- **Predictors:** 74 numeric blood-count and analyzer measurements. Units are included where specified in the original headers.
- **Metadata:** `sample_id`, `row_key`, `cohort`, `evaluation_cohort`, `site`, and `source_file`. Exclude these from model predictors. `row_key` uniquely identifies a released row, not a patient.

## Supplied cohorts

Use `evaluation_cohort` to select the following groups:

| Value | Sites | Records | Purpose |
| --- | --- | ---: | --- |
| `train` | A-C | 26,922 | Fit models and preprocessing |
| `internal_validation` | A-C | 211,612 | Compare models and choose settings |
| `external_validation` | D-G | 149,648 | Evaluate the chosen model at other centers |
| `test` | Later cohort from site A | 58,481 | Final evaluation |

Sites are participating clinical centers. The broader `cohort` column groups both validation cohorts under `validation`.

## Load with Polars

No manual decompression is needed:

```python
import polars as pl

data = pl.read_csv("leukoalert_all_cohorts.csv.gz")
train = data.filter(pl.col("evaluation_cohort") == "train")
validation = data.filter(pl.col("evaluation_cohort") == "internal_validation")
```

## Load with pandas

pandas also reads the compressed file directly:

```python
import pandas as pd

data = pd.read_csv("leukoalert_all_cohorts.csv.gz")
train = data.loc[data["evaluation_cohort"] == "train"].copy()
validation = data.loc[data["evaluation_cohort"] == "internal_validation"].copy()
```

## Important notes

- The study reports patient-level separation between training and validation, while retaining repeated tests on different days. Preserve the supplied cohorts; public sample IDs do not allow reliable patient grouping across files.
- Fit additional preprocessing on training data only. Reserve external validation and test cohorts for evaluation after model selection.
- Site files were concatenated with provenance fields added; no additional imputation or resampling was performed. The released values should not be assumed to be raw clinical measurements.
- The downloaded release has 446,663 rows and 74 predictors, differing from the paper's reported 446,558 records and 72 features.
