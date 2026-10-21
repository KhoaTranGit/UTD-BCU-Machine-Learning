import polars as pl
import re

def split_data(data_path):
  exclude_metas = ["row_key", "sample_id", "cohort",
                 "source_file", "evaluation_cohort"]
  data = pl.read_csv(data_path).rename({col:clean_col(col) for col in data.columns})
  ## Gets train set
  train = data.filter(pl.col("cohort") == "train").select(pl.exclude(exclude_metas))
  ## Gets internal validation set (for tuning & model comparison)
  val_internal = data.filter(
      pl.col("cohort") == "validation",
      pl.col("evaluation_cohort") == "internal_validation"  ## sites A,B,C
  ).select(pl.exclude(exclude_metas))
  ## Gets external validation set (unbiased held-out set for evaluating tuned models)
  val_external = data.filter(
      pl.col("cohort") == "validation",
      pl.col("evaluation_cohort") == "external_validation"  ## sites E,D,F,G
  ).select(pl.exclude(exclude_metas))
  ## Gets test set
  test  = data.filter(pl.col("cohort") == "test").select(pl.exclude(exclude_metas))

  return data, train, val_internal, val_external, test



def clean_col(name: str) -> str:
    """
      data = data.rename({col:clean_col(col) for col in data.columns})
    """
    name = name.strip().lower()
    name = name.replace("#", "_abs")
    name = name.replace("%", "_pct")
    name = name.replace("&", "_alt")
    name = name.replace("[", "").replace("]", "")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    # Eliminate duplicate tokens
    name = re.sub(r"(?:_pct){2,}", "_pct", name)
    return name
