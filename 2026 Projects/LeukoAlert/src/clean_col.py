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
