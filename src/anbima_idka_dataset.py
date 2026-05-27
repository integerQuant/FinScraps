from __future__ import annotations

from pathlib import Path

import pandas as pd


IDKA_CODES = [
    "IDKAPRE3M",
    "IDKAPRE1A",
    "IDKAPRE2A",
    "IDKAPRE3A",
    "IDKAPRE5A",
    "IDKAIPCA2A",
    "IDKAIPCA3A",
    "IDKAIPCA5A",
    "IDKAIPCA10A",
    "IDKAIPCA15A",
    "IDKAIPCA20A",
    "IDKAIPCA30A",
]
COLUMNS = ["date", *IDKA_CODES]
NUMERIC_COLUMNS = IDKA_CODES
DEFAULT_HF_REPO_ID = "rodrigomtorresb/anbima-idka"
DEFAULT_HF_FILENAME = "latest.parquet"


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the public ANBIMA IDKA dataset without changing its schema."""
    missing_columns = [column for column in COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    normalized = df.loc[:, COLUMNS].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.normalize()

    for column in NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")

    return normalized


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_dataset(df)

    null_counts = normalized.isna().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        raise ValueError(f"Null values found: {null_counts.to_dict()}")

    duplicate_count = int(normalized.duplicated(["date"]).sum())
    if duplicate_count:
        raise ValueError(f"Duplicate date rows found: {duplicate_count}")

    if not normalized["date"].is_monotonic_increasing:
        raise ValueError("Dates must be sorted in ascending order.")

    return normalized.loc[:, COLUMNS]


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    validated = validate_dataset(df)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(output_path, index=False)
