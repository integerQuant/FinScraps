from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


COLUMNS = ["date", "type", "b1", "b2", "b3", "b4", "l1", "l2"]
NUMERIC_COLUMNS = ["b1", "b2", "b3", "b4", "l1", "l2"]
ALLOWED_TYPES = {"pre", "ipca"}
DEFAULT_HF_REPO_ID = "rodrigomtorresb/anbima-irts"
DEFAULT_HF_FILENAME = "latest.parquet"
DUPLICATE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CoverageReport:
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    rows: int
    unique_dates: int
    duplicate_keys: int
    missing_dates: list[str]
    extra_dates: list[str]


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the public ANBIMA IRTS dataset without changing its schema."""
    missing_columns = [column for column in COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    normalized = df.loc[:, COLUMNS].copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized["type"] = normalized["type"].astype(str).str.lower()

    for column in NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")

    return normalized


def collapse_duplicate_keys(
    df: pd.DataFrame,
    tolerance: float = DUPLICATE_TOLERANCE,
) -> pd.DataFrame:
    """Collapse duplicate date/type keys only when numeric differences are tiny."""
    normalized = normalize_dataset(df)
    duplicate_rows = normalized[normalized.duplicated(["date", "type"], keep=False)]

    for key, group in duplicate_rows.groupby(["date", "type"], sort=False):
        deltas = group[NUMERIC_COLUMNS].max() - group[NUMERIC_COLUMNS].min()
        if (deltas.abs() > tolerance).any():
            raise ValueError(
                "Duplicate key has conflicting numeric values: "
                f"date={key[0].date()}, type={key[1]}, max_deltas={deltas.to_dict()}"
            )

    deduped = normalized.drop_duplicates(["date", "type"], keep="first")
    return sort_dataset(deduped)


def sort_dataset(df: pd.DataFrame) -> pd.DataFrame:
    type_order = pd.CategoricalDtype(categories=["ipca", "pre"], ordered=True)
    sorted_df = df.copy()
    sorted_df["type"] = sorted_df["type"].astype(type_order)
    sorted_df = sorted_df.sort_values(["date", "type"]).reset_index(drop=True)
    sorted_df["type"] = sorted_df["type"].astype(str)
    return sorted_df.loc[:, COLUMNS]


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_dataset(df)

    invalid_types = sorted(set(normalized["type"].dropna()) - ALLOWED_TYPES)
    if invalid_types:
        raise ValueError(f"Invalid type values: {invalid_types}")

    null_counts = normalized.isna().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        raise ValueError(f"Null values found: {null_counts.to_dict()}")

    duplicate_count = int(normalized.duplicated(["date", "type"]).sum())
    if duplicate_count:
        raise ValueError(f"Duplicate date/type rows found: {duplicate_count}")

    rows_per_date = normalized.groupby("date").size()
    bad_dates = rows_per_date[rows_per_date != 2]
    if not bad_dates.empty:
        formatted = {date.strftime("%Y-%m-%d"): int(count) for date, count in bad_dates.items()}
        raise ValueError(f"Expected exactly two rows per date: {formatted}")

    return sort_dataset(normalized)


def merge_new_rows(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    existing = validate_dataset(existing_df) if not existing_df.empty else pd.DataFrame(columns=COLUMNS)
    new = validate_dataset(new_df)

    existing_keys = set(_keys(existing))
    new_rows = new[[key not in existing_keys for key in _keys(new)]]
    if new_rows.empty:
        return existing, 0

    merged = pd.concat([existing, new_rows], ignore_index=True)
    return validate_dataset(merged), len(new_rows)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    validated = validate_dataset(df)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(output_path, index=False)


def coverage_report(
    df: pd.DataFrame,
    holiday_dates: Iterable[pd.Timestamp] = (),
) -> CoverageReport:
    normalized = collapse_duplicate_keys(df)
    actual_dates = pd.DatetimeIndex(normalized["date"].drop_duplicates()).sort_values()
    if actual_dates.empty:
        return CoverageReport(None, None, 0, 0, 0, [], [])

    holidays = pd.DatetimeIndex(pd.to_datetime(list(holiday_dates))).normalize()
    expected_dates = pd.bdate_range(actual_dates.min(), actual_dates.max()).difference(holidays)
    missing_dates = [date.strftime("%Y-%m-%d") for date in expected_dates.difference(actual_dates)]
    extra_dates = [date.strftime("%Y-%m-%d") for date in actual_dates.difference(expected_dates)]

    return CoverageReport(
        start_date=actual_dates.min(),
        end_date=actual_dates.max(),
        rows=len(normalized),
        unique_dates=len(actual_dates),
        duplicate_keys=int(df.duplicated(["date", "type"]).sum()),
        missing_dates=missing_dates,
        extra_dates=extra_dates,
    )


def _keys(df: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
    return list(zip(df["date"], df["type"]))
