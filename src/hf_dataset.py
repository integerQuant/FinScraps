from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import logging
import os
from pathlib import Path
import random
import time
from typing import Callable, TypeVar

import pandas as pd

from src.anbima_irts_dataset import DEFAULT_HF_FILENAME, DEFAULT_HF_REPO_ID, write_parquet


LOGGER = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_RETRY_SECONDS = 120
DEFAULT_MAX_SLEEP_SECONDS = 30
DEFAULT_INITIAL_DELAY_SECONDS = 2
T = TypeVar("T")


def _call_with_retries(
    operation: Callable[[], T],
    description: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_retry_seconds: float = DEFAULT_MAX_RETRY_SECONDS,
    max_sleep_seconds: float = DEFAULT_MAX_SLEEP_SECONDS,
    initial_delay_seconds: float = DEFAULT_INITIAL_DELAY_SECONDS,
) -> T:
    start = time.monotonic()
    attempt = 0

    while True:
        try:
            return operation()
        except Exception as error:
            if not _is_retryable_hf_error(error) or attempt >= max_retries:
                raise

            delay = _retry_delay_seconds(
                error,
                attempt,
                max_sleep_seconds=max_sleep_seconds,
                initial_delay_seconds=initial_delay_seconds,
            )
            elapsed = time.monotonic() - start
            if delay > max_sleep_seconds or elapsed + delay > max_retry_seconds:
                LOGGER.warning(
                    "Not retrying %s after transient Hugging Face error; "
                    "delay %.1fs exceeds retry budget.",
                    description,
                    delay,
                )
                raise

            LOGGER.warning(
                "Retrying %s in %.1fs after transient Hugging Face error.",
                description,
                delay,
            )
            time.sleep(delay)
            attempt += 1


def _is_retryable_hf_error(error: Exception) -> bool:
    status_code = _status_code(error)
    if status_code is not None:
        return status_code in RETRYABLE_STATUS_CODES

    module = error.__class__.__module__.split(".", 1)[0]
    return module in {"httpx", "requests"}


def _retry_delay_seconds(
    error: Exception,
    attempt: int,
    max_sleep_seconds: float,
    initial_delay_seconds: float,
) -> float:
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        return retry_after

    delay = initial_delay_seconds * (2 ** attempt)
    jitter = random.uniform(0, min(1.0, delay * 0.1))
    return min(max_sleep_seconds, delay + jitter)


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    value = headers.get("Retry-After")
    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _status_code(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def load_latest_dataset(
    repo_id: str = DEFAULT_HF_REPO_ID,
    filename: str = DEFAULT_HF_FILENAME,
    token: str | None = None,
) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    try:
        path = _call_with_retries(
            lambda: hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token or os.environ.get("HF_TOKEN"),
            ),
            f"download {repo_id}/{filename}",
        )
    except LocalEntryNotFoundError:
        raise
    except RepositoryNotFoundError as error:
        if _status_code(error) not in (None, 404):
            raise
        return pd.DataFrame()
    except EntryNotFoundError:
        return pd.DataFrame()

    return pd.read_parquet(path)


def upload_latest_dataset(
    df: pd.DataFrame,
    repo_id: str = DEFAULT_HF_REPO_ID,
    filename: str = DEFAULT_HF_FILENAME,
    token: str | None = None,
    workdir: str | Path = "dist",
    writer=write_parquet,
    create_repo: bool = False,
) -> None:
    from huggingface_hub import HfApi

    output_path = Path(workdir) / filename
    writer(df, output_path)

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    if create_repo:
        _call_with_retries(
            lambda: api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True),
            f"create Hugging Face dataset repo {repo_id}",
        )
    _call_with_retries(
        lambda: api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update {filename}",
        ),
        f"upload {repo_id}/{filename}",
    )
