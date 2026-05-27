from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.anbima_irts_dataset import DEFAULT_HF_FILENAME, DEFAULT_HF_REPO_ID, write_parquet


def load_latest_dataset(
    repo_id: str = DEFAULT_HF_REPO_ID,
    filename: str = DEFAULT_HF_FILENAME,
    token: str | None = None,
) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token or os.environ.get("HF_TOKEN"),
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return pd.DataFrame()

    return pd.read_parquet(path)


def upload_latest_dataset(
    df: pd.DataFrame,
    repo_id: str = DEFAULT_HF_REPO_ID,
    filename: str = DEFAULT_HF_FILENAME,
    token: str | None = None,
    workdir: str | Path = "dist",
    writer=write_parquet,
) -> None:
    from huggingface_hub import HfApi

    output_path = Path(workdir) / filename
    writer(df, output_path)

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update {filename}",
    )
