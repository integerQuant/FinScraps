from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anbima_irts_dataset import (
    DEFAULT_HF_FILENAME,
    DEFAULT_HF_REPO_ID,
    collapse_duplicate_keys,
    coverage_report,
    read_parquet,
    validate_dataset,
    write_parquet,
)


KNOWN_MISSING_DATES = [
    "2012-07-09",
    "2017-11-20",
    "2018-01-25",
    "2018-11-20",
    "2018-12-31",
    "2023-11-17",
]
KNOWN_EXTRA_DATES = [
    "2012-02-20",
    "2012-02-21",
    "2018-12-30",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and optionally upload ANBIMA IRTS latest.parquet.")
    parser.add_argument(
        "--input",
        default=r"C:\Users\rodri\git\models-regimes-trend\anbima_irts.parquet",
        help="Backlog Parquet path.",
    )
    parser.add_argument("--output", default="dist/latest.parquet", help="Output Parquet path.")
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID, help="Hugging Face Dataset repo id.")
    parser.add_argument("--filename", default=DEFAULT_HF_FILENAME, help="Filename inside the HF Dataset repo.")
    parser.add_argument("--upload", action="store_true", help="Upload to Hugging Face after writing the file.")
    args = parser.parse_args()

    raw_df = read_parquet(args.input)
    collapsed_df = collapse_duplicate_keys(raw_df)
    validated_df = validate_dataset(collapsed_df)
    write_parquet(validated_df, args.output)

    report = coverage_report(validated_df)
    payload = {
        "input": str(Path(args.input)),
        "output": str(Path(args.output)),
        "repo_id": args.repo_id,
        "filename": args.filename,
        "rows": report.rows,
        "unique_dates": report.unique_dates,
        "start_date": report.start_date.strftime("%Y-%m-%d") if report.start_date is not None else None,
        "end_date": report.end_date.strftime("%Y-%m-%d") if report.end_date is not None else None,
        "duplicate_key_rows_in_input": int(raw_df.duplicated(["date", "type"]).sum()),
        "known_missing_dates": KNOWN_MISSING_DATES,
        "known_extra_dates": KNOWN_EXTRA_DATES,
    }
    print(json.dumps(payload, indent=2))

    if args.upload:
        try:
            from src.hf_dataset import upload_latest_dataset

            upload_latest_dataset(validated_df, repo_id=args.repo_id, filename=args.filename, create_repo=True)
        except ImportError:
            subprocess.run(
                [
                    "hf",
                    "upload",
                    args.repo_id,
                    args.output,
                    args.filename,
                    "--repo-type",
                    "dataset",
                    "--commit-message",
                    f"Update {args.filename}",
                ],
                check=True,
            )
        print(f"Uploaded {args.output} to {args.repo_id}/{args.filename}")


if __name__ == "__main__":
    main()
