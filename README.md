# FinScraps

Open-source scrapers for Brazilian financial-market datasets.

## ANBIMA IRTS Dataset

FinScraps currently publishes ANBIMA Interest Rate Term Structure (IRTS)
parameters as a public Parquet dataset.

- Dataset repo: https://huggingface.co/datasets/rodrigomtorresb/anbima-irts
- File: `latest.parquet`
- Schema: `date`, `type`, `b1`, `b2`, `b3`, `b4`, `l1`, `l2`

The dataset is accumulated in a single latest blob. Generated data is published
to Hugging Face and is not committed back into this Git repository.

## Usage

```python
import pandas as pd

url = "https://huggingface.co/datasets/rodrigomtorresb/anbima-irts/resolve/main/latest.parquet"
df = pd.read_parquet(url)
```

## Automation

The scraper runs on GitHub Actions after Brazilian market close, fetches the
previous Brazilian business day, appends any missing rows to the latest Parquet
blob, validates the result, and uploads the replacement file to Hugging Face.