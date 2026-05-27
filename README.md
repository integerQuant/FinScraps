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

## ANBIMA IDKA Dataset

FinScraps also publishes ANBIMA IDKA historical index levels as a wide public
Parquet dataset.

- Dataset repo: https://huggingface.co/datasets/rodrigomtorresb/anbima-idka
- File: `latest.parquet`
- Schema: `date`, `IDKAPRE3M`, `IDKAPRE1A`, `IDKAPRE2A`, `IDKAPRE3A`,
  `IDKAPRE5A`, `IDKAIPCA2A`, `IDKAIPCA3A`, `IDKAIPCA5A`, `IDKAIPCA10A`,
  `IDKAIPCA15A`, `IDKAIPCA20A`, `IDKAIPCA30A`

Each IDKA column contains ANBIMA's `Número Índice` value. Derived return
columns from the source workbooks are intentionally omitted.

## Usage

```python
import pandas as pd

url = "https://huggingface.co/datasets/rodrigomtorresb/anbima-irts/resolve/main/latest.parquet"
df = pd.read_parquet(url)
```

```python
import pandas as pd

url = "https://huggingface.co/datasets/rodrigomtorresb/anbima-idka/resolve/main/latest.parquet"
df = pd.read_parquet(url)
```

## Automation

The scraper runs on GitHub Actions after Brazilian market close. It fetches the
previous Brazilian business day for IRTS, rebuilds the cumulative IDKA wide
frame from ANBIMA's historical workbooks, validates the resulting datasets, and
uploads replacement files to Hugging Face when data changed.
