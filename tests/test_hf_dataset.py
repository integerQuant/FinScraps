import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
import pandas as pd
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, LocalEntryNotFoundError

from src.hf_dataset import _call_with_retries, load_latest_dataset, upload_latest_dataset


def _hf_error(status_code, retry_after=None):
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    request = httpx.Request("GET", "https://huggingface.co/datasets/test/repo/resolve/main/latest.parquet")
    response = httpx.Response(status_code, headers=headers, request=request)
    return HfHubHTTPError(f"HTTP {status_code}", response=response)


class HuggingFaceDatasetTests(unittest.TestCase):
    @patch("huggingface_hub.hf_hub_download")
    def test_load_latest_dataset_returns_empty_for_missing_file(self, hf_hub_download):
        hf_hub_download.side_effect = EntryNotFoundError("missing file")

        result = load_latest_dataset("test/repo", "latest.parquet")

        self.assertTrue(result.empty)

    @patch("huggingface_hub.hf_hub_download")
    def test_load_latest_dataset_raises_local_cache_miss(self, hf_hub_download):
        hf_hub_download.side_effect = LocalEntryNotFoundError("rate limited before download")

        with self.assertRaises(LocalEntryNotFoundError):
            load_latest_dataset("test/repo", "latest.parquet")

    @patch("src.hf_dataset.time.sleep")
    def test_retry_after_above_cap_fails_fast(self, sleep):
        operation = Mock(side_effect=_hf_error(429, retry_after=600))

        with self.assertRaises(HfHubHTTPError):
            _call_with_retries(
                operation,
                "download test dataset",
                max_retries=5,
                max_retry_seconds=120,
                max_sleep_seconds=30,
            )

        self.assertEqual(operation.call_count, 1)
        sleep.assert_not_called()

    @patch("src.hf_dataset.time.sleep")
    def test_retry_uses_short_retry_after_then_succeeds(self, sleep):
        operation = Mock(side_effect=[_hf_error(429, retry_after=2), "ok"])

        result = _call_with_retries(
            operation,
            "download test dataset",
            max_retries=5,
            max_retry_seconds=120,
            max_sleep_seconds=30,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(2.0)

    @patch("src.hf_dataset.time.sleep")
    def test_retry_handles_transient_network_error(self, sleep):
        request = httpx.Request("GET", "https://huggingface.co/datasets/test/repo")
        operation = Mock(side_effect=[httpx.ConnectError("offline", request=request), "ok"])

        result = _call_with_retries(
            operation,
            "download test dataset",
            max_retries=5,
            max_retry_seconds=120,
            max_sleep_seconds=30,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once()

    @patch("huggingface_hub.HfApi")
    def test_upload_skips_repo_creation_by_default(self, hf_api):
        api = hf_api.return_value
        writer = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            upload_latest_dataset(
                pd.DataFrame({"date": [pd.Timestamp("2025-01-02")]}),
                "test/repo",
                "latest.parquet",
                workdir=tmpdir,
                writer=writer,
            )

        api.create_repo.assert_not_called()
        api.upload_file.assert_called_once()

    @patch("src.hf_dataset.time.sleep")
    @patch("huggingface_hub.HfApi")
    def test_upload_file_retries_transient_rate_limit(self, hf_api, sleep):
        api = hf_api.return_value
        api.upload_file.side_effect = [_hf_error(429, retry_after=1), None]

        with tempfile.TemporaryDirectory() as tmpdir:
            upload_latest_dataset(
                pd.DataFrame({"date": [pd.Timestamp("2025-01-02")]}),
                "test/repo",
                "latest.parquet",
                workdir=tmpdir,
                writer=Mock(),
            )

        self.assertEqual(api.upload_file.call_count, 2)
        sleep.assert_called_once_with(1.0)


if __name__ == "__main__":
    unittest.main()
