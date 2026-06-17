import unittest
from unittest.mock import patch

import logging
import pandas as pd

from src.anbima_irts_dataset import (
    COLUMNS,
    collapse_duplicate_keys,
    merge_new_rows,
    validate_dataset,
)
from src.managers.Managers import AnbimaIRTSManager


class AnbimaIRTSDatasetTests(unittest.TestCase):
    def test_collapse_duplicate_keys_allows_tiny_numeric_noise(self):
        df = pd.DataFrame(
            [
                ["2025-01-29", "pre", 0.1, 0.2, 0.3, 0.4, 1.0, 2.0],
                ["2025-01-29", "pre", 0.1, 0.2 + 1e-14, 0.3, 0.4, 1.0, 2.0],
                ["2025-01-29", "ipca", 0.5, 0.6, 0.7, 0.8, 3.0, 4.0],
            ],
            columns=COLUMNS,
        )

        collapsed = collapse_duplicate_keys(df)

        self.assertEqual(len(collapsed), 2)
        self.assertEqual(collapsed.duplicated(["date", "type"]).sum(), 0)

    def test_collapse_duplicate_keys_rejects_material_conflicts(self):
        df = pd.DataFrame(
            [
                ["2025-01-29", "pre", 0.1, 0.2, 0.3, 0.4, 1.0, 2.0],
                ["2025-01-29", "pre", 0.1, 0.2001, 0.3, 0.4, 1.0, 2.0],
                ["2025-01-29", "ipca", 0.5, 0.6, 0.7, 0.8, 3.0, 4.0],
            ],
            columns=COLUMNS,
        )

        with self.assertRaises(ValueError):
            collapse_duplicate_keys(df)

    def test_merge_new_rows_appends_only_missing_keys(self):
        existing = pd.DataFrame(
            [
                ["2025-01-29", "ipca", 0.5, 0.6, 0.7, 0.8, 3.0, 4.0],
                ["2025-01-29", "pre", 0.1, 0.2, 0.3, 0.4, 1.0, 2.0],
            ],
            columns=COLUMNS,
        )
        new = pd.DataFrame(
            [
                ["2025-01-29", "ipca", 0.5, 0.6, 0.7, 0.8, 3.0, 4.0],
                ["2025-01-29", "pre", 0.1, 0.2, 0.3, 0.4, 1.0, 2.0],
                ["2025-01-30", "ipca", 0.51, 0.61, 0.71, 0.81, 3.1, 4.1],
                ["2025-01-30", "pre", 0.11, 0.21, 0.31, 0.41, 1.1, 2.1],
            ],
            columns=COLUMNS,
        )

        merged, added_rows = merge_new_rows(existing, new)

        self.assertEqual(added_rows, 2)
        self.assertEqual(len(merged), 4)
        validate_dataset(merged)


class AnbimaIRTSManagerTests(unittest.TestCase):
    def _rows(self, date, offset=0.0):
        return pd.DataFrame(
            [
                [date, "ipca", 0.5 + offset, 0.6, 0.7, 0.8, 3.0, 4.0],
                [date, "pre", 0.1 + offset, 0.2, 0.3, 0.4, 1.0, 2.0],
            ],
            columns=COLUMNS,
        )

    def _manager(self, scraped_by_date, recent_dates):
        manager = AnbimaIRTSManager.__new__(AnbimaIRTSManager)
        manager.scraper = type(
            "DummyScraper",
            (),
            {"scrape": lambda self, date: scraped_by_date[pd.Timestamp(date)]},
        )()
        manager.hf_repo_id = "test/irts"
        manager.hf_filename = "latest.parquet"
        manager.logger = logging.getLogger("AnbimaIRTSManagerTests")
        manager._validate_date = lambda date: True
        manager._recent_business_dates = lambda date, lookback_business_days: recent_dates
        return manager

    @patch("src.managers.Managers.upload_latest_dataset")
    @patch("src.managers.Managers.load_latest_dataset")
    def test_manager_backfills_missing_recent_date_when_target_is_present(self, load_latest, upload_latest):
        target = pd.Timestamp("2025-01-06")
        missing = pd.Timestamp("2025-01-03")
        existing = pd.concat(
            [self._rows("2025-01-02"), self._rows(target, offset=0.1)],
            ignore_index=True,
        )
        load_latest.return_value = existing
        manager = self._manager(
            {missing: self._rows(missing, offset=0.2)},
            [pd.Timestamp("2025-01-02"), missing, target],
        )

        changed = manager.scrape_and_update(target)

        self.assertTrue(changed)
        upload_latest.assert_called_once()
        uploaded_df = upload_latest.call_args.args[0]
        self.assertEqual(len(uploaded_df), 6)
        self.assertEqual(int((uploaded_df["date"] == missing).sum()), 2)


if __name__ == "__main__":
    unittest.main()
