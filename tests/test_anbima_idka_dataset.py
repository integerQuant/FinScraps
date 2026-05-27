from io import BytesIO
import logging
import unittest
from unittest.mock import patch

import pandas as pd

from src.anbima_idka_dataset import COLUMNS, IDKA_CODES, validate_dataset
from src.managers.Managers import AnbimaIDKAManager
from src.scrapers.Scrapers import AnbimaIDKAScraper


class AnbimaIDKADatasetTests(unittest.TestCase):
    def _valid_df(self):
        rows = []
        for date, offset in [("2025-01-02", 0), ("2025-01-03", 100)]:
            row = {"date": date}
            for position, code in enumerate(IDKA_CODES):
                row[code] = 1000.0 + offset + position
            rows.append(row)
        return pd.DataFrame(rows, columns=COLUMNS)

    def test_validate_dataset_accepts_public_schema(self):
        validated = validate_dataset(self._valid_df())

        self.assertEqual(list(validated.columns), COLUMNS)
        self.assertEqual(validated["date"].tolist(), [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")])

    def test_validate_dataset_rejects_missing_required_columns(self):
        df = self._valid_df().drop(columns=["IDKAPRE3M"])

        with self.assertRaises(ValueError):
            validate_dataset(df)

    def test_validate_dataset_rejects_duplicate_dates(self):
        df = pd.concat([self._valid_df(), self._valid_df().iloc[[0]]], ignore_index=True)

        with self.assertRaises(ValueError):
            validate_dataset(df)

    def test_validate_dataset_rejects_null_values(self):
        df = self._valid_df()
        df.loc[0, "IDKAIPCA10A"] = None

        with self.assertRaises(ValueError):
            validate_dataset(df)

    def test_validate_dataset_rejects_non_numeric_values(self):
        df = self._valid_df()
        df["IDKAPRE1A"] = df["IDKAPRE1A"].astype(object)
        df.loc[0, "IDKAPRE1A"] = "not-a-number"

        with self.assertRaises(ValueError):
            validate_dataset(df)

    def test_validate_dataset_rejects_unsorted_dates(self):
        df = self._valid_df().iloc[::-1].reset_index(drop=True)

        with self.assertRaises(ValueError):
            validate_dataset(df)


class AnbimaIDKAScraperTests(unittest.TestCase):
    def _workbook(self, values):
        output = BytesIO()
        df = pd.DataFrame(
            {
                "Índice": ["IDkA Test"] * len(values),
                "Data de Referência": [date for date, _ in values],
                "Número Índice": [value for _, value in values],
                "Variação Diária (%)": [0.1] * len(values),
            }
        )
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Historico", index=False)
        return output.getvalue()

    def test_parse_workbook_extracts_date_and_index_value(self):
        scraper = AnbimaIDKAScraper()

        parsed = scraper.parse_workbook(
            self._workbook([("2025-01-02", 1000.0), ("2025-01-03", 1001.5)]),
            "IDKAPRE3M",
        )

        expected = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")],
                "IDKAPRE3M": [1000.0, 1001.5],
            }
        )
        pd.testing.assert_frame_equal(parsed, expected)

    def test_scrape_builds_wide_frame_in_public_column_order(self):
        scraper = AnbimaIDKAScraper()
        workbooks = {
            code: self._workbook([("2025-01-02", 1000.0 + idx), ("2025-01-03", 1100.0 + idx)])
            for idx, code in enumerate(IDKA_CODES)
        }
        scraper.download_workbook = lambda code: workbooks[code]

        result = scraper.scrape()

        self.assertEqual(list(result.columns), COLUMNS)
        self.assertEqual(result.shape, (2, len(COLUMNS)))
        self.assertEqual(result.loc[0, "IDKAPRE3M"], 1000.0)
        self.assertEqual(result.loc[1, "IDKAIPCA30A"], 1100.0 + len(IDKA_CODES) - 1)


class AnbimaIDKAManagerTests(unittest.TestCase):
    def _manager(self, scraped_df):
        manager = AnbimaIDKAManager.__new__(AnbimaIDKAManager)
        manager.scraper = type("DummyScraper", (), {"scrape": lambda self: scraped_df})()
        manager.hf_repo_id = "test/idka"
        manager.hf_filename = "latest.parquet"
        manager.logger = logging.getLogger("AnbimaIDKAManagerTests")
        manager._validate_date = lambda date: True
        return manager

    def _valid_df(self, extra_day=False):
        rows = []
        dates = ["2025-01-02", "2025-01-03"]
        if extra_day:
            dates.append("2025-01-06")
        for offset, date in enumerate(dates):
            row = {"date": pd.Timestamp(date)}
            for position, code in enumerate(IDKA_CODES):
                row[code] = 1000.0 + (offset * 100) + position
            rows.append(row)
        return pd.DataFrame(rows, columns=COLUMNS)

    @patch("src.managers.Managers.upload_latest_dataset")
    @patch("src.managers.Managers.load_latest_dataset")
    def test_manager_skips_upload_when_dataset_is_unchanged(self, load_latest, upload_latest):
        df = self._valid_df()
        load_latest.return_value = df.copy()
        manager = self._manager(df.copy())

        changed = manager.scrape_and_update(pd.Timestamp("2025-01-03"))

        self.assertFalse(changed)
        upload_latest.assert_not_called()

    @patch("src.managers.Managers.upload_latest_dataset")
    @patch("src.managers.Managers.load_latest_dataset")
    def test_manager_uploads_when_dataset_changed(self, load_latest, upload_latest):
        existing = self._valid_df()
        scraped = self._valid_df(extra_day=True)
        load_latest.return_value = existing
        manager = self._manager(scraped)

        changed = manager.scrape_and_update(pd.Timestamp("2025-01-06"))

        self.assertTrue(changed)
        upload_latest.assert_called_once()


if __name__ == "__main__":
    unittest.main()
