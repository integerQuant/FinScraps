import unittest

import pandas as pd

from src.anbima_irts_dataset import (
    COLUMNS,
    collapse_duplicate_keys,
    merge_new_rows,
    validate_dataset,
)


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


if __name__ == "__main__":
    unittest.main()
