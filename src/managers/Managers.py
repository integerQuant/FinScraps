import logging

import pandas as pd

from src.anbima_idka_dataset import (
    DEFAULT_HF_FILENAME as DEFAULT_IDKA_HF_FILENAME,
    DEFAULT_HF_REPO_ID as DEFAULT_IDKA_HF_REPO_ID,
    validate_dataset as validate_idka_dataset,
    write_parquet as write_idka_parquet,
)
from src.anbima_irts_dataset import (
    DEFAULT_HF_FILENAME,
    DEFAULT_HF_REPO_ID,
    merge_new_rows,
    validate_dataset,
)
from src.hf_dataset import load_latest_dataset, upload_latest_dataset
from src.scrapers.Scrapers import AnbimaIDKAScraper, AnbimaIRTSScraper
from src.utils import BRCal

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s")

class AnbimaIRTSManager:
    """
    A manager for the AnbimaIRTSScraper. Handles the workflow of:
    - Validating the requested date
    - Downloading new data for a given date
    - Comparing new data with the latest Hugging Face dataset
    - Appending and uploading the latest Parquet blob
    """

    def __init__(
        self,
        hf_repo_id: str = DEFAULT_HF_REPO_ID,
        hf_filename: str = DEFAULT_HF_FILENAME,
    ):
        """
        Parameters
        ----------
        hf_repo_id : str
            Hugging Face Dataset repo that stores the latest Parquet blob.
        hf_filename : str
            Dataset filename inside the Hugging Face repo.
        """
        self.scraper = AnbimaIRTSScraper()
        self.hf_repo_id = hf_repo_id
        self.hf_filename = hf_filename
        self.calendar = BRCal()

        self.logger = logging.getLogger(self.__class__.__name__)

    def scrape_and_update(self, date):
        """Download and parse IRTS data, then update the Hugging Face latest blob.

        Parameters
        ----------
        date : datetime or datetime-like
        Returns
        -------
        pd.DataFrame or None
            DataFrame containing the merged dataset (old + new) if scraping is done;
            None if skipped due to invalid date or data already present.
        """
        if not self._validate_date(date):
            return False

        existing_df = load_latest_dataset(self.hf_repo_id, self.hf_filename)
        if existing_df.empty:
            self.logger.info("No existing Hugging Face dataset found. Creating a new one.")
        else:
            existing_df = validate_dataset(existing_df)
            if date in existing_df["date"].unique():
                self.logger.info(f"Data for date {date.date()} is already present. Skipping scrape.")
                return False
            self.logger.info(f"Existing dataset loaded with {existing_df.shape[0]} rows.")

        self.logger.info(f"Starting data scrape for date: {date}...")
        try:
            new_data = self.scraper.scrape(date)
            new_nrows = new_data.shape[0]
        except Exception as e:
            self.logger.error(f"Error during scraping: {e}")
            raise

        self.logger.info(
            f"New data fetched for date {date.date()}: {new_nrows} rows"
        )

        combined_df, added_rows = merge_new_rows(existing_df, new_data)
        if added_rows == 0:
            self.logger.info(f"No new rows for date {date.date()}. Skipping upload.")
            return False

        upload_latest_dataset(combined_df, self.hf_repo_id, self.hf_filename)
        self.logger.info(
            f"Uploaded {self.hf_repo_id}/{self.hf_filename} with {combined_df.shape[0]} rows."
        )

        return True

    def _validate_date(self, date):
        """
        Validate the requested date to ensure:
        - It's not in the future
        - It's not older than 5 business days from today

        Parameters
        ----------
        date : datetime or datetime-like

        Returns
        -------
        bool
            True if the date is valid for scraping; otherwise False.
        """
        
        if not self.calendar.is_business_day(date):
            self.logger.warning(
                f"Provided date {date.date()} is not a business day. Skipping."
            )
            return False

        if date > self.calendar.today:
            self.logger.warning(
                f"Provided date {date.date()} is in the future. Skipping."
            )
            return False

        day_count = len(self.calendar.day_range(date, self.calendar.today))
        if day_count > 5:
            self.logger.warning(
                f"Provided date {date.date()} is older than 5 business days. Skipping."
            )
            return False

        return True


class AnbimaIDKAManager:
    """
    A manager for the AnbimaIDKAScraper. Handles the workflow of:
    - Validating the requested date
    - Downloading the cumulative IDKA source workbooks
    - Comparing the canonical wide frame with the latest Hugging Face dataset
    - Uploading a replacement Parquet blob only when the dataset changed
    """

    def __init__(
        self,
        hf_repo_id: str = DEFAULT_IDKA_HF_REPO_ID,
        hf_filename: str = DEFAULT_IDKA_HF_FILENAME,
    ):
        """
        Parameters
        ----------
        hf_repo_id : str
            Hugging Face Dataset repo that stores the latest Parquet blob.
        hf_filename : str
            Dataset filename inside the Hugging Face repo.
        """
        self.scraper = AnbimaIDKAScraper()
        self.hf_repo_id = hf_repo_id
        self.hf_filename = hf_filename
        self.calendar = BRCal()

        self.logger = logging.getLogger(self.__class__.__name__)

    def scrape_and_update(self, date):
        """Download IDKA history, then update the Hugging Face latest blob if changed."""
        if not self._validate_date(date):
            return False

        existing_df = load_latest_dataset(self.hf_repo_id, self.hf_filename)
        if existing_df.empty:
            self.logger.info("No existing Hugging Face dataset found. Creating a new one.")
            existing = pd.DataFrame()
        else:
            existing = validate_idka_dataset(existing_df)
            self.logger.info(f"Existing dataset loaded with {existing.shape[0]} rows.")

        self.logger.info("Starting IDKA historical workbook scrape...")
        try:
            new_data = self.scraper.scrape()
        except Exception as e:
            self.logger.error(f"Error during scraping: {e}")
            raise

        if date not in set(new_data["date"]):
            self.logger.warning(
                f"Scraped IDKA data does not contain requested date {date.date()}."
            )

        self.logger.info(
            f"IDKA data fetched: {new_data.shape[0]} rows from "
            f"{new_data['date'].min().date()} to {new_data['date'].max().date()}."
        )

        if not existing.empty and existing.equals(new_data):
            self.logger.info("IDKA dataset is unchanged. Skipping upload.")
            return False

        upload_latest_dataset(
            new_data,
            self.hf_repo_id,
            self.hf_filename,
            writer=write_idka_parquet,
        )
        self.logger.info(
            f"Uploaded {self.hf_repo_id}/{self.hf_filename} with {new_data.shape[0]} rows."
        )

        return True

    def _validate_date(self, date):
        """
        Validate the requested date to ensure:
        - It's not in the future
        - It's not older than 5 business days from today
        """

        if not self.calendar.is_business_day(date):
            self.logger.warning(
                f"Provided date {date.date()} is not a business day. Skipping."
            )
            return False

        if date > self.calendar.today:
            self.logger.warning(
                f"Provided date {date.date()} is in the future. Skipping."
            )
            return False

        day_count = len(self.calendar.day_range(date, self.calendar.today))
        if day_count > 5:
            self.logger.warning(
                f"Provided date {date.date()} is older than 5 business days. Skipping."
            )
            return False

        return True
