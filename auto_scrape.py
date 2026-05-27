from src.managers.Managers import AnbimaIDKAManager, AnbimaIRTSManager
from src.utils import BRCal
import logging
import os

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s")

brcal = BRCal()
br_business_day = brcal.previous_business_day(brcal.today)

if br_business_day:
    AIRTSM = AnbimaIRTSManager(
        hf_repo_id=os.getenv("ANBIMA_IRTS_HF_REPO_ID", "rodrigomtorresb/anbima-irts"),
        hf_filename=os.getenv("ANBIMA_IRTS_HF_FILENAME", "latest.parquet"),
    )
    AIRTSM.scrape_and_update(br_business_day)
    logging.info("Anbima IRTS data successfully scraped and updated.")

    AIDKAM = AnbimaIDKAManager(
        hf_repo_id=os.getenv("ANBIMA_IDKA_HF_REPO_ID", "rodrigomtorresb/anbima-idka"),
        hf_filename=os.getenv("ANBIMA_IDKA_HF_FILENAME", "latest.parquet"),
    )
    AIDKAM.scrape_and_update(br_business_day)
    logging.info("Anbima IDKA data successfully scraped and updated.")
else:
    logging.info("Not a BR business day. Skipping BR scraping routines.")
