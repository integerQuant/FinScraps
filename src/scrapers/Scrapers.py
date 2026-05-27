import time
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from io import BytesIO
from functools import reduce

from src.anbima_idka_dataset import IDKA_CODES, validate_dataset

class AnbimaIRTSScraper:
    """A scraper for retrieving and parsing IRTS (Term Structure) data from ANBIMA in XML format.
    """

    def __init__(self):
        self.url = "https://www.anbima.com.br/informacoes/est-termo/CZ-down.asp"

    def download_xml(self, date) -> bytes:
        """Download the XML content from ANBIMA for a given date, with simple retry logic.

        Parameters
        ----------
        date : datetime-like

        Returns
        -------
        bytes: The raw XML content in bytes.

        Raises
        ------
        requests.exceptions.RequestException: If the request ultimately fails after all retries.
        """
        date_str = pd.Timestamp(date).strftime("%d/%m/%Y")
        form_data = {
            "Idioma": "PT",
            "Dt_Ref": date_str,
            "saida": "xml"
        }

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(self.url, data=form_data)
                response.raise_for_status()
                return response.content
            except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1) 
                else:
                    raise e

    def parse_params(self, xml_content, date):
        """Parse the XML content to extract IRTS parameters.

        Parameters
        ----------
        xml_content : The raw XML content in bytes.
        date : The date to associate with the extracted parameters (same date used in download_xml).

        Returns
        -------
        list of dict A list of dictionaries, each containing parameters 
        """
        
        root = ET.fromstring(xml_content)
        parametros = []

        type_mapping = {
            "PREFIXADOS": "pre",
            "IPCA": "ipca"
        }

        def convert(value):
            """Convert a string with decimal comma to float. Returns None if value is empty or None.
            """
            return float(value.replace(',', '.')) if value else None

        for parametro in root.findall(".//PARAMETRO"):
            original_type = parametro.get("Grupo")
            renamed_type = type_mapping.get(original_type, original_type)

            parametros.append({
                "date": pd.Timestamp(date),
                "type": renamed_type,
                "b1": convert(parametro.get("B1")),
                "b2": convert(parametro.get("B2")),
                "b3": convert(parametro.get("B3")),
                "b4": convert(parametro.get("B4")),
                "l1": convert(parametro.get("L1")),
                "l2": convert(parametro.get("L2")),
            })

        return parametros

    def scrape(self, date):
        """Download and parse parameters for a given date, returning a DataFrame.

        Parameters
        ----------
        date : The date for which data is to be downloaded and parsed.

        Returns
        -------
        DataFrame containing the scraped parameters on the provided date.
        """
        xml_content = self.download_xml(date)
        parametros = self.parse_params(xml_content, date)
        return pd.DataFrame(parametros)


class AnbimaIDKAScraper:
    """A scraper for retrieving and parsing ANBIMA IDKA historical workbooks."""

    BASE_URL = "https://s3-data-prd-use1-precos.s3.us-east-1.amazonaws.com/arquivos/indices-historico"
    SOURCE_COLUMNS = ["Data de Referência", "Número Índice"]

    def __init__(self):
        self.urls = {
            code: f"{self.BASE_URL}/{code}-HISTORICO.xls"
            for code in IDKA_CODES
        }

    def download_workbook(self, code: str) -> bytes:
        """Download one IDKA historical workbook, with simple retry logic."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.get(self.urls[code], timeout=30)
                response.raise_for_status()
                return response.content
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise e

    def parse_workbook(self, workbook_content: bytes, code: str) -> pd.DataFrame:
        """Parse an IDKA workbook into a two-column date/value frame."""
        df = pd.read_excel(
            BytesIO(workbook_content),
            sheet_name="Historico",
            engine="openpyxl",
            usecols=self.SOURCE_COLUMNS,
        )
        parsed = df.rename(
            columns={
                "Data de Referência": "date",
                "Número Índice": code,
            }
        )
        parsed["date"] = pd.to_datetime(parsed["date"], errors="raise").dt.normalize()
        parsed[code] = pd.to_numeric(parsed[code], errors="raise")
        return parsed.loc[:, ["date", code]]

    def scrape(self) -> pd.DataFrame:
        """Download all IDKA workbooks and return the public wide dataset."""
        series_frames = [
            self.parse_workbook(self.download_workbook(code), code)
            for code in IDKA_CODES
        ]

        wide_df = reduce(
            lambda left, right: left.merge(right, on="date", how="outer", validate="one_to_one"),
            series_frames,
        )
        wide_df = wide_df.sort_values("date").reset_index(drop=True)
        return validate_dataset(wide_df)
