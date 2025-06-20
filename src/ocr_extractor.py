import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from bs4 import BeautifulSoup
from paddleocr import PPStructureV3
from pdf2image import convert_from_path
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEBUG_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "debug_ocr")
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)


class OCRExtractor:
    """
    Extracts tabulated text from a PDF page using PaddleOCR's PPStructureV3
    pipeline. Identifies specific tables (Metadata, Limits, Claims, Benefits)
    based on content.

    Args:
        lang: Language setting for PaddleOCR. Default is "en" (English).
        use_gpu: Whether to use GPU for PaddleOCR. Default is True.
        show_log: Whether to enable PaddleOCR's internal logging. Default is False.
    """

    def __init__(
        self, lang: str = "en", use_gpu: bool = True, show_log: bool = False
    ):
        self.lang = lang
        self.use_gpu = use_gpu
        self.show_log = show_log
        self.structure_pipeline = None

        device_type = "gpu" if self.use_gpu else "cpu"
        logging.info(
            f"Initializing PPStructureV3 with device='{device_type}', "
            f"show_log={self.show_log}"
        )

        try:
            self.structure_pipeline = PPStructureV3(
                device=device_type,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            logging.info("PPStructureV3 initialized successfully.")
        except Exception as e:
            logging.error(
                f"Failed to initialize PPStructureV3: {e}", exc_info=True
            )

    def _convert_pdf_to_image(
        self, pdf_path: str, page_number: int = 0
    ) -> Optional[Image.Image]:
        """
        Converts a specific PDF page to a PIL Image object.

        Args:
            pdf_path: Path to the PDF file.
            page_number: The 0-indexed page number to convert.

        Returns:
            A PIL Image object if conversion is successful, otherwise None.
        """
        try:
            logging.info(
                f"Converting PDF page {page_number} from '{pdf_path}' "
                "to image."
            )
            images = convert_from_path(
                pdf_path,
                first_page=page_number + 1,
                last_page=page_number + 1,
                dpi=300,
            )
            if images:
                logging.info("PDF page converted to image successfully.")
                img_save_path = os.path.join(
                    DEBUG_OUTPUT_DIR, f"page_{page_number}_converted.png"
                )
                images[0].save(img_save_path)
                logging.info(
                    "Saved converted image for debugging to: "
                    f"{img_save_path}"
                )
                return images[0]
            else:
                logging.error(
                    f"Could not convert PDF page {page_number}. No images "
                    "returned."
                )
                return None
        except Exception as e:
            logging.error(
                f"Error converting PDF to image: {e}", exc_info=True
            )
            return None

    def _debug_result_structure(
        self, result: Any, result_save_path: str
    ) -> None:
        """
        Debug helper to print and save the structure of PPStructureV3 results.

        Args:
            result: The raw result from PPStructureV3.
            result_save_path: Path to save the raw result.
        """
        logging.info("=" * 10 + " DEBUG: PPStructureV3 Raw Result " + "=" * 10)
        logging.info(f"Result type: {type(result)}")
        try:
            with open(result_save_path, "w", encoding="utf-8") as f:
                if isinstance(result, (list, dict)):
                    json.dump(result, f, indent=2, default=str)
                else:
                    f.write(str(result))
            logging.info(
                f"Saved raw PPStructureV3 result to: {result_save_path}"
            )
        except Exception as e:
            logging.error(f"Could not save raw result: {e}")

        if isinstance(result, list) and len(result) > 0:
            logging.info(
                f"Result is a list with {len(result)} items (pages/docs)."
            )
            for i, page_item in enumerate(result):
                logging.info(f"Page item {i} type: {type(page_item)}")
                if isinstance(page_item, dict):
                    logging.info(f"Page item {i} keys: {list(page_item.keys())}")
                    if "table_res_list" in page_item:
                        logging.info(
                            f"Page item {i} has 'table_res_list' with "
                            f"{len(page_item['table_res_list'])} tables."
                        )
                        for tbl_idx, tbl_data in enumerate(
                            page_item["table_res_list"]
                        ):
                            if isinstance(tbl_data, dict) and \
                               "pred_html" in tbl_data:
                                logging.info(
                                    f"  Table {tbl_idx} contains 'pred_html' "
                                    "(length: "
                                    f"{len(tbl_data['pred_html'])})."
                                )
        logging.info("=" * 10 + " END DEBUG: PPStructureV3 Raw Result " + "=" * 10)

    def _extract_html_from_ppstructure_result(
        self, result: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extracts HTML table content from PPStructureV3 result.
        Table HTML is found in page_result['table_res_list'][i]['pred_html'].

        Args:
            result: The result from PPStructureV3, expected to be a list of
                    dictionaries (one per page).

        Returns:
            A list of HTML strings, each representing an extracted table.
        """
        extracted_tables_html: List[str] = []
        if not isinstance(result, list):
            logging.warning(
                f"PPStructureV3 result is not a list as expected, but "
                f"{type(result)}. Cannot extract HTML."
            )
            return extracted_tables_html

        for page_idx, page_result in enumerate(result):  # Iterate pages
            if not isinstance(page_result, dict):
                logging.warning(
                    f"Item {page_idx} in PPStructureV3 result list is not a "
                    f"dict: {type(page_result)}"
                )
                continue

            table_res_list = page_result.get("table_res_list")
            if not isinstance(table_res_list, list):
                logging.debug(
                    f"Page {page_idx}: No 'table_res_list' found or it's not "
                    f"a list. Keys: {list(page_result.keys())}"
                )
                continue

            logging.info(
                f"Page {page_idx}: Found {len(table_res_list)} item(s) in "
                "'table_res_list'."
            )
            for table_idx, table_data in enumerate(table_res_list):
                if isinstance(table_data, dict):
                    html_content = table_data.get("pred_html")
                    if html_content and isinstance(html_content, str) and \
                       html_content.strip():
                        logging.info(
                            f"Page {page_idx}, Table {table_idx} in list: "
                            "Extracted HTML from 'pred_html'."
                        )
                        extracted_tables_html.append(html_content)
                    else:
                        logging.debug(
                            f"Page {page_idx}, Table {table_idx} in list: No "
                            "'pred_html' or it's empty."
                        )
                else:
                    logging.debug(
                        f"Page {page_idx}, Table {table_idx} in list: Item is "
                        f"not a dictionary ({type(table_data)})."
                    )

        if not extracted_tables_html:
            logging.warning(
                "No HTML tables extracted from "
                "'table_res_list[*]['pred_html']' structure across all pages."
            )
        return extracted_tables_html

    def _is_metadata_table(self, html_content: str) -> bool:
        """
        Checks if the HTML content likely represents a metadata table.

        Args:
            html_content: The HTML string of the table.

        Returns:
            True if it seems to be a metadata table, False otherwise.
        """
        if not html_content:
            return False
        soup = BeautifulSoup(html_content, "lxml")
        text = soup.get_text().lower()
        keywords = [
            "group number",
            "policy inception date",
            "policy expiry date",
            "class",
            "deductible",
        ]
        return any(keyword in text for keyword in keywords)

    def _is_limits_table(self, html_content: str) -> bool:
        """
        Checks if the HTML content likely represents a limits table.

        Args:
            html_content: The HTML string of the table.

        Returns:
            True if it seems to be a limits table, False otherwise.
        """
        if not html_content:
            return False
        soup = BeautifulSoup(html_content, "lxml")
        text = soup.get_text().lower()
        keywords = [
            "overall benefit limit",
            "inpatient/outpatient limit",
            "dental limit",
            "optical limit",
            "maternity limit",
        ]
        return any(keyword in text for keyword in keywords)

    def _is_claims_table(self, html_content: str) -> bool:
        """
        Checks if the HTML content likely represents a claims table.

        Args:
            html_content: The HTML string of the table.

        Returns:
            True if it seems to be a claims table, False otherwise.
        """
        if not html_content:
            return False
        soup = BeautifulSoup(html_content, "lxml")
        text = soup.get_text().lower()
        # Keywords for claims sections headers
        header_keywords = [
            "monthly claims",
            "number of lives insured",
            "number of paid claims",
            "amount of paid claims",
            "amount of paid claims with vat",
        ]
        policy_year_keywords = [
            "policy year - 2 years prior",
            "prior policy year",
            "last policy year",
        ]

        has_header = sum(kw in text for kw in header_keywords) >= 3
        has_policy_year_ref = any(kw in text for kw in policy_year_keywords)

        # Claims tables usually have numeric data like YYYYMM or totals
        has_data_like_ym = any(
            code in text for code in ["202102", "202103", "202202", "202203"]
        )
        has_data_like_totals = "overall - total" in text or \
                               "210" in text or "179" in text

        return has_header and has_policy_year_ref and \
               (has_data_like_ym or has_data_like_totals)

    def _is_benefits_table(self, html_content: str) -> bool:
        """
        Checks if the HTML content likely represents a benefits table.

        Args:
            html_content: The HTML string of the table.

        Returns:
            True if it seems to be a benefits table, False otherwise.
        """
        if not html_content:
            return False
        soup = BeautifulSoup(html_content, "lxml")
        text = soup.get_text().lower()
        # Keywords tolerant of OCR variations (e.g., "benefit sama")
        keywords = [
            "benefit sama",  # Covers "Benefit tSama" from OCR
            "benefit_sama",
            "overall benefits",  # Main header
            "amt of claims (vat)",
            "notes",
            "number of paid claims",  # Column headers
            "op lab & diagnostics",
            "op consultation",
            "op pharmacy",  # Benefit items
        ]
        match_count = sum(keyword in text for keyword in keywords)
        # "Overall Benefits" or "benefit sama" are strong indicators
        return match_count >= 2 and (
            "overall benefits" in text
            or "benefit sama" in text
            or "benefit_sama" in text
        )

    def extract_tables_from_pdf(
        self, pdf_path: str
    ) -> Dict[str, Optional[str]]:
        """
        Extracts and identifies specific tables (metadata, limits, claims,
        benefits) from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A dictionary where keys are table types (e.g.,
            'metadata_table_html') and values are the corresponding HTML
            strings, or None if not found.
        """
        default_return = {
            key: None
            for key in [
                "metadata_table_html",
                "limits_table_html",
                "claims_table_html",
                "benefits_table_html",
            ]
        }

        if not self.structure_pipeline:
            logging.error(
                "PPStructureV3 pipeline not initialized. Cannot extract "
                "tables."
            )
            return default_return

        image = self._convert_pdf_to_image(pdf_path, page_number=0)
        if not image:
            return default_return

        logging.info(
            f"Running PPStructureV3 prediction on the image from '{pdf_path}'."
        )
        img_np = np.array(image.convert("RGB"))

        result = None
        try:
            result = self.structure_pipeline.predict(img_np)
            logging.info("PPStructureV3 prediction complete.")
        except Exception as e:
            logging.error(
                f"Error during PPStructureV3 prediction: {e}", exc_info=True
            )
            if result is not None:
                self._debug_result_structure(
                    result,
                    os.path.join(DEBUG_OUTPUT_DIR, "ocr_result_on_error.json"),
                )
            return default_return

        self._debug_result_structure(
            result, os.path.join(DEBUG_OUTPUT_DIR, "ocr_result_raw.json")
        )

        extracted_tables_html = self._extract_html_from_ppstructure_result(
            result
        )

        if not extracted_tables_html:
            logging.warning(
                "No HTML table content found by PPStructureV3 after parsing "
                "results."
            )
            return default_return

        logging.info(
            f"Successfully extracted {len(extracted_tables_html)} HTML table "
            "structures. Identifying them..."
        )

        identified_tables: Dict[str, Optional[str]] = {
            "metadata_table_html": None,
            "limits_table_html": None,
            "claims_table_html": None,
            "benefits_table_html": None,
        }

        # Strategy:
        # 1. Identify specific "Benefits" table (often smaller).
        # 2. Use other tables for Metadata, Limits, Claims. These might point
        #    to the same HTML if it's a composite table.

        candidate_benefits_htmls = []
        for i, html_content in enumerate(extracted_tables_html):
            if self._is_benefits_table(html_content):
                candidate_benefits_htmls.append(
                    {"html": html_content, "len": len(html_content), "idx": i}
                )

        if candidate_benefits_htmls:
            # Prefer shorter HTML for specific benefits table
            candidate_benefits_htmls.sort(key=lambda x: x["len"])
            best_benefits_table_info = candidate_benefits_htmls[0]
            identified_tables["benefits_table_html"] = (
                best_benefits_table_info["html"]
            )
            logging.info(
                f"Identified Benefits Table from HTML block "
                f"{best_benefits_table_info['idx']} (shortest match, "
                f"len {best_benefits_table_info['len']})."
            )

        # Assign Metadata, Limits, Claims
        for i, html_content in enumerate(extracted_tables_html):
            if not identified_tables["metadata_table_html"] and \
               self._is_metadata_table(html_content):
                identified_tables["metadata_table_html"] = html_content
                logging.info(f"Identified Metadata Table from HTML block {i}.")

            if not identified_tables["limits_table_html"] and \
               self._is_limits_table(html_content):
                identified_tables["limits_table_html"] = html_content
                logging.info(f"Identified Limits Table from HTML block {i}.")

            if not identified_tables["claims_table_html"] and \
               self._is_claims_table(html_content):
                identified_tables["claims_table_html"] = html_content
                logging.info(f"Identified Claims Table from HTML block {i}.")

        # Fallback: If meta, limits, or claims are still None, and a large
        # table exists, assign it if it matches the criteria.
        if len(extracted_tables_html) > 0:
            # Sort by length, descending (largest first)
            sorted_html_by_len_desc = sorted(
                extracted_tables_html, key=len, reverse=True
            )
            largest_html = sorted_html_by_len_desc[0]

            idx_largest = -1 # Find original index of largest_html
            for i_orig, orig_html_content in enumerate(extracted_tables_html):
                if orig_html_content == largest_html:
                    idx_largest = i_orig
                    break

            if not identified_tables["metadata_table_html"] and \
               self._is_metadata_table(largest_html):
                identified_tables["metadata_table_html"] = largest_html
                logging.info(
                    f"Assigned largest HTML (block {idx_largest}, "
                    f"len {len(largest_html)}) to Metadata Table (fallback)."
                )

            if not identified_tables["limits_table_html"] and \
               self._is_limits_table(largest_html):
                identified_tables["limits_table_html"] = largest_html
                logging.info(
                    f"Assigned largest HTML (block {idx_largest}, "
                    f"len {len(largest_html)}) to Limits Table (fallback)."
                )

            if not identified_tables["claims_table_html"] and \
               self._is_claims_table(largest_html):
                identified_tables["claims_table_html"] = largest_html
                logging.info(
                    f"Assigned largest HTML (block {idx_largest}, "
                    f"len {len(largest_html)}) to Claims Table (fallback)."
                )

        # Log unassigned tables for debugging
        used_html_contents = set(filter(None, identified_tables.values()))
        unassigned_count = 0
        for i, html_content in enumerate(extracted_tables_html):
            if html_content not in used_html_contents:
                unassigned_count += 1
                unassigned_save_path = os.path.join(
                    DEBUG_OUTPUT_DIR, f"unassigned_table_{i}.html"
                )
                try:
                    with open(
                        unassigned_save_path, "w", encoding="utf-8"
                    ) as f:
                        f.write(html_content)
                    logging.info(
                        "Saved unassigned table (original index "
                        f"{i}) to {unassigned_save_path}"
                    )
                except Exception as e_save:
                    logging.error(
                        f"Failed to save unassigned_table_{i}.html: {e_save}"
                    )

        if unassigned_count > 0:
            logging.warning(
                f"{unassigned_count} HTML table(s) were extracted but not "
                "ultimately assigned to a specific category."
            )

        for name, content in identified_tables.items():
            if content is None:
                logging.warning(f"TABLE NOT IDENTIFIED/EXTRACTED: {name}")
            else:
                logging.info(
                    f"Final assignment for {name}: HTML from block with "
                    f"original length {len(content)}"
                )

        return identified_tables

# --- Example Usage ---
if __name__ == "__main__":
    pdf_file_path = os.path.join(
        PROJECT_ROOT, "data", "sample_insurance_report.pdf"
    )

    if not os.path.exists(pdf_file_path):
        logging.error(
            f"Sample PDF not found at {pdf_file_path}. Please place it "
            "there or update the path."
        )
    else:
        logging.info(f"Starting OCR extraction for: {pdf_file_path}")
        # Initialize OCRExtractor.
        extractor = OCRExtractor(lang="en", use_gpu=True, show_log=False)

        if extractor.structure_pipeline is None:
            logging.error(
                "OCRExtractor could not initialize PPStructureV3. Aborting."
            )
        else:
            extracted_data_html = extractor.extract_tables_from_pdf(
                pdf_file_path
            )

            main_output_dir = os.path.join(PROJECT_ROOT, "output")
            os.makedirs(main_output_dir, exist_ok=True)

            for table_name, html_content in extracted_data_html.items():
                if html_content:
                    logging.info(
                        f"--- Successfully Identified: {table_name} "
                        f"(HTML length: {len(html_content)}) ---"
                    )
                    # Preview of the HTML content
                    logging.info(
                        html_content[:200].replace("\n", " ") + "..."
                    )
                    output_html_path = os.path.join(
                        main_output_dir, f"{table_name}.html"
                    )
                    with open(
                        output_html_path, "w", encoding="utf-8"
                    ) as f:
                        f.write(html_content)
                    logging.info(
                        f"Saved full HTML for {table_name} to "
                        f"{output_html_path}"
                    )
                else:
                    logging.warning(
                        f"--- Failed to Identify/Extract: {table_name} ---"
                    )

            logging.info("OCR extraction process finished.")
            logging.info(
                "Debug files (converted image, raw OCR result, unassigned "
                f"tables) are in: {DEBUG_OUTPUT_DIR}"
            )
            logging.info(
                f"Identified table HTMLs are in: {main_output_dir}"
            )