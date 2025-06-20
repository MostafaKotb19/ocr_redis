import io
import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataProcessor:
    """
    Processes HTML table strings to extract structured data into Pandas
    DataFrames.

    Args:
        html_tables: Dictionary containing HTML strings for claims,
            benefits, and metadata tables. Keys should include
            'claims_table_html', 'benefits_table_html', and optionally
            'metadata_table_html'.
    """
    def __init__(self, html_tables: Dict[str, Optional[str]]):
        self.html_tables = html_tables
        self.extracted_metadata = {
            "end_date": None,
            "class": None,
            "overall_limit": None,
            "group_number": None,
            "policy_inception_date": None,
            "policy_expiry_date_raw": None,
            "deductible": None,
        }
        logging.info("DataProcessor initialized.")

    def _clean_text(self, text: Any) -> Optional[str]:
        """
        Cleans text by stripping whitespace.

        Args:
            text: The input text to clean, can be str, int, float, or None.

        Returns:
            Cleaned string or None if input is not a valid string or is empty
            after stripping.
        """
        if isinstance(text, str):
            cleaned = text.strip()
            return cleaned if cleaned else None
        return None

    def _clean_numeric_string(
        self, value: Any, remove_chars: str = ","
    ) -> Optional[str]:
        """
        Removes specified characters (e.g., commas) from a string value.

        Args:
            value: The input value to clean, can be str, int, float, or None.
            remove_chars: Characters to remove from the string.

        Returns:
            Cleaned string with specified characters removed, or None if input
            is not valid.
        """
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            for char in remove_chars:
                value = value.replace(char, "")
            return value.strip()
        return None

    def _to_float(
        self, value: Any, default_if_empty: Optional[float] = 0.0
    ) -> Optional[float]:
        """
        Converts a value to float, handling potential errors and cleaning.

        Args:
            value: The input value to convert, can be str, int, float, or
                   None.
            default_if_empty: Default value to return if conversion fails or
                              value is empty.

        Returns:
            Converted float value or default_if_empty if conversion fails.
        """
        if value is None:
            return default_if_empty
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned_value = self._clean_numeric_string(value)
            if not cleaned_value:
                return default_if_empty
            try:
                return float(cleaned_value)
            except ValueError:
                logging.warning(f"Could not convert '{value}' to float.")
                return None
        return default_if_empty

    def _to_int(
        self, value: Any, default_if_empty: Optional[int] = 0
    ) -> Optional[int]:
        """
        Converts a value to int, handling potential errors and cleaning.

        Args:
            value: The input value to convert, can be str, int, float, or
                   None.
            default_if_empty: Default value to return if conversion fails or
                              value is empty.

        Returns:
            Converted int value or default_if_empty if conversion fails.
        """
        float_val = self._to_float(value, default_if_empty=None)
        if float_val is not None:
            try:
                return int(float_val)
            except ValueError:
                logging.warning(
                    f"Could not convert float '{float_val}' (from '{value}') to int."
                )
                return None
        return default_if_empty

    def _parse_date_flexible(
        self, date_str: Optional[str], output_format: str = "%Y-%m-%d"
    ) -> Optional[str]:
        """
        Parses date string with common formats to a specific output format.

        Args:
            date_str: The date string to parse, can be None or empty.
            output_format: Desired output format for the date string.

        Returns:
            Formatted date string or None if parsing fails.
        """
        if not date_str:
            return None

        date_str = date_str.replace("Date", "").strip()  # Remove "Date" prefix

        common_formats = [
            "%b %d, %Y",  # Feb 16, 2023
            "%Y-%m-%d",  # 2023-02-16
            "%d/%m/%Y",  # 16/02/2023
        ]
        for fmt in common_formats:
            try:
                dt_obj = datetime.strptime(date_str, fmt)
                return dt_obj.strftime(output_format)
            except ValueError:
                continue
        logging.warning(
            f"Could not parse date string: '{date_str}' with known formats."
        )
        return None

    def _extract_specific_metadata_from_df(self, df_raw: pd.DataFrame) -> None:
        """
        Extracts key metadata fields from a raw DataFrame.

        Args:
            df_raw: The raw DataFrame containing the combined table data.
        """
        logging.info(
            "Attempting to extract metadata and limits from the combined "
            "table's DataFrame."
        )
        for index, row in df_raw.iterrows():
            for col_idx, cell_content_any in enumerate(row):
                cell_content = self._clean_text(str(cell_content_any))
                if not cell_content:
                    continue

                if "group number" in cell_content.lower():
                    match = re.search(
                        r"group number\s*(\d+)", cell_content.lower()
                    )
                    if match:
                        self.extracted_metadata["group_number"] = (
                            self._clean_text(match.group(1))
                        )
                        logging.info(
                            "Extracted Group Number: "
                            f"{self.extracted_metadata['group_number']}"
                        )

                if "policy inception date" in cell_content.lower():
                    date_part = (
                        cell_content.lower()
                        .replace("policy inception date", "")
                        .strip()
                    )
                    # If date part is empty or unparsable from current cell,
                    # check next cell
                    if not self._parse_date_flexible(date_part):
                        if col_idx + 1 < len(row):
                            next_cell_content = self._clean_text(
                                str(row.iloc[col_idx + 1])
                            )
                            date_part = next_cell_content
                    parsed_date = self._parse_date_flexible(date_part)
                    if parsed_date:
                        self.extracted_metadata["policy_inception_date"] = (
                            parsed_date
                        )
                        logging.info(
                            "Extracted Policy Inception Date: "
                            f"{parsed_date}"
                        )

                if "policy expiry date" in cell_content.lower():
                    self.extracted_metadata["policy_expiry_date_raw"] = (
                        cell_content.lower()
                        .replace("policy expiry date", "")
                        .strip()
                    )
                    date_part = self.extracted_metadata[
                        "policy_expiry_date_raw"
                    ]
                    # If date part is empty or unparsable from current cell,
                    # check next cell
                    if not self._parse_date_flexible(date_part):
                        if col_idx + 1 < len(row):
                            next_cell_content = self._clean_text(
                                str(row.iloc[col_idx + 1])
                            )
                            date_part = next_cell_content
                    parsed_date = self._parse_date_flexible(date_part)
                    if parsed_date:
                        self.extracted_metadata["end_date"] = parsed_date
                        logging.info(
                            "Extracted End Date (from Expiry): "
                            f"{self.extracted_metadata['end_date']}"
                        )

                if "class " in cell_content.lower() and not \
                   self.extracted_metadata["class"]:
                    # Match any letter for class
                    match = re.search(r"class\s*([A-Za-z])", cell_content)
                    if match:
                        self.extracted_metadata["class"] = self._clean_text(
                            match.group(1).upper()
                        )
                        logging.info(
                            "Extracted Class: "
                            f"{self.extracted_metadata['class']}"
                        )

                if "deductible" == cell_content.lower() and not \
                   self.extracted_metadata["deductible"]:
                    # Check next cell first for deductible value
                    if col_idx + 1 < len(row):
                        deductible_val = self._clean_text(
                            str(row.iloc[col_idx + 1])
                        )
                        if deductible_val:
                            self.extracted_metadata["deductible"] = (
                                deductible_val
                            )
                            logging.info(
                                "Extracted Deductible: "
                                f"{self.extracted_metadata['deductible']}"
                            )
                    # Fallback: check if value is in the same cell as
                    # "deductible" label
                    if not self.extracted_metadata["deductible"]:
                        potential_val = (
                            cell_content.lower()
                            .replace("deductible", "")
                            .strip()
                        )
                        if potential_val:
                            self.extracted_metadata["deductible"] = (
                                potential_val
                            )
                            logging.info(
                                "Extracted Deductible (from same cell "
                                "fallback): "
                                f"{self.extracted_metadata['deductible']}"
                            )

                if "overall benefit limit" in cell_content.lower() and not \
                   self.extracted_metadata["overall_limit"]:
                    limit_val_str = None
                    # Search subsequent cells in the same row for a numeric
                    # limit value
                    for k in range(col_idx + 1, len(row)):
                        potential_limit = self._clean_numeric_string(
                            str(row.iloc[k])
                        )
                        if potential_limit and \
                           potential_limit.replace(".", "", 1).isdigit():
                            limit_val_str = potential_limit
                            break
                    if limit_val_str:
                        self.extracted_metadata["overall_limit"] = (
                            self._to_float(limit_val_str, None)
                        )
                        logging.info(
                            "Extracted Overall Benefit Limit: "
                            f"{self.extracted_metadata['overall_limit']}"
                        )

            # Optimization: Stop search if key metadata items are found
            if index > 10 and all(
                self.extracted_metadata.get(k)
                for k in [
                    "end_date",
                    "class",
                    "overall_limit",
                    "deductible",
                ]
            ):
                logging.info("Found key metadata, stopping metadata search.")
                break

        if not self.extracted_metadata["end_date"]:
            logging.warning("Could not extract 'End Date' from metadata.")
        if not self.extracted_metadata["class"]:
            logging.warning("Could not extract 'Class' from metadata.")
        if not self.extracted_metadata["overall_limit"]:
            logging.warning("Could not extract 'Overall Limit' from metadata.")
        if not self.extracted_metadata["deductible"]:
            logging.warning("Could not extract 'Deductible' from metadata.")

    def _parse_claims_table_from_df(
        self, df_raw: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Parses claims data from a raw DataFrame, handling different policy
        year sections.

        Args:
            df_raw: The raw DataFrame containing the claims data.

        Returns:
            A structured DataFrame with claims data, or None if parsing fails.
        """
        logging.info("Parsing claims data from the raw DataFrame.")
        claims_data_rows = []

        claims_header_keywords = [
            "monthly claims",
            "number of lives insured",
            "number of paid claims",
            "amount of paid claims",
            "amount of paid claims with vat",
        ]
        header_row_index = -1
        actual_headers: list[str] = []

        # Identify the header row for the claims table
        for i, row in df_raw.iterrows():
            row_str_lower = " ".join(str(x).lower() for x in row.dropna())
            # Heuristic: Check for presence of first few keywords
            if all(
                keyword in row_str_lower
                for keyword in claims_header_keywords[:3]
            ):
                header_row_index = i
                raw_headers = [self._clean_text(str(h)) for h in row]
                # Filter out empty or very short headers
                actual_headers = [h for h in raw_headers if h and len(h) > 2]
                logging.info(
                    f"Found claims header row at index {i}. Raw headers: "
                    f"{raw_headers}, Filtered: {actual_headers}"
                )
                break

        if header_row_index == -1:
            logging.error(
                "Could not find the claims data header row in the combined "
                "table."
            )
            return None

        expected_col_count = 5  # Expected number of data columns for claims
        if len(actual_headers) < expected_col_count and len(actual_headers) > 2:
            logging.warning(
                f"Detected only {len(actual_headers)} headers, expected "
                f"~{expected_col_count}. Will use positional mapping for data."
            )

        # State machine for parsing different policy year sections
        # 0: "2 years Prior" section.
        # 1: "Prior Policy Year" section.
        # 2: "Last Policy Year" section.
        parsing_state = 0
        current_policy_year_label = "2 years Prior"  # Initial section label
        logging.info(
            f"Initial claims parsing state {parsing_state} "
            f"({current_policy_year_label})"
        )

        for i, row_series in df_raw.iloc[header_row_index + 1 :].iterrows():
            row_values = [self._clean_text(str(x)) for x in row_series]

            meaningful_values_in_row = [
                v for v in row_values[:expected_col_count] if v is not None
            ]
            # Skip row if it has too few meaningful values
            if not meaningful_values_in_row or \
               len(meaningful_values_in_row) < 3:
                logging.debug(
                    "Skipping row due to insufficient meaningful values: "
                    f"{row_values}"
                )
                continue

            row_str_lower_full = " ".join(x.lower() for x in row_values if x)
            first_cell_val_cleaned = row_values[0]
            first_cell_val_lower = (
                first_cell_val_cleaned.lower() if first_cell_val_cleaned else ""
            )

            # State transitions based on specific marker strings in first cell
            if "_t] [taleet |" == first_cell_val_lower:  # Prior Policy Year
                parsing_state = 1
                current_policy_year_label = "Prior Policy Year"
                logging.info(
                    f"Marker found, moving to state {parsing_state} "
                    f"({current_policy_year_label})"
                )
                continue
            elif "t_at] [tableee]" == first_cell_val_lower:  # Last Policy Year
                parsing_state = 2
                current_policy_year_label = "Last Policy Year"
                logging.info(
                    f"Marker found, moving to state {parsing_state} "
                    f"({current_policy_year_label})"
                )
                continue
            # Stop conditions for claims data parsing
            elif ("overall benefits" in row_str_lower_full or
                  "benefit tsama" in row_str_lower_full):
                logging.info(
                    "Reached 'Overall Benefits' section, stopping claims "
                    "parsing."
                )
                break
            elif "overall - total" == first_cell_val_lower:  # Skip totals
                logging.info(
                    f"Found 'Overall - Total' row for "
                    f"{current_policy_year_label}, skipping this row."
                )
                continue

            # Data row validation: first col is YYYYMM format or '0'
            is_data_row_candidate = bool(
                re.match(r"^\d{6}$", str(first_cell_val_cleaned))
                or str(first_cell_val_cleaned) == "0"
            )
            if not is_data_row_candidate:
                logging.debug(
                    "Skipping row, not a claims data candidate based on first "
                    f"cell: {row_values}"
                )
                continue

            try:
                monthly_val = first_cell_val_cleaned
                actual_num_insured_lives_for_row = self._to_int(row_values[1])
                if actual_num_insured_lives_for_row is None:
                    logging.warning(
                        "Could not parse 'Number of insured lives' from row: "
                        f"{row_values}. Using 0."
                    )
                    actual_num_insured_lives_for_row = 0

                num_claims_val = self._to_int(row_values[2])
                paid_claims_val = self._to_float(row_values[3])
                paid_claims_vat_val = self._to_float(row_values[4])

                claims_data_rows.append(
                    {
                        "Monthly claims": monthly_val,
                        "Number of insured lives":
                            actual_num_insured_lives_for_row,
                        "Number of claims": num_claims_val,
                        "Amount of paid claims": paid_claims_val,
                        "Amount of paid claims (with VAT)":
                            paid_claims_vat_val,
                        "Policy Year": current_policy_year_label,
                        "End date": self.extracted_metadata["end_date"],
                        "Class": self.extracted_metadata["class"],
                        "Overall Limit":
                            self.extracted_metadata["overall_limit"],
                    }
                )
            except IndexError:
                logging.warning(
                    f"IndexError processing claims row (expected "
                    f"{expected_col_count} data values, got "
                    f"{len(row_values)}): {row_values}"
                )
                continue
            except Exception as e:
                logging.error(
                    f"Generic error processing claims row '{row_values}': {e}",
                    exc_info=True,
                )
                continue

        if not claims_data_rows:
            logging.warning("No data rows extracted for the claims_df.")
            return None

        claims_df = pd.DataFrame(claims_data_rows)
        claims_cols_ordered = [
            "Monthly claims",
            "Number of insured lives",
            "Number of claims",
            "Amount of paid claims",
            "Amount of paid claims (with VAT)",
            "Policy Year",
            "End date",
            "Class",
            "Overall Limit",
        ]
        return claims_df[claims_cols_ordered]

    def _process_benefits_notes_column(
        self, note_series: pd.Series
    ) -> pd.Series:
        """
        Processes the 'Notes' column for the benefits table with specific
        rules.

        Args:
            note_series: The 'Notes' column from the benefits DataFrame.

        Returns:
            Processed 'Notes' column with cleaned text and specific rules
            applied.
        """
        processed_notes: list[str] = []
        for note in note_series:
            cleaned_note = self._clean_text(str(note))
            if not cleaned_note or cleaned_note.lower() == "nan":
                processed_notes.append("No info")
            elif "%" in cleaned_note:
                # Keep the full string if it contains '%'
                processed_notes.append(cleaned_note)
            elif "cesarean" in cleaned_note.lower():
                # Determine coverage for "cesarean" based on keywords
                if "is covered" in cleaned_note.lower() or \
                   "covered" in cleaned_note.lower():
                    processed_notes.append("yes")
                else:
                    processed_notes.append("no")
            else:
                processed_notes.append(cleaned_note)
        return pd.Series(processed_notes, index=note_series.index)

    def _parse_benefits_table(
        self, benefits_html: str
    ) -> Optional[pd.DataFrame]:
        """
        Parses the benefits table from its HTML string.

        Args:
            benefits_html: The HTML string containing the benefits table.

        Returns:
            A structured DataFrame with benefits data, or None if parsing fails.
        """
        logging.info("Parsing benefits table HTML.")
        try:
            dfs = pd.read_html(io.StringIO(benefits_html), flavor="bs4")
        except ValueError as e:
            logging.error(f"pd.read_html failed for benefits_table: {e}.")
            # Fallback check if BeautifulSoup can find a table
            soup = BeautifulSoup(benefits_html, "lxml")
            if not soup.find("table"):
                logging.error("No <table> tag found in benefits_html.")
            return None

        if not dfs:
            logging.error(
                "No DataFrames returned by pd.read_html for benefits table."
            )
            return None

        df_benefits_raw = dfs[0]  # Assume the first table is relevant

        header_row_idx = -1
        # Identify the header row for the benefits table
        for idx, row in df_benefits_raw.iterrows():
            row_str = " ".join(str(x).lower() for x in row if pd.notna(x))
            # Keywords for header
            if "benefit tsama" in row_str or "benefit sama" in row_str:
                header_row_idx = idx
                break

        if header_row_idx == -1:
            logging.error("Could not find header row in benefits table.")
            return None

        df_benefits = df_benefits_raw.iloc[header_row_idx + 1 :].copy()

        cleaned_headers: list[str] = []
        raw_column_names = df_benefits_raw.iloc[header_row_idx]
        for i, c in enumerate(raw_column_names):
            original_cleaned_header = self._clean_text(str(c))
            # Generate placeholder for empty headers
            if not original_cleaned_header:
                cleaned_headers.append(f"col_{i}")
                continue

            normalized_header = " ".join(
                original_cleaned_header.lower().split()
            )

            # Specific header transformations based on content
            if "benefit tsama" == normalized_header:
                final_header = "Benefit_Sama"
            elif "amt of claims (vat)" == normalized_header:
                final_header = "Amount of Claims with VAT"
            else:
                # Cleaned header; renaming handled by rename_map
                final_header = original_cleaned_header
            cleaned_headers.append(final_header)

        df_benefits.columns = cleaned_headers
        df_benefits.reset_index(drop=True, inplace=True)

        # Standardize column names further using a rename map
        rename_map = {
            "Number of Paid Claims": "Number of Claims",
            "Amount of Paid Claims": "Amount of Claims",
            "Notes": "Notes",  # Ensure "Notes" is consistently named
        }
        current_cols = df_benefits.columns.tolist()
        new_cols = [rename_map.get(col, col) for col in current_cols]
        df_benefits.columns = new_cols

        # Remove trailing "overall" summary row if present
        if not df_benefits.empty and "overall" in \
           str(df_benefits.iloc[-1, 0]).lower():
            df_benefits = df_benefits.iloc[:-1]

        if df_benefits.empty:
            logging.warning("Benefits DataFrame is empty after processing.")
            return None

        # Process the 'Notes' column
        if "Notes" in df_benefits.columns:
            df_benefits["Notes"] = self._process_benefits_notes_column(
                df_benefits["Notes"]
            )
        else:
            logging.warning(
                "Column 'Notes' not found in benefits_df, cannot process. "
                "Adding 'Notes' with 'No info'."
            )
            df_benefits["Notes"] = "No info"  # Add Notes column if missing

        # Add common metadata columns
        # Benefits table usually refers to the last policy year
        df_benefits["Policy Year"] = "Last Policy Year"
        df_benefits["End date"] = self.extracted_metadata["end_date"]

        benefits_cols_ordered = [
            "Benefit_Sama",
            "Number of Claims",
            "Amount of Claims",
            "Amount of Claims with VAT",
            "Notes",
            "Policy Year",
            "End date",
        ]

        # Ensure all required columns exist, add if missing, then reorder
        for col_req in benefits_cols_ordered:
            if col_req not in df_benefits.columns:
                logging.warning(
                    f"Benefits DF: Required column '{col_req}' was missing. "
                    "Adding it with None values."
                )
                df_benefits[col_req] = None  # Add missing column

        return df_benefits[benefits_cols_ordered]

    def process_data(
        self,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Main method to process all provided HTML tables and return structured
        DataFrames.

        Returns:
            A tuple containing:
                - claims_df: DataFrame with claims data, or None if parsing
                             fails.
                - benefits_df: DataFrame with benefits data, or None if parsing
                               fails.
        """
        claims_df = None
        benefits_df = None

        # Combined HTML may contain claims, metadata, and limits
        combined_html = self.html_tables.get("claims_table_html")
        # Fallback if claims_table_html is not explicitly provided
        if not combined_html:
            combined_html = self.html_tables.get("metadata_table_html")

        if combined_html:
            try:
                dfs_combined = pd.read_html(
                    io.StringIO(combined_html), flavor="bs4"
                )
                if dfs_combined:
                    # Assume first table is the main one
                    df_combined_raw = dfs_combined[0]
                    self._extract_specific_metadata_from_df(df_combined_raw)
                    claims_df = self._parse_claims_table_from_df(
                        df_combined_raw
                    )
                else:
                    logging.error(
                        "pd.read_html returned no tables for the "
                        "combined_html."
                    )
            except Exception as e:
                logging.error(
                    "Error parsing combined_html with pd.read_html: "
                    f"{e}",
                    exc_info=True,
                )
        else:
            logging.warning("No HTML found for claims/metadata/limits table.")

        # Process benefits table HTML if provided
        benefits_html = self.html_tables.get("benefits_table_html")
        if benefits_html:
            benefits_df = self._parse_benefits_table(benefits_html)
        else:
            logging.warning("No HTML found for benefits table.")

        logging.info("Data processing phase complete.")
        return claims_df, benefits_df


# --- Example Usage ---
if __name__ == "__main__":
    import os

    logging.info("Starting DataProcessor direct test.")
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    test_html_tables: Dict[str, Optional[str]] = {}
    html_files_to_load = {
        "claims_table_html": "claims_table_html.html",
        "benefits_table_html": "benefits_table_html.html",
    }

    # Assumes HTML files are in PROJECT_ROOT/output
    html_output_dir = os.path.join(PROJECT_ROOT, "output")

    for key, filename in html_files_to_load.items():
        filepath = os.path.join(html_output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                test_html_tables[key] = f.read()
            logging.info(f"Loaded HTML for {key} from {filepath}")
        else:
            test_html_tables[key] = None
            logging.warning(f"HTML file not found for {key} at {filepath}")

    if not test_html_tables.get("claims_table_html") and not \
       test_html_tables.get("benefits_table_html"):
        logging.error(
            "No HTML files loaded for testing. Aborting DataProcessor test."
        )
    else:
        processor = DataProcessor(test_html_tables)
        claims_dataframe, benefits_dataframe = processor.process_data()

        output_dir_processed = os.path.join(
            PROJECT_ROOT, "output", "processed_data"
        )
        os.makedirs(output_dir_processed, exist_ok=True)

        if claims_dataframe is not None:
            logging.info("\n--- Claims DataFrame ---")
            print(claims_dataframe.head())
            print(f"\nShape: {claims_dataframe.shape}")
            claims_csv_path = os.path.join(
                output_dir_processed, "claims_df_output.csv"
            )
            claims_dataframe.to_csv(claims_csv_path, index=False)
            logging.info(f"Claims DataFrame saved to {claims_csv_path}")
        else:
            logging.warning("Claims DataFrame is None.")

        if benefits_dataframe is not None:
            logging.info("\n--- Benefits DataFrame ---")
            print(benefits_dataframe.head())
            print(f"\nShape: {benefits_dataframe.shape}")
            benefits_csv_path = os.path.join(
                output_dir_processed, "benefits_df_output.csv"
            )
            benefits_dataframe.to_csv(benefits_csv_path, index=False)
            logging.info(f"Benefits DataFrame saved to {benefits_csv_path}")
        else:
            logging.warning("Benefits DataFrame is None.")

        logging.info("DataProcessor direct test finished.")
        logging.info(
            "Extracted Metadata for reference: "
            f"{processor.extracted_metadata}"
        )