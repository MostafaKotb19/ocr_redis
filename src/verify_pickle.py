"""
Script to verify the pickled data and data stored in Redis.

This script loads data from a specified pickle file (typically 'dataframes.txt'
in the 'output' directory) and from a Redis instance using a predefined key.
It then performs checks on the structure and content of the loaded data,
specifically looking for 'claim_experiences' containing 'claims' and 'benefits'
Pandas DataFrames, and verifies their expected columns.
"""

import logging
import os
import pickle
from typing import Any, Dict, Optional

import pandas as pd

from src.redis_handler import RedisHandler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PICKLE_FILE_PATH = os.path.join(PROJECT_ROOT, "output", "dataframes.txt")
REDIS_KEY = "insurance_data_extract"


def verify_dataframes(
    data_dict: Optional[Dict[str, Any]], source: str
) -> bool:
    """
    Verifies the structure and content of the DataFrames within the loaded
    dictionary.

    Args:
        data_dict: The dictionary loaded from the source (pickle file or Redis).
        source: A string indicating the source of the data (e.g.,
            "Pickle File", "Redis").

    Returns:
        True if verification passes basic structural checks, False otherwise.
    """
    logging.info(f"--- Verifying data loaded from {source} ---")

    if data_dict is None:
        logging.error(f"No data provided from {source} for verification.")
        return False

    if not isinstance(data_dict, dict):
        log_msg = (
            f"Loaded data from {source} is not a dictionary. "
            f"Type: {type(data_dict)}"
        )
        logging.error(log_msg)
        return False

    if "claim_experiences" not in data_dict:
        logging.error(
            f"'claim_experiences' key missing in data from {source}."
        )
        return False

    claim_experiences = data_dict.get("claim_experiences")
    if not isinstance(claim_experiences, dict):
        log_msg = (
            f"'claim_experiences' value is not a dictionary. "
            f"Type: {type(claim_experiences)}"
        )
        logging.error(log_msg)
        return False

    # Verify Claims DataFrame
    if "claims" not in claim_experiences:
        logging.error(
            f"'claims' DataFrame missing in 'claim_experiences' from {source}."
        )
        return False

    claims_df = claim_experiences.get("claims")
    # Handle case where DataFrame might be None if generation failed
    if claims_df is None:
        log_msg = (
            f"Claims DataFrame from {source} is None. This might be "
            "expected if processing failed."
        )
        logging.warning(log_msg)
    elif not isinstance(claims_df, pd.DataFrame):
        log_msg = (
            f"Claims data from {source} is not a Pandas DataFrame. "
            f"Type: {type(claims_df)}"
        )
        logging.error(log_msg)
        return False
    else:
        log_msg = (
            f"Claims DataFrame from {source} loaded successfully. "
            f"Shape: {claims_df.shape}"
        )
        logging.info(log_msg)
        print("\nClaims DataFrame Head:")
        print(claims_df.head())

        expected_claims_cols = [
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
        if not all(col in claims_df.columns for col in expected_claims_cols):
            log_msg = (
                f"Claims DataFrame from {source} is missing some expected "
                f"columns. Found: {claims_df.columns.tolist()}"
            )
            logging.error(log_msg)
            return False
        if claims_df.empty:
            logging.warning(f"Claims DataFrame from {source} is empty.")

    # Verify Benefits DataFrame
    if "benefits" not in claim_experiences:
        logging.error(
            f"'benefits' DataFrame missing in 'claim_experiences' from "
            f"{source}."
        )
        return False

    benefits_df = claim_experiences.get("benefits")
    if benefits_df is None:
        log_msg = (
            f"Benefits DataFrame from {source} is None. This might be "
            "expected if processing failed."
        )
        logging.warning(log_msg)
    elif not isinstance(benefits_df, pd.DataFrame):
        log_msg = (
            f"Benefits data from {source} is not a Pandas DataFrame. "
            f"Type: {type(benefits_df)}"
        )
        logging.error(log_msg)
        return False
    else:
        log_msg = (
            f"Benefits DataFrame from {source} loaded successfully. "
            f"Shape: {benefits_df.shape}"
        )
        logging.info(log_msg)
        print("\nBenefits DataFrame Head:")
        print(benefits_df.head())
        expected_benefits_cols = [
            "Benefit_Sama",
            "Number of Claims",
            "Amount of Claims",
            "Amount of Claims with VAT",
            "Notes",
            "Policy Year",
            "End date",
        ]
        if not all(
            col in benefits_df.columns for col in expected_benefits_cols
        ):
            log_msg = (
                f"Benefits DataFrame from {source} is missing some expected "
                f"columns. Found: {benefits_df.columns.tolist()}"
            )
            logging.error(log_msg)
            return False
        if benefits_df.empty:
            logging.warning(f"Benefits DataFrame from {source} is empty.")

    logging.info(f"Verification for data from {source} complete.")
    return True


if __name__ == "__main__":
    # 1. Verify data from the pickle file (dataframes.txt)
    logging.info(
        "Attempting to load and verify data from pickle file: "
        f"{PICKLE_FILE_PATH}"
    )
    loaded_from_file: Optional[Dict[str, Any]] = None
    if os.path.exists(PICKLE_FILE_PATH):
        try:
            with open(PICKLE_FILE_PATH, "rb") as f:
                loaded_from_file = pickle.load(f)
            logging.info("Successfully unpickled data from file.")
            verify_dataframes(loaded_from_file, "Pickle File")
        except pickle.UnpicklingError as e:
            logging.error(
                f"Error unpickling data from file {PICKLE_FILE_PATH}: {e}"
            )
        except Exception as e:
            logging.error(
                f"An unexpected error occurred reading pickle file: {e}"
            )
    else:
        logging.error(
            f"Pickle file not found at {PICKLE_FILE_PATH}. Cannot verify."
        )

    print("\n" + "=" * 50 + "\n")  # Separator

    # 2. Verify data from Redis
    logging.info(
        f"Attempting to load and verify data from Redis (Key: '{REDIS_KEY}')"
    )
    # Get Redis host from environment variable or default to localhost
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))

    redis_handler = RedisHandler(host=redis_host, port=redis_port)
    if redis_handler.redis_client:
        loaded_from_redis = redis_handler.load_data(REDIS_KEY)
        if loaded_from_redis:
            verify_dataframes(loaded_from_redis, "Redis")
        else:
            logging.warning(
                f"No data loaded from Redis for key '{REDIS_KEY}'. "
                "Verification skipped."
            )
    else:
        logging.error(
            "Could not connect to Redis. Cannot verify Redis data."
        )