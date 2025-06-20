"""
Main parsing pipeline for extracting insurance data from PDF documents.

This script orchestrates the OCR extraction of tables from a PDF,
processes the extracted HTML data into structured Pandas DataFrames,
pickles the resulting DataFrames, and saves them to a Redis instance.
It uses command-line arguments for configuration, including input PDF path,
output directory, Redis connection details, and OCR processing options.
"""

import argparse
import logging
import os
import pickle
from typing import Dict, Optional

import pandas as pd

from src.data_processor import DataProcessor
from src.ocr_extractor import OCRExtractor
from src.redis_handler import RedisHandler

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_PICKLE_FILENAME = "dataframes.txt"
DEFAULT_REDIS_KEY = "insurance_data_extract"


def main(
    pdf_file_path: str,
    output_dir: str,
    pickle_filename: str,
    redis_key: str,
    redis_host: str,
    redis_port: int,
    use_gpu: bool,
    show_ocr_log: bool,
):
    """
    Main pipeline to extract data from a PDF, process it, pickle the result,
    and save it to Redis.

    Args:
        pdf_file_path: Absolute path to the input PDF file.
        output_dir: Directory to save output files (e.g., pickled data).
        pickle_filename: Filename for the pickled output.
        redis_key: Redis key for storing the extracted data.
        redis_host: Hostname or IP address of the Redis server.
        redis_port: Port number of the Redis server.
        use_gpu: Flag to enable GPU for OCR processing.
        show_ocr_log: Flag to enable detailed logging from PaddleOCR.
    """
    logging.info(f"Starting main_parser for PDF: {pdf_file_path}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Phase 1: OCR Extraction ---
    logging.info("Initializing OCRExtractor...")
    ocr_extractor = OCRExtractor(
        lang="en", use_gpu=use_gpu, show_log=show_ocr_log
    )
    if ocr_extractor.structure_pipeline is None:
        logging.error("Failed to initialize OCRExtractor. Aborting.")
        return

    logging.info("Extracting tables from PDF...")
    extracted_html_tables: Dict[
        str, Optional[str]
    ] = ocr_extractor.extract_tables_from_pdf(pdf_file_path)

    # --- Phase 2: Data Processing ---
    logging.info("Initializing DataProcessor...")
    data_processor = DataProcessor(extracted_html_tables)

    logging.info("Processing extracted HTML tables into DataFrames...")
    claims_df, benefits_df = data_processor.process_data()

    if claims_df is None:
        logging.warning("Claims DataFrame could not be generated.")
    else:
        logging.info(
            f"Claims DataFrame generated with shape: {claims_df.shape}"
        )

    if benefits_df is None:
        logging.warning("Benefits DataFrame could not be generated.")
    else:
        logging.info(
            f"Benefits DataFrame generated with shape: {benefits_df.shape}"
        )

    # --- Phase 3: Output Formatting and Pickling ---
    # Create the required dictionary structure for output
    output_data_dict: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {
        "claim_experiences": {"claims": claims_df, "benefits": benefits_df}
    }

    # Pickle the dictionary
    pickle_file_path = os.path.join(output_dir, pickle_filename)
    try:
        with open(pickle_file_path, "wb") as f:
            pickle.dump(output_data_dict, f)
        logging.info(
            f"Successfully pickled DataFrames to: {pickle_file_path}"
        )
    except pickle.PicklingError as e:
        logging.error(
            f"Error pickling DataFrames to {pickle_file_path}: {e}"
        )
    except IOError as e:
        logging.error(
            f"IOError writing pickle file {pickle_file_path}: {e}"
        )

    # --- Phase 4: Save to Redis ---
    logging.info(
        f"Initializing RedisHandler (Host: {redis_host}, Port: {redis_port})..."
    )
    redis_handler = RedisHandler(host=redis_host, port=redis_port)

    if redis_handler.redis_client:  # Check if connection was successful
        logging.info(f"Saving data to Redis with key: {redis_key}")
        save_success = redis_handler.save_data(redis_key, output_data_dict)
        if save_success:
            logging.info("Data successfully saved to Redis.")
        else:
            logging.warning("Failed to save data to Redis.")
    else:
        logging.warning(
            "Could not connect to Redis. Data not saved to Redis."
        )

    logging.info(f"Main parser finished for PDF: {pdf_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract tabulated data from PDF, process, pickle, "
            "and save to Redis."
        )
    )
    parser.add_argument("pdf_file", help="Path to the single-page PDF file.")
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save output files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--pickle_file",
        default=DEFAULT_PICKLE_FILENAME,
        help=(
            "Filename for the pickled output "
            f"(default: {DEFAULT_PICKLE_FILENAME})."
        ),
    )
    parser.add_argument(
        "--redis_key",
        default=DEFAULT_REDIS_KEY,
        help=(
            "Redis key for storing the data "
            f"(default: {DEFAULT_REDIS_KEY})."
        ),
    )

    # Get Redis connection details from environment variables or use defaults
    default_redis_host = os.environ.get("REDIS_HOST", "localhost")
    default_redis_port = int(os.environ.get("REDIS_PORT", 6379))

    parser.add_argument(
        "--redis_host",
        default=default_redis_host,
        help=(
            f"Redis host (default: {default_redis_host}, or from "
            "REDIS_HOST env var)."
        ),
    )
    parser.add_argument(
        "--redis_port",
        type=int,
        default=default_redis_port,
        help=(
            f"Redis port (default: {default_redis_port}, or from "
            "REDIS_PORT env var)."
        ),
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help=(
            "Enable GPU for OCR (if available and paddlepaddle-gpu is "
            "installed)."
        ),
    )
    parser.add_argument(
        "--show_ocr_log",
        action="store_true",
        help="Show detailed logs from PaddleOCR.",
    )

    args = parser.parse_args()

    # Determine absolute path for the PDF file
    if not os.path.isabs(args.pdf_file):
        # Assume relative paths are relative to the project root
        pdf_input_path = os.path.join(PROJECT_ROOT, args.pdf_file)
    else:
        pdf_input_path = args.pdf_file

    # Normalize the path to resolve any '..' etc. and ensure it's absolute
    pdf_input_path = os.path.abspath(pdf_input_path)

    if not os.path.exists(pdf_input_path):
        logging.error(f"PDF file not found: {pdf_input_path}")
    else:
        main(
            pdf_file_path=pdf_input_path,
            output_dir=args.output_dir,
            pickle_filename=args.pickle_file,
            redis_key=args.redis_key,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            use_gpu=args.use_gpu,
            show_ocr_log=args.show_ocr_log,
        )