import logging
import os
import pickle
from typing import Any, Dict, Optional

import pandas as pd
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RedisHandler:
    """
    Handles saving and loading Python dictionaries (potentially containing
    Pandas DataFrames) to and from a Redis server using pickling.

    Args:
        host (str): The hostname or IP address of the Redis server.
            Defaults to 'localhost'.
        port (int): The port number of the Redis server. Defaults to 6379.
        db (int): The Redis database number to connect to. Defaults to 0.
    """

    def __init__(
        self, host: str = "localhost", port: int = 6379, db: int = 0
    ):
        self.host = host
        self.port = port
        self.db = db
        self.redis_client: Optional[redis.Redis] = None
        try:
            self.redis_client = redis.Redis(
                host=self.host, port=self.port, db=self.db,
                decode_responses=False # Important for pickle
            )
            self.redis_client.ping()  # Check connection
            logging.info(
                f"Successfully connected to Redis at {self.host}:{self.port}, "
                f"DB: {self.db}"
            )
        except redis.exceptions.ConnectionError as e:
            log_message = (
                f"Could not connect to Redis at {self.host}:{self.port}, "
                f"DB: {self.db}. Error: {e}"
            )
            logging.error(log_message)
            self.redis_client = None  # Ensure it's None if connection failed

    def save_data(self, key: str, data_dict: Dict[str, Any]) -> bool:
        """
        Saves the provided dictionary to Redis after pickling it.

        Args:
            key: The Redis key to save the data under.
            data_dict: The dictionary containing data (e.g., DataFrames)
                to save.

        Returns:
            True if saving was successful, False otherwise.
        """
        if not self.redis_client:
            logging.error("Redis client not available. Cannot save data.")
            return False

        try:
            pickled_data = pickle.dumps(data_dict)
            self.redis_client.set(key, pickled_data)
            logging.info(
                f"Successfully saved data to Redis with key '{key}'."
            )
            return True
        except redis.exceptions.RedisError as e:
            logging.error(
                f"Redis error while saving data with key '{key}': {e}"
            )
            return False
        except pickle.PicklingError as e:
            logging.error(
                f"Error pickling data for Redis key '{key}': {e}"
            )
            return False
        except Exception as e:
            log_message = (
                f"An unexpected error occurred while saving to Redis for key "
                f"'{key}': {e}"
            )
            logging.error(log_message)
            return False

    def load_data(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Loads and unpickles data from Redis for a given key.

        Args:
            key: The Redis key to load data from.

        Returns:
            The unpickled dictionary, or None if key not found or an error
            occurred.
        """
        if not self.redis_client:
            logging.error("Redis client not available. Cannot load data.")
            return None

        try:
            pickled_data = self.redis_client.get(key)
            if pickled_data:
                data_dict = pickle.loads(pickled_data)
                logging.info(
                    f"Successfully loaded data from Redis with key '{key}'."
                )
                return data_dict
            else:
                logging.info(f"No data found in Redis for key '{key}'.")
                return None
        except redis.exceptions.RedisError as e:
            logging.error(
                f"Redis error while loading data for key '{key}': {e}"
            )
            return None
        except pickle.UnpicklingError as e:
            logging.error(
                f"Error unpickling data from Redis key '{key}': {e}"
            )
            return None
        except Exception as e:
            log_message = (
                f"An unexpected error occurred while loading from Redis for "
                f"key '{key}': {e}"
            )
            logging.error(log_message)
            return None

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure Redis server is running
    # (e.g., via Docker: docker run -d -p 6379:6379 redis:alpine)

    # Get Redis host from environment variable or default to localhost
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))

    logging.info(
        f"Attempting to connect to Redis at {redis_host}:{redis_port} "
        "for testing."
    )

    handler = RedisHandler(host=redis_host, port=redis_port)

    if handler.redis_client:  # Proceed only if connection was successful
        # Create some dummy data
        sample_claims_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        sample_benefits_df = pd.DataFrame({"X": ["a", "b"], "Y": ["c", "d"]})

        test_data_to_save: Dict[str, Any] = {
            "claim_experiences": {
                "claims": sample_claims_df,
                "benefits": sample_benefits_df,
            }
        }
        test_key = "test_insurance_data"

        logging.info(f"Saving test data to Redis with key '{test_key}'...")
        save_success = handler.save_data(test_key, test_data_to_save)

        if save_success:
            logging.info(
                f"Loading test data from Redis with key '{test_key}'..."
            )
            loaded_data = handler.load_data(test_key)
            if loaded_data:
                print("\nLoaded Data:")
                # Accessing potentially nested structures safely
                claims_data = loaded_data.get("claim_experiences", {}).get(
                    "claims"
                )
                benefits_data = loaded_data.get("claim_experiences", {}).get(
                    "benefits"
                )

                if claims_data is not None:
                    print("Claims DataFrame head from Redis:")
                    print(claims_data.head())
                else:
                    print("Claims data not found in loaded structure.")

                if benefits_data is not None:
                    print("\nBenefits DataFrame head from Redis:")
                    print(benefits_data.head())
                else:
                    print("Benefits data not found in loaded structure.")
            else:
                print("Failed to load data or key does not exist.")
        else:
            print("Failed to save test data to Redis.")
    else:
        logging.error(
            "Cannot run RedisHandler test: Redis client "
            "initialization failed."
        )