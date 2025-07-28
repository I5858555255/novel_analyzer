"""
Provides a singleton PerformanceLogger for recording API call metrics.

This module defines the PerformanceLogger class, which is responsible for
logging details of API calls (like latency, tokens used, success/failure)
to a CSV file for performance analysis and monitoring.
"""
import os
import threading
import time
import csv


class PerformanceLogger:
    """
    Singleton class to log performance metrics of API calls to a CSV file.

    Ensures that only one instance of the logger exists across the application.
    It handles creating log directories and files, and provides a thread-safe
    method to write log entries.
    """
    _instance = None
    _lock = threading.Lock()  # Lock for thread-safe singleton creation and file writing

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of PerformanceLogger is created (Singleton pattern).
        """
        if not cls._instance:
            with cls._lock:
                # Double-check locking
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, filename="api_calls.csv", log_dir="logs"):
        """
        Initializes the PerformanceLogger instance.

        This method is called only once due to the singleton implementation in __new__.
        It sets up the log file path and ensures the CSV header is written if the
        file is new or empty.

        Args:
            filename (str, optional): The name of the CSV log file.
                                      Defaults to "api_calls.csv".
            log_dir (str, optional): The directory where the log file will be stored.
                                     Defaults to "logs".
        """
        # The hasattr check ensures __init__ logic runs only once per instance
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.log_dir = log_dir
        self.filename = os.path.join(self.log_dir, filename)
        self.fieldnames = ["timestamp_utc", "model_id", "api_url", "success", "http_status_code",
                           "latency_ms", "input_tokens", "output_tokens", "error_message", "context_provided"]
        self._setup_logging()
        self._initialized = True

    def _setup_logging(self):
        """
        Sets up the logging directory and the CSV log file.

        Creates the log directory if it doesn't exist.
        If the log file doesn't exist or is empty, it writes the CSV header row.
        """
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)  # Create log directory if it's missing
            file_exists = os.path.isfile(self.filename)
            # Check if file is empty, which can happen if creation was interrupted
            is_empty = file_exists and os.path.getsize(self.filename) == 0
            if not file_exists or is_empty:
                with open(self.filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(
                        csvfile, fieldnames=self.fieldnames)
                    writer.writeheader()
        except IOError as e:
            # Print error if setup fails, as logging is crucial for monitoring
            print(f"Error setting up performance logger: {e}")

    def log_api_call(self, model_id, api_url, success, http_status_code, latency_ms,
                     input_tokens=0, output_tokens=0, error_message="", context_provided=False):
        """
        Logs a single API call event to the CSV file in a thread-safe manner.

        Args:
            model_id (str): Identifier of the model used for the API call.
            api_url (str): The URL of the API endpoint.
            success (bool): True if the API call was successful, False otherwise.
            http_status_code (int, optional): The HTTP status code of the response.
            latency_ms (float, optional): Latency of the API call in milliseconds.
            input_tokens (int, optional): Number of input tokens used. Defaults to 0.
            output_tokens (int, optional): Number of output tokens generated. Defaults to 0.
            error_message (str, optional): Error message if the call failed. Defaults to "".
            context_provided (bool, optional): Whether context was provided for the call. Defaults to False.
        """
        try:
            with self._lock:  # Ensure thread-safe write to the CSV
                with open(self.filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(
                        csvfile, fieldnames=self.fieldnames)
                    log_entry = {
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "model_id": model_id,
                        "api_url": api_url,
                        "success": success,
                        "http_status_code": http_status_code if http_status_code is not None else "",
                        "latency_ms": f"{latency_ms:.2f}" if latency_ms is not None else "",
                        "input_tokens": input_tokens if input_tokens else 0,
                        "output_tokens": output_tokens if output_tokens else 0,
                        "error_message": error_message if error_message else "",
                        "context_provided": context_provided
                    }
                    writer.writerow(log_entry)
        except IOError as e:
            # Print error if logging fails to ensure visibility of logging issues
            print(f"Error writing to performance log: {e}")
