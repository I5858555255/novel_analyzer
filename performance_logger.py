# performance_logger.py
import csv
import os
import time
from threading import Lock

class PerformanceLogger:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        # Singleton pattern to ensure all parts of the app use the same logger instance
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, filename="api_calls.csv", log_dir="logs"):
        # Ensure __init__ is only run once for the singleton
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.log_dir = log_dir
        self.filename = os.path.join(self.log_dir, filename)
        self.fieldnames = [
            "timestamp_utc", "model_id", "api_url",
            "success", "http_status_code", "latency_ms",
            "input_tokens", "output_tokens", "error_message",
            "context_provided" # Boolean: True if context was part of the prompt
        ]

        self._setup_logging()
        self._initialized = True

    def _setup_logging(self):
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            # Write header if file is new or empty
            file_exists = os.path.isfile(self.filename)
            is_empty = file_exists and os.path.getsize(self.filename) == 0

            if not file_exists or is_empty:
                with open(self.filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writeheader()
        except IOError as e:
            print(f"Error setting up performance logger: {e}")


    def log_api_call(self, model_id, api_url, success, http_status_code,
                     latency_ms, input_tokens=0, output_tokens=0,
                     error_message="", context_provided=False):
        try:
            with self._lock: # Ensure thread-safe writes
                with open(self.filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
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
            print(f"Error writing to performance log: {e}")

# Optional: A global instance for easy access, or instantiate where needed.
# For simplicity in a single-process app, a global instance can be fine.
# logger = PerformanceLogger()
# If instantiated globally, ensure it's done safely for threading if modules are loaded by threads.
# Better to instantiate it once in the main application and pass around or use singleton access.
