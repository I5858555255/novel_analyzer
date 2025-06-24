"""
Handles interactions with Large Language Models (LLMs) for text summarization.

This module defines the LLMProcessor class, which is responsible for
sending text to a configured LLM API, handling responses, retries,
and basic error management. It also integrates with PerformanceLogger
to record metrics about each API call.
"""
import requests
import time
import re
import tiktoken  # For token calculation fallbacks
from performance_logger import PerformanceLogger
import hashlib # For cache key
import json    # For cache key
import os      # For cache directory path
try:
    from diskcache import Cache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    Cache = None # Placeholder if diskcache is not installed

# Define a cache directory
# Try to place it in a user-specific cache location if possible, else local.
try:
    CACHE_DIR_BASE = os.path.join(os.path.expanduser("~"), ".cache", "novel_analyzer")
except Exception:
    CACHE_DIR_BASE = os.path.join(os.getcwd(), ".cache", "novel_analyzer")
LLM_CACHE_DIR = os.path.join(CACHE_DIR_BASE, "llm_responses")
if DISKCACHE_AVAILABLE and not os.path.exists(LLM_CACHE_DIR):
    os.makedirs(LLM_CACHE_DIR, exist_ok=True)


class LLMProcessor:
    """
    Processes text by sending it to a Large Language Model (LLM) for summarization.

    Manages API configuration, request preparation, response handling,
    error retries, and performance logging for LLM interactions.
    """

    def __init__(self, api_config, custom_prompt_text=None, encoding_object=None):
        """
        Initializes the LLMProcessor.

        Args:
            api_config (dict): Configuration for the LLM API. Must contain 'url' and 'model'.
                               Can optionally contain 'key'.
            custom_prompt_text (str, optional): A custom prompt template to use for summarization.
                                                If None, a default template is used.
            encoding_object (tiktoken.Encoding, optional): A Tiktoken encoding object.
                                                           If None, token calculations might be rough estimates.
        Raises:
            ValueError: If api_config is invalid or missing required fields,
                        or if encoding_object is not provided.
        """
        if not isinstance(api_config, dict):
            raise ValueError("LLMProcessor: api_config must be a dictionary.")
        if not api_config.get('url') or not isinstance(api_config.get('url'), str) or not api_config.get('url').strip():
            raise ValueError(
                "LLMProcessor: api_config must contain a non-empty 'url' string.")
        if not api_config.get('model') or not isinstance(api_config.get('model'), str) or not api_config.get('model').strip():
            raise ValueError(
                "LLMProcessor: api_config must contain a non-empty 'model' string.")

        self.api_url = api_config['url']
        self.api_key = api_config.get('key', "")  # API key can be optional for some local models
        self.model = api_config['model']
        self.custom_prompt_for_processor = custom_prompt_text
        self.encoding = encoding_object # Tiktoken encoding object
        if self.encoding is None: # An encoding object is crucial for accurate token counting
            raise ValueError(
                "LLMProcessor requires a valid Tiktoken encoding object.")
        # self.max_tokens_override = max_tokens_override # Removed

        self.session = requests.Session()  # Use a session for potential connection pooling
        self.DEFAULT_MAX_RETRIES = 3
        self.TRANSIENT_MAX_RETRIES = 50
        self.INITIAL_BACKOFF_FACTOR = 1  # In seconds
        self.last_call = 0  # Timestamp of the last API call, for rate limiting

        try:
            self.perf_logger = PerformanceLogger()
        except NameError:
            print("Warning: PerformanceLogger class not found. Logging will be basic.")
            self.perf_logger = None
        except Exception as e_perf_init:
            print(f"Error initializing PerformanceLogger: {e_perf_init}. Logging will be basic.")
            self.perf_logger = None

        if DISKCACHE_AVAILABLE:
            if not hasattr(LLMProcessor, '_shared_cache'):
                try:
                    LLMProcessor._shared_cache = Cache(LLM_CACHE_DIR, size_limit=2**28) # 256MB cache limit
                    print(f"DiskCache initialized at: {LLM_CACHE_DIR}")
                except Exception as e:
                    print(f"Error initializing DiskCache at {LLM_CACHE_DIR}: {e}. Cache will be disabled.")
                    LLMProcessor._shared_cache = None
            self.cache = LLMProcessor._shared_cache
        else:
            self.cache = None
            if not hasattr(LLMProcessor, '_diskcache_warning_shown'):
                print("Warning: diskcache library not found. LLM response caching will be disabled. Install with: pip install diskcache")
                LLMProcessor._diskcache_warning_shown = True


    def _generate_cache_key(self, prompt_content, model_name, temperature, top_p):
        # Ensure all parts of the key are strings or simple types that json.dumps can handle
        key_data = {
            "prompt_content_hash": hashlib.md5(prompt_content.encode('utf-8')).hexdigest(), # Hash long prompt
            "model": str(model_name),
            "temp": float(temperature), # Ensure consistent type
            "top_p": float(top_p)       # Ensure consistent type
            # Consider adding a version number to the key if prompt structures change often
            # "cache_version": "1.0"
        }
        canonical_key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(canonical_key_string.encode('utf-8')).hexdigest()

    def calculate_tokens(self, text):
        """
        Calculates the number of tokens for a given text.

        Uses the provided Tiktoken encoding object. If unavailable or if encoding fails,
        falls back to a rough estimation method based on character counts.

        Args:
            text (str): The text to calculate tokens for.

        Returns:
            int: The estimated number of tokens.
        """
        if not self.encoding:
            # Fallback if encoding object is somehow missing after init (should not happen)
            print(
                "Warning: Tiktoken encoding not available in calculate_tokens. Using rough estimate.")
            chinese_chars = len(re.findall(r'[一-鿿]', text))  # Count Chinese characters
            other_chars = len(text) - chinese_chars
            # Rough estimation: Chinese chars often take more tokens than English chars
            return int(chinese_chars / 1.5 + other_chars / 4)
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            # Handle potential errors during encoding, e.g., special characters
            error_message = f"Error during tiktoken.encode: {str(e)}. "
            error_message += f"Problematic text (first 100 chars): '{text[:100]}'. "
            error_message += "Falling back to rough token estimation."
            print(error_message)
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text)) # More specific Unicode range for CJK
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4)

    def summarize(self, text, context="", max_retries=None):
        """
        Summarizes the given text using the configured LLM API.

        Handles prompt construction, API call execution, retries with exponential backoff
        for transient errors (like rate limits), and parsing of the response.

        Args:
            text (str): The text content to be summarized.
            context (str, optional): Additional context to provide to the LLM. Defaults to "".
            max_retries (int, optional): Maximum number of retries for the API call.
                                         Defaults to self.MAX_RETRIES.

        Returns:
            tuple: A tuple containing:
                - str: The summarized text. Empty if summarization fails.
                - int: The number of input tokens consumed.
                - int: The number of output tokens generated.

        Raises:
            RuntimeError: If API calls fail after all retries, or if a non-transient
                          API error occurs (e.g., invalid API key, model not found).
            PermissionError: If the API key is invalid.
            ValueError: If the model is not found by the API or the API response is malformed.
        """
        if not text:
            if self.perf_logger:
                self.perf_logger.log_api_call(
                    model_id=self.model, api_url=self.api_url, success=False,
                    http_status_code=None, latency_ms=0,
                    error_message="Input text is empty.", context_provided=bool(context)
                )
            return "", 0, 0  # Return empty summary and zero tokens for empty input

        call_start_time_overall = time.time() # For overall call, not used currently but good for future detailed logging

        # Basic rate limiting: ensure at least 1 second between starts of calls from this instance
        if time.time() - self.last_call < 1.0:
            time.sleep(1.0 - (time.time() - self.last_call))

        effective_max_overall_retries = max_retries if max_retries is not None else self.TRANSIENT_MAX_RETRIES
        last_error_message = "No error message captured"  # Initialize last_error_message
        default_prompt_template = "提炼以下文本的核心要点，仅输出提炼后的内容，不要包含任何额外解释或与原文无关的文字。保留关键情节和人物关系："
        effective_prompt_template = self.custom_prompt_for_processor if self.custom_prompt_for_processor else default_prompt_template

        prompt = f"{effective_prompt_template}\n{text}"
        if context:
            prompt = f"上下文：{context}\n\n{prompt}"

        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}

        # Estimate max_tokens: 1.5% of text length, capped between 100 and 4000
        # This is a heuristic and might need adjustment based on model and language.
        placeholder_text_for_full_prompt_mode = "请根据以上提供的完整指令和文本内容进行分析。"
        if text == placeholder_text_for_full_prompt_mode and self.custom_prompt_for_processor:
            text_len_for_max_tokens = len(self.custom_prompt_for_processor)
        else:
            text_len_for_max_tokens = len(text) if text else 1

        data = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,  # Lower temperature for more deterministic, focused summaries
                "top_p": 0.8}        # Nucleus sampling parameter

        # Caching logic
        cache_key = None
        if self.cache:
            try:
                cache_key = self._generate_cache_key(prompt, self.model, data["temperature"], data["top_p"])
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    # print(f"LLM Cache HIT for model {self.model}, key prefix {cache_key[:8]}") # For debugging
                    summary_text_final, input_tokens_final, output_tokens_final = cached_result
                    if self.perf_logger:
                        self.perf_logger.log_api_call(
                            model_id=self.model, api_url=self.api_url, success=True, http_status_code=200,
                            latency_ms=1, input_tokens=input_tokens_final, output_tokens=output_tokens_final,
                            context_provided=bool(context), error_message="CACHED_RESPONSE"
                        )
                    return summary_text_final, input_tokens_final, output_tokens_final
            except Exception as e_cache_get:
                print(f"Error during cache get: {e_cache_get}. Proceeding without cache.")
        # else:
            # print(f"LLM Cache MISS for model {self.model}, key prefix {cache_key[:8]}" if cache_key else "Cache not available or miss.")


        summary_text_final = ""
        input_tokens_final = 0
        output_tokens_final = 0

        for attempt in range(effective_max_overall_retries):
            call_attempt_start_time = time.time()
            response_obj = None
            http_status_to_log = None

            try:
                response_obj = self.session.post(
                    self.api_url, headers=headers, json=data, timeout=30)
                http_status_to_log = response_obj.status_code
                response_obj.raise_for_status()
                result = response_obj.json()

                if 'error' in result:
                    error_msg_detail = result['error'].get('message', 'Unknown API error in response')
                    if 'model' in error_msg_detail.lower() or result['error'].get('code') == 'model_not_found':
                        raise ValueError(f"Model {self.model} not found (API Error: {error_msg_detail})")
                    if 'api_key' in error_msg_detail.lower() or result['error'].get('code') == 'invalid_api_key':
                        raise PermissionError(f"Invalid API key (API Error: {error_msg_detail})")
                    last_error_message = f"API returned error: {error_msg_detail}"
                    if http_status_to_log is None : http_status_to_log = 500
                    raise requests.exceptions.HTTPError(f"Error in API response JSON: {error_msg_detail}", response=response_obj)

                if 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                    summary_text_final = result['choices'][0]['message']['content']
                elif "claude" in self.model.lower() and "content" in result and isinstance(result["content"], list) and len(result["content"]) > 0 and "text" in result["content"][0]:
                    summary_text_final = result["content"][0]["text"]
                else:
                    raise ValueError("API response format error or empty content.")

                usage = result.get('usage', {})
                input_tokens_final = usage.get('prompt_tokens', 0)
                output_tokens_final = usage.get('completion_tokens', 0)

                if input_tokens_final == 0: input_tokens_final = self.calculate_tokens(prompt)
                if output_tokens_final == 0: output_tokens_final = self.calculate_tokens(summary_text_final)

                if self.cache and cache_key:
                    try:
                        self.cache.set(cache_key, (summary_text_final, input_tokens_final, output_tokens_final), expire=None) # Persist indefinitely unless evicted
                        # print(f"LLM Result STORED to cache for model {self.model}, key prefix {cache_key[:8]}") # For debugging
                    except Exception as e_cache_set:
                        print(f"Error during cache set: {e_cache_set}")

                if self.perf_logger:
                    self.perf_logger.log_api_call(model_id=self.model, api_url=self.api_url, success=True, http_status_code=http_status_to_log, latency_ms=(time.time() - call_attempt_start_time) * 1000, input_tokens=input_tokens_final, output_tokens=output_tokens_final, context_provided=bool(context))

                self.last_call = time.time()
                return summary_text_final, input_tokens_final, output_tokens_final

            except requests.exceptions.HTTPError as e_http:
                http_status_to_log = e_http.response.status_code if e_http.response is not None else None
                last_error_message = f"HTTP Error {http_status_to_log}: {str(e_http)}"
                if self.perf_logger:
                    self.perf_logger.log_api_call(model_id=self.model, api_url=self.api_url, success=False, http_status_code=http_status_to_log, latency_ms=(time.time() - call_attempt_start_time) * 1000, error_message=last_error_message + " (will retry if applicable)", context_provided=bool(context))

                if http_status_to_log == 429 or (http_status_to_log is not None and 500 <= http_status_to_log <= 599): # Transient errors
                    if attempt == effective_max_overall_retries - 1:
                        self.last_call = time.time()
                        raise RuntimeError(f"{last_error_message}. Max transient retries ({effective_max_overall_retries}) reached.")
                    wait_time = self.INITIAL_BACKOFF_FACTOR * (2 ** attempt)
                    if http_status_to_log == 429:
                        retry_after_seconds_str = e_http.response.headers.get("Retry-After")
                        if retry_after_seconds_str:
                            try:
                                wait_time = int(retry_after_seconds_str)
                            except ValueError:
                                pass # Use calculated exponential backoff
                    time.sleep(wait_time)
                    continue
                elif http_status_to_log in [401, 403, 404]: # Fatal/config errors
                    if attempt >= self.DEFAULT_MAX_RETRIES - 1:
                        self.last_call = time.time()
                        raise RuntimeError(f"{last_error_message}. Max retries ({self.DEFAULT_MAX_RETRIES}) reached for fatal HTTP error {http_status_to_log}.")
                    time.sleep(self.INITIAL_BACKOFF_FACTOR * (2 ** attempt)) # Still backoff before retry
                    continue
                else: # Other HTTP errors, treat as fatal with default retries
                    if attempt >= self.DEFAULT_MAX_RETRIES - 1:
                        self.last_call = time.time()
                        raise RuntimeError(f"{last_error_message}. Max retries ({self.DEFAULT_MAX_RETRIES}) reached for HTTP error {http_status_to_log}.")
                    time.sleep(self.INITIAL_BACKOFF_FACTOR * (2 ** attempt))
                    continue

            except requests.exceptions.RequestException as e_req: # Network errors (transient)
                last_error_message = f"Network Request Failed: {str(e_req)}"
                if self.perf_logger:
                    self.perf_logger.log_api_call(model_id=self.model, api_url=self.api_url, success=False, http_status_code=None, latency_ms=(time.time() - call_attempt_start_time) * 1000, error_message=last_error_message + " (will retry if applicable)", context_provided=bool(context))
                if attempt == effective_max_overall_retries - 1:
                    self.last_call = time.time()
                    raise RuntimeError(f"{last_error_message}. Max transient retries ({effective_max_overall_retries}) reached for network error.")
                time.sleep(self.INITIAL_BACKOFF_FACTOR * (2 ** attempt))
                continue

            except (ValueError, PermissionError) as e_api_resp_fatal: # Fatal errors from response processing
                last_error_message = f"API Call/Processing Error: {str(e_api_resp_fatal)}"
                if self.perf_logger:
                    self.perf_logger.log_api_call(model_id=self.model, api_url=self.api_url, success=False, http_status_code=http_status_to_log, latency_ms=(time.time() - call_attempt_start_time) * 1000, error_message=last_error_message + " (will retry if applicable)", context_provided=bool(context))
                if attempt >= self.DEFAULT_MAX_RETRIES - 1:
                    self.last_call = time.time()
                    raise RuntimeError(f"{last_error_message}. Max retries ({self.DEFAULT_MAX_RETRIES}) reached for API response error.")
                time.sleep(self.INITIAL_BACKOFF_FACTOR * (2 ** attempt))
                continue

            except Exception as e_other: # Other unexpected errors (treat as fatal with default retries)
                last_error_message = f"API Call/Processing Error: {str(e_other)}"
                # http_status_to_log might be None here if error happened before HTTP call attempt or after response
                if self.perf_logger:
                    self.perf_logger.log_api_call(model_id=self.model, api_url=self.api_url, success=False, http_status_code=http_status_to_log, latency_ms=(time.time() - call_attempt_start_time) * 1000, error_message=last_error_message + " (will retry if applicable)", context_provided=bool(context))
                if attempt >= self.DEFAULT_MAX_RETRIES - 1:
                    self.last_call = time.time()
                    raise RuntimeError(f"{last_error_message}. Max retries ({self.DEFAULT_MAX_RETRIES}) reached for unexpected error.")
                time.sleep(self.INITIAL_BACKOFF_FACTOR * (2 ** attempt))
                continue

        # If loop completes, it means all retries failed
        self.last_call = time.time()
        raise RuntimeError(f"API call failed after {effective_max_overall_retries} attempts. Last error: {last_error_message}")
