# llm_processor.py
import re
import time
import requests
import tiktoken # tiktoken import is needed for type hinting if used, but not for direct use in __init__
# from performance_logger import PerformanceLogger # Assuming this was added and should be kept

class LLMProcessor:
    def __init__(self, api_config, custom_prompt_text=None, encoding_object=None):
        # print(f"DEBUG: LLMProcessor.__init__ called with api_config type: {type(api_config)}, custom_prompt_text type: {type(custom_prompt_text)}, encoding_object type: {type(encoding_object)}") # Optional detailed debug

        # Validate api_config (this was a proactive fix from a previous step, ensure it's here)
        if not isinstance(api_config, dict):
            raise ValueError("LLMProcessor: api_config must be a dictionary.")
        if not api_config.get('url') or not isinstance(api_config.get('url'), str) or not api_config.get('url').strip():
            raise ValueError("LLMProcessor: api_config must contain a non-empty 'url' string.")
        if not api_config.get('model') or not isinstance(api_config.get('model'), str) or not api_config.get('model').strip():
            raise ValueError("LLMProcessor: api_config must contain a non-empty 'model' string.")
        # 'key' can sometimes be optional for local models, but generally expected.
        # if not api_config.get('key') or not isinstance(api_config.get('key'), str): # Key can be empty string for some public APIs
            # print("Warning: LLMProcessor api_config does not contain 'key' or it's not a string.")


        self.api_url = api_config['url']
        self.api_key = api_config.get('key', "") # Default to empty string if key is missing
        self.model = api_config['model']
        self.custom_prompt_for_processor = custom_prompt_text

        # Use the passed-in encoding_object
        self.encoding = encoding_object
        if self.encoding is None:
            # This case should ideally be prevented by the calling code (MainWindow)
            # which should verify encoding_object before instantiating LLMProcessor.
            # However, as a safeguard:
            print("ERROR: LLMProcessor initialized without a valid Tiktoken encoding object!")
            # Option 1: Raise an error to make it explicit
            raise ValueError("LLMProcessor requires a valid Tiktoken encoding object.")
            # Option 2: Try to get a default one here (less ideal as it defeats centralization)
            # print("Warning: LLMProcessor trying to get default 'cl100k_base' due to None encoding_object.")
            # try:
            #    self.encoding = tiktoken.get_encoding('cl100k_base')
            # except Exception as e_default_tiktoken:
            #    print(f"CRITICAL: Failed to get default tiktoken encoder in LLMProcessor: {e_default_tiktoken}")
            #    raise ValueError("LLMProcessor could not obtain any Tiktoken encoder.") from e_default_tiktoken

        self.session = requests.Session() # For connection pooling
        self.MAX_RETRIES = 3
        self.INITIAL_BACKOFF_FACTOR = 1
        self.last_call = 0

        # Instantiate PerformanceLogger - ensure this import is present
        try:
            from performance_logger import PerformanceLogger
            self.perf_logger = PerformanceLogger()
        except ImportError:
            print("Warning: PerformanceLogger module not found. Performance logging will be disabled.")
            self.perf_logger = None # Or a dummy logger class

    # Ensure calculate_tokens and summarize methods are correctly defined below,
    # using self.encoding, self.session, self.perf_logger etc. as established.
    # (The content of calculate_tokens and summarize is not changed by this subtask,
    # only __init__ is the focus).

    def calculate_tokens(self, text):
        if not self.encoding: # Should not happen if __init__ raises ValueError
            print("Warning: Tiktoken encoding not available in calculate_tokens. Using rough estimate.")
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4)
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            print(f"Error in calculate_tokens with tiktoken: {e}. Using rough estimate.")
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4)

    def summarize(self, text, context="", max_retries=None):
        # (Ensure the summarize method uses self.session, self.perf_logger, etc.
        #  The previous implementation of summarize with detailed retry logic should be here.)
        # This is a simplified placeholder to ensure the subtask focuses on __init__.
        # The actual summarize method is more complex and was defined in a previous step.
        # This subtask should NOT alter the existing summarize method's core logic,
        # only ensure __init__ is correct.

        # For the purpose of this subtask, the existing summarize method from
        # the previous iteration (with session, perf_logger, detailed retry) is assumed
        # to be present and correct. The only change is that self.encoding is now
        # guaranteed by __init__ (or __init__ would have raised an error).

        if not text: # Added initial check for empty input text
            if self.perf_logger:
                self.perf_logger.log_api_call(
                    model_id=self.model, api_url=self.api_url, success=False,
                    http_status_code=None, latency_ms=0,
                    error_message="Input text is empty.", context_provided=bool(context)
                )
            return "", 0, 0

        call_start_time_overall = time.time() # For the whole summarize operation including internal rate limit wait

        # Internal rate limiting per LLMProcessor instance
        if time.time() - self.last_call < 1.0: # Ensure at least 1 sec between starts of calls for this instance
            time.sleep(1.0 - (time.time() - self.last_call))

        effective_max_retries = max_retries if max_retries is not None else self.MAX_RETRIES

        default_prompt_template = "提炼以下文本的核心要点，仅输出提炼后的内容，不要包含任何额外解释或与原文无关的文字。保留关键情节和人物关系，压缩至原文1%字数："
        effective_prompt_template = self.custom_prompt_for_processor if self.custom_prompt_for_processor else default_prompt_template
        prompt = f"{effective_prompt_template}\n{text}"

        if context:
            prompt = f"上下文：{context}\n\n{prompt}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Ensure text has length for max_tokens calculation
        text_len_for_max_tokens = len(text) if text else 1
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": min(4000, max(100, int(text_len_for_max_tokens * 0.015)))
        }

        summary_text_final = ""
        input_tokens_final = 0
        output_tokens_final = 0

        for attempt in range(effective_max_retries):
            call_attempt_start_time = time.time()
            response_obj = None # To store response for logging if error occurs after getting response
            http_status_to_log = None

            try:
                response_obj = self.session.post(
                    self.api_url, headers=headers, json=data, timeout=30
                )
                http_status_to_log = response_obj.status_code

                if http_status_to_log == 429:
                    retry_after_seconds_str = response_obj.headers.get("Retry-After")
                    wait_time = self.INITIAL_BACKOFF_FACTOR * (2 ** attempt)
                    if retry_after_seconds_str:
                        try:
                            wait_time = int(retry_after_seconds_str)
                            print(f"API rate limit (429). Retrying after {wait_time}s (Retry-After header).")
                        except ValueError:
                            print(f"API rate limit (429). Invalid Retry-After. Default backoff: {wait_time}s.")
                    else:
                        print(f"API rate limit (429). No Retry-After. Default backoff: {wait_time}s.")

                    if attempt == effective_max_retries - 1:
                        raise RuntimeError(f"API rate limit (429). Max retries reached.")

                    # Log this attempt before sleeping
                    if self.perf_logger:
                        self.perf_logger.log_api_call(
                            model_id=self.model, api_url=self.api_url, success=False,
                            http_status_code=429, latency_ms=(time.time() - call_attempt_start_time) * 1000,
                            error_message="HTTP 429 Too Many Requests (will retry)", context_provided=bool(context)
                        )
                    time.sleep(wait_time)
                    continue

                response_obj.raise_for_status() # Other 4xx/5xx errors

                result = response_obj.json()

                if 'error' in result:
                    error_msg_detail = result['error'].get('message', 'Unknown API error in response')
                    # Specific error checks to prevent retrying unrecoverable errors
                    if 'model' in error_msg_detail.lower() or result['error'].get('code') == 'model_not_found':
                        raise ValueError(f"Model {self.model} not found (API Error: {error_msg_detail})")
                    if 'api_key' in error_msg_detail.lower() or result['error'].get('code') == 'invalid_api_key':
                        raise PermissionError(f"Invalid API key (API Error: {error_msg_detail})")
                    raise RuntimeError(f"API returned error: {error_msg_detail}")

                if 'choices' in result and len(result['choices']) > 0 and \
                   'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                    summary_text_final = result['choices'][0]['message']['content']
                elif "claude" in self.model.lower() and "content" in result and \
                     isinstance(result["content"], list) and len(result["content"]) > 0 and \
                     "text" in result["content"][0]:
                    summary_text_final = result["content"][0]["text"]
                else:
                    raise ValueError("API response format error or empty content.")

                usage = result.get('usage', {})
                input_tokens_final = usage.get('prompt_tokens', 0)
                output_tokens_final = usage.get('completion_tokens', 0)

                if input_tokens_final == 0: input_tokens_final = self.calculate_tokens(prompt)
                if output_tokens_final == 0: output_tokens_final = self.calculate_tokens(summary_text_final)

                if self.perf_logger:
                    self.perf_logger.log_api_call(
                        model_id=self.model, api_url=self.api_url, success=True,
                        http_status_code=http_status_to_log, latency_ms=(time.time() - call_attempt_start_time) * 1000,
                        input_tokens=input_tokens_final, output_tokens=output_tokens_final,
                        context_provided=bool(context)
                    )
                self.last_call = time.time()
                return summary_text_final, input_tokens_final, output_tokens_final

            except requests.exceptions.HTTPError as e_http:
                err_msg = f"HTTP Error {e_http.response.status_code}: {str(e_http)}"
                if self.perf_logger:
                    self.perf_logger.log_api_call(
                        model_id=self.model, api_url=self.api_url, success=False,
                        http_status_code=http_status_to_log, latency_ms=(time.time() - call_attempt_start_time) * 1000,
                        error_message=err_msg, context_provided=bool(context)
                    )
                if attempt == effective_max_retries - 1: self.last_call = time.time(); raise RuntimeError(f"{err_msg}. Max retries reached.")
                time.sleep(self.INITIAL_BACKOFF_FACTOR * (2 ** attempt))
            except requests.exceptions.RequestException as e_req: # Network errors, timeout
                err_msg = f"Network Request Failed: {str(e_req)}"
                if self.perf_logger:
                     self.perf_logger.log_api_call(
                        model_id=self.model, api_url=self.api_url, success=False,
                        http_status_code=None, latency_ms=(time.time() - call_attempt_start_time) * 1000,
                        error_message=err_msg, context_provided=bool(context)
                    )
                if attempt == effective_max_retries - 1: self.last_call = time.time(); raise RuntimeError(f"{err_msg}. Max retries reached.")
                time.sleep(self.INITIAL_BACKOFF_FACTOR * (2 ** attempt))
            except Exception as e_other: # JSONDecodeError, PermissionError, ValueError, RuntimeError from result parsing
                err_msg = f"API Call/Processing Error: {str(e_other)}"
                if self.perf_logger:
                    self.perf_logger.log_api_call(
                        model_id=self.model, api_url=self.api_url, success=False,
                        http_status_code=http_status_to_log, # May be None if error before response
                        latency_ms=(time.time() - call_attempt_start_time) * 1000,
                        error_message=err_msg, context_provided=bool(context)
                    )
                if attempt == effective_max_retries - 1: self.last_call = time.time(); raise RuntimeError(f"{err_msg}. Max retries reached.")
                time.sleep(self.INITIAL_BACKOFF_FACTOR * (2 ** attempt))

        self.last_call = time.time() # Update last_call even if all retries failed
        return "", 0, 0 # Fallback if all retries fail
