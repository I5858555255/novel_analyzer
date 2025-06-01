# llm_processor.py
import re
import time
import requests
import tiktoken # Ensure tiktoken is available or handled
from performance_logger import PerformanceLogger

class LLMProcessor:
    def __init__(self, api_config, custom_prompt_text=None):
        if not isinstance(api_config, dict):
            raise ValueError("api_config must be a dictionary.")

        self.api_url = api_config.get('url')
        self.api_key = api_config.get('key')
        self.model = api_config.get('model')

        if not self.api_url or not self.model: # API key can sometimes be optional for local models
            # Key presence is usually validated by UI (validate_config) before this point,
            # but model and URL are essential for the processor.
            raise ValueError("API URL and model name must be provided in api_config.")

        self.custom_prompt_for_processor = custom_prompt_text
        self.session = requests.Session()
        self.MAX_RETRIES = 3  # Max retries for API calls
        self.INITIAL_BACKOFF_FACTOR = 1  # Seconds for initial backoff
        self.perf_logger = PerformanceLogger() # Get the singleton instance

        # 安全地初始化tiktoken编码器
        print(f"DEBUG: LLMProcessor.__init__ - model: {self.model} - Attempting tiktoken init")
        try:
            model_name = self.model.split('/')[-1].lower()
            encoding_map = {
                'gpt-4': 'cl100k_base',
                'gpt-3.5-turbo': 'cl100k_base',
                'deepseek-chat': 'cl100k_base',
                'qwen-turbo': 'cl100k_base',
                'chatglm-pro': 'cl100k_base',
                'claude': 'cl100k_base',
                'gemini': 'cl100k_base'
            }
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
                print(f"DEBUG: LLMProcessor.__init__ - tiktoken.encoding_for_model succeeded for {model_name}")
            except KeyError:
                encoding_name = encoding_map.get(model_name.split('-')[0], 'cl100k_base')
                self.encoding = tiktoken.get_encoding(encoding_name)
                print(f"DEBUG: LLMProcessor.__init__ - tiktoken.get_encoding succeeded for {encoding_name}")
        except Exception as e:
            print(f"DEBUG: LLMProcessor.__init__ - tiktoken EXCEPTION: {e}, using default cl100k_base encoder")
            self.encoding = tiktoken.get_encoding('cl100k_base')
        print(f"DEBUG: LLMProcessor.__init__ - tiktoken setup complete")

        self.last_call = 0

    def calculate_tokens(self, text):
        try:
            return len(self.encoding.encode(text))
        except Exception:
            chinese_chars = len(re.findall(r'[一-鿿]', text)) # More comprehensive CJK range
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4) # Rough estimation

    def summarize(self, text, context="", max_retries=None): # max_retries can be overridden
        if text is None or not text.strip():
            # self.last_call = time.time() # Update last_call even for empty text to maintain rate limiting logic
            return "", 0, 0

        current_time = time.time()
        if current_time - self.last_call < 1.0: # Basic rate limiting per instance
            time.sleep(1.0 - (current_time - self.last_call))

        default_prompt_template = "提炼以下文本的核心要点，仅输出提炼后的内容，不要包含任何额外解释或与原文无关的文字。保留关键情节和人物关系，压缩至原文1%字数："
        effective_prompt_template = self.custom_prompt_for_processor if self.custom_prompt_for_processor else default_prompt_template
        prompt = f"{effective_prompt_template}\n{text}"

        if context:
            prompt = f"上下文：{context}\n\n{prompt}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Ensure text has some length for max_tokens calculation, otherwise default to a reasonable value
        text_len_for_tokens = len(text) if text else 100 # Use 100 if text is empty for some reason
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": min(4000, max(100, int(text_len_for_tokens * 0.015)))
        }

        effective_max_retries = max_retries if max_retries is not None else self.MAX_RETRIES
        call_start_time = None # Ensure it's defined in this scope

        for attempt in range(effective_max_retries):
            call_start_time = time.time() # Capture start time for each attempt
            response = None # Ensure response is defined for logging in case of pre-request errors
            try:
                response = self.session.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=30
                )

                if response.status_code == 429:
                    retry_after_seconds_str = response.headers.get("Retry-After")
                    wait_time = self.INITIAL_BACKOFF_FACTOR * (2 ** attempt)
                    if retry_after_seconds_str:
                        try:
                            wait_time = int(retry_after_seconds_str)
                            print(f"API rate limit (429). Retrying after {wait_time}s (Retry-After header).")
                        except ValueError:
                            print(f"API rate limit (429). Invalid Retry-After. Using default backoff: {wait_time}s.")
                    else:
                        print(f"API rate limit (429). No Retry-After. Using default backoff: {wait_time}s.")

                    if attempt == effective_max_retries - 1:
                        self.last_call = time.time()
                        # Log before raising
                        call_latency_ms = (time.time() - call_start_time) * 1000
                        self.perf_logger.log_api_call(
                            model_id=self.model, api_url=self.api_url, success=False,
                            http_status_code=429, latency_ms=call_latency_ms,
                            error_message="HTTP 429 Too Many Requests (Max Retries Reached)",
                            context_provided=bool(context)
                        )
                        raise RuntimeError(f"API rate limit (429) reached after max retries.")

                    # Log non-fatal 429 before sleeping
                    call_latency_ms_for_429 = (time.time() - call_start_time) * 1000
                    self.perf_logger.log_api_call(
                        model_id=self.model, api_url=self.api_url, success=False,
                        http_status_code=429, latency_ms=call_latency_ms_for_429,
                        error_message="HTTP 429 Too Many Requests (Retrying)",
                        context_provided=bool(context)
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                call_latency_ms = (time.time() - call_start_time) * 1000 # Successful call latency
                result = response.json()
                summary_text = None

                if 'error' in result:
                    error_msg = result['error'].get('message', '未知错误')
                    error_code = result['error'].get('code', 'unknown')
                    # Log this API-level error
                    self.perf_logger.log_api_call(
                        model_id=self.model, api_url=self.api_url, success=False,
                        http_status_code=response.status_code, latency_ms=call_latency_ms,
                        error_message=f"API Error: {error_code} - {error_msg}",
                        context_provided=bool(context)
                    )
                    if 'model' in error_msg.lower() or error_code == 'model_not_found':
                        raise ValueError(f"模型 {self.model} 不存在 (API Error)")
                    elif 'api_key' in error_msg.lower() or error_code == 'invalid_api_key':
                        raise PermissionError("API密钥无效 (API Error)")
                    else:
                        raise RuntimeError(f"API错误: {error_msg}")

                if 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                    summary_text = result['choices'][0]['message']['content']
                elif "claude" in self.model.lower() and "content" in result and isinstance(result["content"], list) and len(result["content"]) > 0 and "text" in result["content"][0]:
                     summary_text = result["content"][0]["text"]
                else:
                     # Log this specific error before raising
                    self.perf_logger.log_api_call(
                        model_id=self.model, api_url=self.api_url, success=False,
                        http_status_code=response.status_code, latency_ms=call_latency_ms,
                        error_message="API返回格式异常或内容为空",
                        context_provided=bool(context)
                    )
                    raise ValueError("API返回格式异常或内容为空")

                if summary_text is None:
                    err_msg_summary = "未能从API响应中提取摘要文本。"
                    self.perf_logger.log_api_call(
                        model_id=self.model, api_url=self.api_url, success=False,
                        http_status_code=response.status_code, latency_ms=call_latency_ms,
                        error_message=err_msg_summary,
                        context_provided=bool(context)
                    )
                    raise ValueError(err_msg_summary)

                input_tokens = result.get('usage', {}).get('prompt_tokens', 0)
                output_tokens = result.get('usage', {}).get('completion_tokens', 0)

                if input_tokens == 0: input_tokens = self.calculate_tokens(prompt)
                if output_tokens == 0: output_tokens = self.calculate_tokens(summary_text)

                self.perf_logger.log_api_call(
                    model_id=self.model, api_url=self.api_url, success=True,
                    http_status_code=response.status_code, latency_ms=call_latency_ms,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    context_provided=bool(context)
                )
                self.last_call = time.time()
                return summary_text, input_tokens, output_tokens

            except requests.exceptions.HTTPError as e:
                call_latency_ms = (time.time() - call_start_time) * 1000
                print(f"Attempt {attempt + 1}/{effective_max_retries}: HTTP Error {e.response.status_code} for {self.api_url}. Response: {e.response.text if e.response else 'N/A'}")
                self.perf_logger.log_api_call(
                    model_id=self.model, api_url=self.api_url, success=False,
                    http_status_code=e.response.status_code if e.response else None,
                    latency_ms=call_latency_ms,
                    error_message=str(e),
                    context_provided=bool(context)
                )
                if attempt == effective_max_retries - 1:
                    self.last_call = time.time()
                    raise RuntimeError(f"API HTTP Error {e.response.status_code if e.response else 'Unknown'}: {str(e)}. Max retries reached. Details: {e.response.text if e.response else 'N/A'}")
                time.sleep(self.INITIAL_BACKOFF_FACTOR * (2 ** attempt))
            except requests.exceptions.RequestException as e:
                call_latency_ms = (time.time() - call_start_time) * 1000
                print(f"Attempt {attempt + 1}/{effective_max_retries}: Network Request Failed for {self.api_url}. Error: {str(e)}")
                self.perf_logger.log_api_call(
                    model_id=self.model, api_url=self.api_url, success=False,
                    http_status_code=None,
                    latency_ms=call_latency_ms,
                    error_message=str(e),
                    context_provided=bool(context)
                )
                if attempt == effective_max_retries - 1:
                    self.last_call = time.time()
                    raise RuntimeError(f"API Network Request Failed: {str(e)}. Max retries reached.")
                time.sleep(self.INITIAL_BACKOFF_FACTOR * (2 ** attempt))
            except Exception as e:
                call_latency_ms = (time.time() - call_start_time) * 1000 if call_start_time else -1 # -1 if error was before call_start_time
                status_code_to_log = response.status_code if response and hasattr(response, 'status_code') else None

                print(f"Attempt {attempt + 1}/{effective_max_retries}: API Call/Processing Error for {self.api_url}. Error: {type(e).__name__} - {str(e)}")
                self.perf_logger.log_api_call(
                    model_id=self.model, api_url=self.api_url, success=False,
                    http_status_code=status_code_to_log,
                    latency_ms=call_latency_ms,
                    error_message=str(e),
                    context_provided=bool(context)
                )
                if attempt == effective_max_retries - 1:
                    self.last_call = time.time()
                    raise RuntimeError(f"API Call/Processing Error: {str(e)}. Max retries reached.")
                if isinstance(e, (PermissionError, ValueError)):
                    self.last_call = time.time()
                    raise
                time.sleep(self.INITIAL_BACKOFF_FACTOR * (2 ** attempt))

        self.last_call = time.time()
        return "", 0, 0
