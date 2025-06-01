# llm_processor.py
import re
import time
import requests
import tiktoken # Ensure tiktoken is available or handled

class LLMProcessor:
    def __init__(self, api_config, custom_prompt_text=None):
        self.api_url = api_config['url']
        self.api_key = api_config['key']
        self.model = api_config['model']
        self.custom_prompt_for_processor = custom_prompt_text

        # 安全地初始化tiktoken编码器
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
            except KeyError:
                encoding_name = encoding_map.get(model_name.split('-')[0], 'cl100k_base')
                self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            print(f"LLMProcessor Tiktoken encoder initialization warning: {e}, using default cl100k_base encoder")
            self.encoding = tiktoken.get_encoding('cl100k_base')

        self.last_call = 0

    def calculate_tokens(self, text):
        try:
            return len(self.encoding.encode(text))
        except Exception:
            chinese_chars = len(re.findall(r'[一-鿿]', text))
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4)

    def summarize(self, text, context="", max_retries=3):
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

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": min(4000, max(100, int(len(text) * 0.015))) # Ensure text has length
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
                response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)

                result = response.json()

                if 'error' in result:
                    error_msg = result['error'].get('message', '未知错误')
                    error_code = result['error'].get('code', 'unknown')
                    if 'model' in error_msg.lower() or error_code == 'model_not_found':
                        raise ValueError(f"模型 {self.model} 不存在")
                    elif 'api_key' in error_msg.lower() or error_code == 'invalid_api_key':
                        raise PermissionError("API密钥无效")
                    else:
                        raise RuntimeError(f"API错误: {error_msg}")

                if 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                    summary_text = result['choices'][0]['message']['content']
                else:
                    # Handle cases for different API structures if necessary, e.g. Anthropic
                    if "claude" in self.model.lower() and "content" in result and isinstance(result["content"], list) and len(result["content"]) > 0 and "text" in result["content"][0]:
                         summary_text = result["content"][0]["text"]
                    else:
                         raise ValueError("API返回格式异常或内容为空")

                input_tokens = result.get('usage', {}).get('prompt_tokens', 0)
                output_tokens = result.get('usage', {}).get('completion_tokens', 0)

                if input_tokens == 0: input_tokens = self.calculate_tokens(prompt)
                if output_tokens == 0: output_tokens = self.calculate_tokens(summary_text)

                self.last_call = time.time()
                return summary_text, input_tokens, output_tokens

            except requests.exceptions.RequestException as e: # Catch network errors
                if attempt == max_retries - 1:
                    raise RuntimeError(f"API网络请求失败: {str(e)}")
                time.sleep(2 ** attempt)
            except Exception as e: # Catch other errors like JSONDecodeError, PermissionError, ValueError
                if attempt == max_retries - 1:
                    raise RuntimeError(f"API调用最终失败: {str(e)}")
                time.sleep(2 ** attempt)
        return "", 0, 0 # Should not be reached if max_retries > 0
