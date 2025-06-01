# constants.py

DEFAULT_MODEL_CONFIGS = {
    # OpenAI系列
    "gpt-4": {
        "url": "https://api.openai.com/v1/chat/completions",
        "display_name": "GPT-4"
    },
    "gpt-3.5-turbo": {
        "url": "https://api.openai.com/v1/chat/completions",
        "display_name": "GPT-3.5 Turbo"
    },
    # DeepSeek
    "deepseek-chat": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "display_name": "DeepSeek Chat"
    },
    "deepseek-coder": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "display_name": "DeepSeek Coder"
    },
    # 阿里通义千问
    "qwen-turbo": {
        "url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        "display_name": "通义千问 Turbo"
    },
    "qwen-plus": {
        "url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        "display_name": "通义千问 Plus"
    },
    "qwen-max": {
        "url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        "display_name": "通义千问 Max"
    },
    # 智谱ChatGLM
    "chatglm-pro": {
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "display_name": "ChatGLM Pro"
    },
    "chatglm-std": {
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "display_name": "ChatGLM Standard"
    },
    "chatglm-lite": {
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "display_name": "ChatGLM Lite"
    },
    # 百度文心一言
    "ernie-bot-turbo": {
        "url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant",
        "display_name": "文心一言 Turbo"
    },
    "ernie-bot": {
        "url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions",
        "display_name": "文心一言"
    },
    # 讯飞星火
    "spark-max": {
        "url": "https://spark-api.xf-yun.com/v1/chat/completions",
        "display_name": "讯飞星火 Max"
    },
    # Claude (通过第三方API)
    "claude-3-haiku": {
        "url": "https://api.anthropic.com/v1/messages",
        "display_name": "Claude 3 Haiku"
    },
    "claude-3-sonnet": {
        "url": "https://api.anthropic.com/v1/messages",
        "display_name": "Claude 3 Sonnet"
    },

    # Example for a Mistral model via an OpenAI-compatible endpoint
    "mistral-7b-instruct-openai-compat": {
        "url": "YOUR_OPENAI_COMPATIBLE_ENDPOINT_HERE/v1/chat/completions", # User needs to fill this
        "display_name": "Mistral-7B Instruct (OpenAI Compat)"
    },

    # Example for local Ollama setup (OpenAI-compatible endpoint)
    "ollama-local-model": {
        "url": "http://localhost:11434/v1/chat/completions", # Default Ollama OpenAI-compatible endpoint
        "display_name": "Ollama Local Model (e.g., Llama3)"
        # User would specify the actual model name string in the UI when selecting this
        # or the 'model' field in API call needs to be set to what Ollama expects e.g. "llama3"
        # For simplicity in DEFAULT_MODEL_CONFIGS, we assume the 'model' parameter in the API call
        # will be taken from the model_combo.currentText() or currentData() which is the key here.
        # So, the user would select "ollama-local-model" and the request would send "model": "ollama-local-model".
        # This might need adjustment in LLMProcessor or how model names are handled if the API expects
        # a different 'model' field value than the key used in our combobox.
        # For now, keeping it simple. User can use "Custom Model" for more complex cases.
    },

    # 自定义模型 (Placeholder)
    "custom": {
        "url": "",
        "display_name": "自定义模型"
    }
}

DEFAULT_CUSTOM_PROMPT = "提炼以下文本的核心要点，仅输出提炼后的内容，不要包含任何额外解释或与原文无关的文字。保留关键情节和人物关系，压缩至原文1%字数："
