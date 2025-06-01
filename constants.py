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
    # 自定义模型 (Placeholder)
    "custom": {
        "url": "",
        "display_name": "自定义模型"
    }
}

DEFAULT_CUSTOM_PROMPT = "提炼以下文本的核心要点，仅输出提炼后的内容，不要包含任何额外解释或与原文无关的文字。保留关键情节和人物关系，压缩至原文1%字数："
