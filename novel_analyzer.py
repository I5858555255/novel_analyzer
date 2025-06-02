# novel_analyzer.py (Consolidated)

import os
import re
import json
import threading
import queue
import time
import copy
import tiktoken
import csv
import requests
import sys # For sys.argv and sys.exit

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem, QTextEdit,
    QFileDialog, QPushButton, QComboBox, QProgressBar, QLabel, QSplitter,
    QVBoxLayout, QHBoxLayout, QWidget, QAction, QMessageBox, QLineEdit,
    QHeaderView, QSpinBox, QDialog, QFormLayout, QDialogButtonBox
)
from PyQt5.QtCore import (
    pyqtSignal, Qt, QThread, QTimer, QThreadPool, QObject, QRunnable
)

# Content from constants.py
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
    "mistral-7b-instruct-openai-compat": {
        "url": "YOUR_OPENAI_COMPATIBLE_ENDPOINT_HERE/v1/chat/completions", 
        "display_name": "Mistral-7B Instruct (OpenAI Compat)"
    },
    "ollama-local-model": {
        "url": "http://localhost:11434/v1/chat/completions", 
        "display_name": "Ollama Local Model (e.g., Llama3)"
    },
    "custom": {
        "url": "",
        "display_name": "自定义模型"
    }
}
DEFAULT_CUSTOM_PROMPT = "提炼以下文本的核心要点，仅输出提炼后的内容，不要包含任何额外解释或与原文无关的文字。保留关键情节和人物关系，压缩至原文1%字数："

# Content from performance_logger.py
class PerformanceLogger:
    _instance = None
    _lock = threading.Lock() 
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self, filename="api_calls.csv", log_dir="logs"):
        if hasattr(self, '_initialized') and self._initialized: return
        self.log_dir = log_dir
        self.filename = os.path.join(self.log_dir, filename)
        self.fieldnames = ["timestamp_utc", "model_id", "api_url", "success", "http_status_code", "latency_ms", "input_tokens", "output_tokens", "error_message", "context_provided"]
        self._setup_logging()
        self._initialized = True
    def _setup_logging(self):
        try:
            if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
            file_exists = os.path.isfile(self.filename)
            is_empty = file_exists and os.path.getsize(self.filename) == 0
            if not file_exists or is_empty:
                with open(self.filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writeheader()
        except IOError as e: print(f"Error setting up performance logger: {e}")
    def log_api_call(self, model_id, api_url, success, http_status_code, latency_ms, input_tokens=0, output_tokens=0, error_message="", context_provided=False):
        try:
            with self._lock: 
                with open(self.filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    log_entry = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model_id": model_id, "api_url": api_url, "success": success, "http_status_code": http_status_code if http_status_code is not None else "", "latency_ms": f"{latency_ms:.2f}" if latency_ms is not None else "", "input_tokens": input_tokens if input_tokens else 0, "output_tokens": output_tokens if output_tokens else 0, "error_message": error_message if error_message else "", "context_provided": context_provided}
                    writer.writerow(log_entry)
        except IOError as e: print(f"Error writing to performance log: {e}")

# Content from custom_widgets.py
class ChapterTreeItem(QTreeWidgetItem):
    def __init__(self, title, content, word_count, parent=None):
        super().__init__(parent, [title, f"{word_count}字"])
        self.original_title = title 
        self.content = content
        self.summary = ""
        self.word_count = word_count
        self.is_summarized = False
        self.summary_timestamp = 0
    def update_display_text(self):
        marker = "* "
        expected_text_if_summarized = marker + self.original_title
        expected_text_if_not_summarized = self.original_title
        if self.is_summarized:
            if self.text(0) != expected_text_if_summarized:
                self.setText(0, expected_text_if_summarized)
        else:
            if self.text(0) != expected_text_if_not_summarized:
                self.setText(0, expected_text_if_not_summarized)

# Content from llm_processor.py
class LLMProcessor:
    def __init__(self, api_config, custom_prompt_text=None, encoding_object=None):
        if not isinstance(api_config, dict):
            raise ValueError("LLMProcessor: api_config must be a dictionary.")
        if not api_config.get('url') or not isinstance(api_config.get('url'), str) or not api_config.get('url').strip():
            raise ValueError("LLMProcessor: api_config must contain a non-empty 'url' string.")
        if not api_config.get('model') or not isinstance(api_config.get('model'), str) or not api_config.get('model').strip():
            raise ValueError("LLMProcessor: api_config must contain a non-empty 'model' string.")
        self.api_url = api_config['url']
        self.api_key = api_config.get('key', "") 
        self.model = api_config['model']
        self.custom_prompt_for_processor = custom_prompt_text
        self.encoding = encoding_object
        if self.encoding is None:
            raise ValueError("LLMProcessor requires a valid Tiktoken encoding object.")
        self.session = requests.Session() 
        self.MAX_RETRIES = 3
        self.INITIAL_BACKOFF_FACTOR = 1
        self.last_call = 0
        try:
            self.perf_logger = PerformanceLogger()
        except NameError: 
            print("Warning: PerformanceLogger class not found during LLMProcessor init. Performance logging will be disabled.")
            self.perf_logger = None
        except Exception as e_perf_init: 
            print(f"Error initializing PerformanceLogger in LLMProcessor: {e_perf_init}. Performance logging disabled.")
            self.perf_logger = None
            
    def calculate_tokens(self, text):
        if not self.encoding: 
            print("Warning: Tiktoken encoding not available in calculate_tokens. Using rough estimate.")
            chinese_chars = len(re.findall(r'[一-鿿]', text))
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4)
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            error_message = f"Error during tiktoken.encode: {str(e)}. "
            error_message += f"Problematic text (first 100 chars): '{text[:100]}'. "
            error_message += "Falling back to rough token estimation."
            print(error_message)
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text)) 
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4)

    def summarize(self, text, context="", max_retries=None):
        if not text: 
            if self.perf_logger:
                self.perf_logger.log_api_call(
                    model_id=self.model, api_url=self.api_url, success=False,
                    http_status_code=None, latency_ms=0,
                    error_message="Input text is empty.", context_provided=bool(context)
                )
            return "", 0, 0
        call_start_time_overall = time.time()
        if time.time() - self.last_call < 1.0: 
            time.sleep(1.0 - (time.time() - self.last_call))
        effective_max_retries = max_retries if max_retries is not None else self.MAX_RETRIES
        default_prompt_template = "提炼以下文本的核心要点，仅输出提炼后的内容，不要包含任何额外解释或与原文无关的文字。保留关键情节和人物关系，压缩至原文1%字数："
        effective_prompt_template = self.custom_prompt_for_processor if self.custom_prompt_for_processor else default_prompt_template
        prompt = f"{effective_prompt_template}\n{text}"
        if context:
            prompt = f"上下文：{context}\n\n{prompt}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        text_len_for_max_tokens = len(text) if text else 1
        data = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2, "top_p": 0.8,
                "max_tokens": min(4000, max(100, int(text_len_for_max_tokens * 0.015)))}
        summary_text_final = ""
        input_tokens_final = 0
        output_tokens_final = 0
        for attempt in range(effective_max_retries):
            call_attempt_start_time = time.time()
            response_obj = None 
            http_status_to_log = None
            try:
                response_obj = self.session.post(self.api_url, headers=headers, json=data, timeout=30)
                http_status_to_log = response_obj.status_code
                if http_status_to_log == 429:
                    retry_after_seconds_str = response_obj.headers.get("Retry-After")
                    wait_time = self.INITIAL_BACKOFF_FACTOR * (2 ** attempt)
                    if retry_after_seconds_str:
                        try: wait_time = int(retry_after_seconds_str)
                        except ValueError: pass
                    if attempt == effective_max_retries - 1: raise RuntimeError(f"API rate limit (429). Max retries reached.")
                    if self.perf_logger: self.perf_logger.log_api_call(model_id=self.model, api_url=self.api_url, success=False,http_status_code=429, latency_ms=(time.time() - call_attempt_start_time) * 1000, error_message="HTTP 429 Too Many Requests (will retry)", context_provided=bool(context))
                    time.sleep(wait_time)
                    continue
                response_obj.raise_for_status() 
                result = response_obj.json()
                if 'error' in result:
                    error_msg_detail = result['error'].get('message', 'Unknown API error in response')
                    if 'model' in error_msg_detail.lower() or result['error'].get('code') == 'model_not_found': raise ValueError(f"Model {self.model} not found (API Error: {error_msg_detail})")
                    if 'api_key' in error_msg_detail.lower() or result['error'].get('code') == 'invalid_api_key': raise PermissionError(f"Invalid API key (API Error: {error_msg_detail})")
                    raise RuntimeError(f"API returned error: {error_msg_detail}")
                if 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                    summary_text_final = result['choices'][0]['message']['content']
                elif "claude" in self.model.lower() and "content" in result and isinstance(result["content"], list) and len(result["content"]) > 0 and "text" in result["content"][0]:
                    summary_text_final = result["content"][0]["text"]
                else: raise ValueError("API response format error or empty content.")
                usage = result.get('usage', {})
                input_tokens_final = usage.get('prompt_tokens', 0)
                output_tokens_final = usage.get('completion_tokens', 0)
                if input_tokens_final == 0: input_tokens_final = self.calculate_tokens(prompt)
                if output_tokens_final == 0: output_tokens_final = self.calculate_tokens(summary_text_final)
                if self.perf_logger: self.perf_logger.log_api_call(model_id=self.model, api_url=self.api_url, success=True, http_status_code=http_status_to_log, latency_ms=(time.time() - call_attempt_start_time) * 1000, input_tokens=input_tokens_final, output_tokens=output_tokens_final, context_provided=bool(context))
                self.last_call = time.time()
                return summary_text_final, input_tokens_final, output_tokens_final
            except requests.exceptions.HTTPError as e_http: err_msg = f"HTTP Error {e_http.response.status_code}: {str(e_http)}"
            except requests.exceptions.RequestException as e_req: err_msg = f"Network Request Failed: {str(e_req)}"
            except Exception as e_other: err_msg = f"API Call/Processing Error: {str(e_other)}" 
            if self.perf_logger: self.perf_logger.log_api_call(model_id=self.model, api_url=self.api_url, success=False,http_status_code=http_status_to_log, latency_ms=(time.time() - call_attempt_start_time) * 1000,error_message=err_msg, context_provided=bool(context))
            if attempt == effective_max_retries - 1: self.last_call = time.time(); raise RuntimeError(f"{err_msg}. Max retries reached.")
            time.sleep(self.INITIAL_BACKOFF_FACTOR * (2 ** attempt))
        self.last_call = time.time() 
        return "", 0, 0

# Content from threading_utils.py
class SummarizationSignals(QObject):
    update_signal = pyqtSignal(object, str)
    progress_signal = pyqtSignal(int, int, int)
    error_signal = pyqtSignal(object, str)
    finished_signal = pyqtSignal(object)

class SummarizationTask(QRunnable):
    def __init__(self, chapter_item_identifier, chapter_context, api_config, custom_prompt_text, main_window_ref, encoding_object):
        super().__init__()
        self.identifier = chapter_item_identifier
        self.context = chapter_context
        self.api_config = api_config
        self.custom_prompt_for_processor = custom_prompt_text
        self.signals = SummarizationSignals()
        self.main_window = main_window_ref
        self.encoding_object = encoding_object
    def run(self):
        thread_id = threading.get_ident()
        if self.main_window.stop_batch_requested:
            self.signals.error_signal.emit(self.identifier, "处理被用户中止")
            self.signals.finished_signal.emit(self.identifier)
            return
        current_chapter_content = self.main_window.get_content_for_task(self.identifier)
        if current_chapter_content is None:
            self.signals.error_signal.emit(self.identifier, "内容未找到或已被处理")
            self.signals.finished_signal.emit(self.identifier)
            return
        processor = None
        try:
            processor = LLMProcessor(self.api_config, self.custom_prompt_for_processor, self.encoding_object)
        except Exception as e_proc_init:
            self.signals.error_signal.emit(self.identifier, f"LLMProcessor init error: {str(e_proc_init)}")
            if self.main_window: self.main_window.clear_content_for_task(self.identifier)
            self.signals.finished_signal.emit(self.identifier)
            return
        summary_text = None
        in_tokens, out_tokens = 0, 0
        try:
            if self.main_window.stop_batch_requested:
                self.signals.error_signal.emit(self.identifier, "处理被用户中止")
                self.signals.finished_signal.emit(self.identifier) 
                return
            summary_text, in_tokens, out_tokens = processor.summarize(current_chapter_content, self.context)
            if self.main_window.stop_batch_requested:
                self.signals.error_signal.emit(self.identifier, "处理完成但已被用户中止")
                if summary_text is not None: self.signals.update_signal.emit(self.identifier, summary_text)
                self.signals.progress_signal.emit(in_tokens, out_tokens, 1)
                self.signals.finished_signal.emit(self.identifier) 
                return
            if summary_text is not None: self.signals.update_signal.emit(self.identifier, summary_text)
            self.signals.progress_signal.emit(in_tokens, out_tokens, 1)
        except Exception as e_summarize:
            self.signals.error_signal.emit(self.identifier, str(e_summarize))
            self.signals.progress_signal.emit(in_tokens, out_tokens, 1) 
        finally:
            if self.main_window: 
                self.main_window.clear_content_for_task(self.identifier)
            self.signals.finished_signal.emit(self.identifier)

class WorkerThread(QThread):
    update_signal = pyqtSignal(str, object)
    progress_signal = pyqtSignal(int, int, int)
    error_signal = pyqtSignal(str)
    def __init__(self, work_queue, llm_processor_instance):
        super().__init__()
        self.work_queue = work_queue
        self.llm_processor = llm_processor_instance
        self.running = True
    def run(self):
        while self.running and not self.work_queue.empty():
            try:
                task_data = self.work_queue.get_nowait()
                item, context = task_data
                summary, in_tokens, out_tokens = self.llm_processor.summarize(item.content, context)
                self.update_signal.emit("summary", (item, summary))
                self.progress_signal.emit(in_tokens, out_tokens, 1)
                time.sleep(0.5) 
            except queue.Empty: break
            except Exception as e: self.error_signal.emit(f"处理错误: {str(e)}"); break
    def stop(self): self.running = False

# Content from dialogs.py
class ManageModelsDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window 
        self.setWindowTitle("管理自定义模型")
        self.setMinimumSize(600, 400) 
        layout = QVBoxLayout(self)
        info_label = QLabel("以下是您添加的自定义模型。预定义模型无法在此处移除。")
        layout.addWidget(info_label)
        self.models_list_widget = QTreeWidget()
        self.models_list_widget.setHeaderLabels(["模型显示名称", "模型ID", "API 地址", "操作"])
        self.models_list_widget.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.models_list_widget.header().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(self.models_list_widget)
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.accept)
        layout.addWidget(self.close_button)
        self.setLayout(layout)
        self.populate_models_list()
    def populate_models_list(self):
        self.models_list_widget.clear()
        custom_models_found = False
        for model_key, config_data in self.main_window.model_configs.items():
            if model_key not in self.main_window.initial_model_keys:
                custom_models_found = True
                display_name = config_data.get("display_name", model_key)
                url = config_data.get("url", "N/A")
                tree_item = QTreeWidgetItem(self.models_list_widget, [display_name, model_key, url])
                remove_button = QPushButton("移除")
                remove_button.setProperty("model_key_to_remove", model_key)
                remove_button.clicked.connect(self.handle_remove_model)
                self.models_list_widget.setItemWidget(tree_item, 3, remove_button)
        if not custom_models_found:
            item = QTreeWidgetItem(self.models_list_widget, ["没有自定义模型可管理。"])
            self.models_list_widget.setEnabled(False)
    def handle_remove_model(self):
        button_clicked = self.sender()
        if not button_clicked: return
        model_key_to_remove = button_clicked.property("model_key_to_remove")
        if not model_key_to_remove: return
        model_display_name = self.main_window.model_configs.get(model_key_to_remove, {}).get("display_name", model_key_to_remove)
        reply = QMessageBox.question(self, "确认移除",
                                     f"确定要移除自定义模型 '{model_display_name}' ({model_key_to_remove}) 吗？此操作无法撤销。",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if model_key_to_remove in self.main_window.model_configs:
                del self.main_window.model_configs[model_key_to_remove]
            combo = self.main_window.model_combo
            for i in range(combo.count()):
                if combo.itemData(i) == model_key_to_remove:
                    if combo.currentIndex() == i:
                        combo.setCurrentIndex(-1)
                        self.main_window.api_url_input.clear()
                    combo.removeItem(i)
                    break
            if hasattr(self.main_window, 'status_label'):
                 self.main_window.status_label.setText(f"自定义模型 '{model_display_name}' 已移除。")
            self.populate_models_list() 

# Content from main_window.py
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("小说智能分析工具 v2.0")
        self.setGeometry(100, 100, 1200, 800)

        self.book_data = {"title": "", "volumes": []}
        self.llm_processor = None 
        self.work_queue = queue.Queue()
        self.worker_thread = None 
        self.total_tokens = [0, 0]
        self.default_export_path = ""
        self.custom_prompt = DEFAULT_CUSTOM_PROMPT 
        self.model_configs = copy.deepcopy(DEFAULT_MODEL_CONFIGS) 
        self.initial_model_keys = set(self.model_configs.keys())

        self.tiktoken_encoding_cache = {}
        self.tiktoken_cache_lock = threading.Lock()

        self.thread_pool = QThreadPool()
        desired_thread_count = 8
        try:
            cpu_cores = os.cpu_count()
            if cpu_cores: pass 
        except Exception: pass 
        self.thread_pool.setMaxThreadCount(desired_thread_count)
        print(f"Thread pool max threads set to: {self.thread_pool.maxThreadCount()}")

        self.active_batch_tasks = 0
        self.batch_start_time = 0
        self.chapters_to_process_total = 0
        self.chapters_processed_count = 0
        self.total_input_tokens_batch = 0
        self.total_output_tokens_batch = 0
        self.average_time_per_chapter = 0
        self.stop_batch_requested = False

        self.pending_batch_tasks_data = []
        self.current_batch_chunk_offset = 0

        self.batch_content_store = {} 
        self.content_store_lock = threading.Lock() 

        self._is_ui_ready = False
        self.auto_export_base_dir = os.path.join(os.path.expanduser("~"), "Desktop", "NovelAnalyzer_Exports")

        self.init_ui()
        self._is_ui_ready = True

        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(lambda: self.save_config(silent=True))
        self.auto_save_timer.start(30000)
        self.load_config(silent=True)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        control_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        for model_key, config_item in self.model_configs.items():
            self.model_combo.addItem(config_item["display_name"], model_key)
        self.model_combo.setCurrentIndex(-1)
        self.model_combo.setPlaceholderText("选择或输入模型名称")
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        control_layout.addWidget(QLabel("选择模型:"))
        control_layout.addWidget(self.model_combo)
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("输入API密钥")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        control_layout.addWidget(QLabel("API密钥:"))
        control_layout.addWidget(self.api_key_input)
        self.api_url_input = QLineEdit()
        self.api_url_input.setPlaceholderText("API服务地址")
        control_layout.addWidget(QLabel("API地址:"))
        control_layout.addWidget(self.api_url_input)
        self.test_btn = QPushButton("测试连接")
        self.test_btn.clicked.connect(self.test_connection)
        control_layout.addWidget(self.test_btn)
        main_layout.addLayout(control_layout)

        control_layout2 = QHBoxLayout()
        self.load_btn = QPushButton("导入小说")
        self.load_btn.clicked.connect(self.load_novel)
        control_layout2.addWidget(self.load_btn)
        self.summary_mode_btn = QPushButton("显示原文")
        self.summary_mode_btn.setCheckable(True)
        self.summary_mode_btn.clicked.connect(self.toggle_display_mode)
        control_layout2.addWidget(self.summary_mode_btn)
        self.save_config_btn = QPushButton("保存配置")
        self.save_config_btn.clicked.connect(self.save_config)
        control_layout2.addWidget(self.save_config_btn)
        self.load_config_btn = QPushButton("加载配置")
        self.load_config_btn.clicked.connect(self.load_config)
        control_layout2.addWidget(self.load_config_btn)
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("输入自定义提示词")
        self.prompt_input.setText(self.custom_prompt)
        self.prompt_input.textChanged.connect(self.update_prompt)
        control_layout2.addWidget(QLabel("提示词:"))
        control_layout2.addWidget(self.prompt_input)
        main_layout.addLayout(control_layout2)

        splitter = QSplitter(Qt.Horizontal)
        self.chapter_tree = QTreeWidget()
        self.chapter_tree.setHeaderLabels(["章节/卷", "字数"])
        self.chapter_tree.itemClicked.connect(self.show_content)

        header = self.chapter_tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setMinimumSectionSize(300)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)

        splitter.addWidget(self.chapter_tree)
        right_panel_layout = QVBoxLayout()
        self.content_display = QTextEdit()
        self.content_display.setReadOnly(True)
        right_panel_layout.addWidget(self.content_display)

        btn_layout = QHBoxLayout()
        self.summarize_btn = QPushButton("提炼当前")
        self.summarize_btn.clicked.connect(self.summarize_selected)
        btn_layout.addWidget(self.summarize_btn)

        self.batch_size_label = QLabel("每批章节数:")
        btn_layout.addWidget(self.batch_size_label)

        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setMinimum(1)
        self.batch_size_spinbox.setMaximum(100)
        self.batch_size_spinbox.setValue(10)
        self.batch_size_spinbox.setSingleStep(1)
        self.batch_size_spinbox.setToolTip("设置“一键提炼”时每批处理的章节数量 (1-100)")
        btn_layout.addWidget(self.batch_size_spinbox)

        self.summarize_all_btn = QPushButton("一键提炼")
        self.summarize_all_btn.clicked.connect(self.summarize_all)
        btn_layout.addWidget(self.summarize_all_btn)

        self.stop_btn = QPushButton("停止处理")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        right_panel_layout.addLayout(btn_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_panel_layout)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])
        main_layout.addWidget(splitter, 4)

        status_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        status_layout.addWidget(self.progress_bar)
        self.token_label = QLabel("Token消耗: 输入 0 | 输出 0")
        status_layout.addWidget(self.token_label)
        self.status_label = QLabel("就绪")
        status_layout.addWidget(self.status_label)
        self.eta_label = QLabel("")
        status_layout.addWidget(self.eta_label)
        self.metrics_label = QLabel("")
        status_layout.addWidget(self.metrics_label)
        main_layout.addLayout(status_layout)

        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件')
        export_action = QAction('设置导出路径', self)
        export_action.triggered.connect(self.set_export_path)
        file_menu.addAction(export_action)
        model_menu = menubar.addMenu('模型管理')
        self.add_model_action = QAction('添加自定义模型', self)
        self.add_model_action.triggered.connect(self.add_custom_model)
        model_menu.addAction(self.add_model_action)
        self.manage_models_action = QAction("管理自定义模型", self)
        self.manage_models_action.triggered.connect(self.open_manage_models_dialog)
        model_menu.addAction(self.manage_models_action)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def open_manage_models_dialog(self):
        dialog = ManageModelsDialog(self, self) 
        dialog.exec_()

    def on_model_changed(self):
        try:
            current_text = self.model_combo.currentText()
            current_data = self.model_combo.currentData()
            if current_data and current_data in self.model_configs:
                config_data = self.model_configs[current_data]
                self.api_url_input.setText(config_data["url"])
            elif current_text and current_text not in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
                pass
        except Exception as e:
            print(f"模型切换警告: {e}")

    def load_novel(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择小说文件", "", "文本文件 (*.txt);;所有文件 (*)")
        if not file_path: return
        self.status_label.setText("解析文件中...")
        QApplication.processEvents()
        try:
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
            content = None
            detected_encoding = None
            for encoding_attempt in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding_attempt) as f:
                        content = f.read()
                    detected_encoding = encoding_attempt
                    break
                except UnicodeDecodeError: continue
            if content is None: raise ValueError("无法解码文件")
            chapters = self.parse_chapters(content)
            self.book_data["title"] = os.path.splitext(os.path.basename(file_path))[0]
            self.book_data["file_path"] = file_path
            self.book_data["encoding"] = detected_encoding
            self.build_chapter_tree(chapters)
            self.status_label.setText(f"已加载: {self.book_data['title']} (编码: {detected_encoding})")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"文件加载失败: {str(e)}")

    def parse_chapters(self, content):
        chapters = []
        patterns = [
            r'第([一二三四五六七八九十百千万零\d]+)[卷部][\s　]*(.+?)(?=\n|$)',
            r'([卷部])([一二三四五六七八九十百千万零\d]+)[\s　]*(.+?)(?=\n|$)',
            r'第([一二三四五六七八九十百千万零\d]+)[章节回][\s　]*(.+?)(?=\n|$)',
            r'([章节回])([一二三四五六七八九十百千万零\d]+)[\s　]*(.+?)(?=\n|$)',
            r'^\s*(\d+)\.(.+?)(?=\n|$)',
        ]
        lines = content.split('\n')
        current_volume = None
        current_chapter = None
        content_buffer = []
        for line in lines:
            line = line.strip()
            if not line:
                content_buffer.append('')
                continue
            volume_match = None
            for pattern in patterns[:2]:
                volume_match = re.match(pattern, line)
                if volume_match: break
            if volume_match:
                if current_chapter:
                    current_chapter['content'] = '\n'.join(content_buffer).strip()
                    current_chapter['word_count'] = len(current_chapter['content'])
                    content_buffer = []
                if current_volume: chapters.append(current_volume)
                current_volume = {'title': line, 'chapters': [], 'content': '', 'word_count': 0}
                current_chapter = None
                continue
            chapter_match = None
            for pattern in patterns[2:]:
                chapter_match = re.match(pattern, line)
                if chapter_match: break
            if chapter_match:
                if current_chapter:
                    current_chapter['content'] = '\n'.join(content_buffer).strip()
                    current_chapter['word_count'] = len(current_chapter['content'])
                    if current_volume: current_volume['chapters'].append(current_chapter)
                current_chapter = {'title': line, 'content': '', 'word_count': 0}
                content_buffer = []
                if not current_volume:
                    current_volume = {'title': '正文', 'chapters': [], 'content': '', 'word_count': 0}
                continue
            content_buffer.append(line)
        if current_chapter:
            current_chapter['content'] = '\n'.join(content_buffer).strip()
            current_chapter['word_count'] = len(current_chapter['content'])
            if current_volume: current_volume['chapters'].append(current_chapter)
        if current_volume: chapters.append(current_volume)
        if not chapters:
            chapters = [{'title': '全文', 'chapters': [{'title': '内容', 'content': content, 'word_count': len(content)}], 'content': '', 'word_count': len(content)}]
        return chapters

    def build_chapter_tree(self, chapters):
        self.chapter_tree.setUpdatesEnabled(False)
        try:
            self.chapter_tree.clear()
            root_title = self.book_data.get("title", "未命名书籍")
            root = QTreeWidgetItem(self.chapter_tree, [root_title, ""])
            total_chapters = 0
            total_words = 0
            for volume_data in chapters:
                vol_words = sum(c['word_count'] for c in volume_data['chapters'])
                vol_item = QTreeWidgetItem(root, [volume_data['title'], f"{len(volume_data['chapters'])}章, {vol_words}字"])
                for chapter_data in volume_data['chapters']:
                    chapter_item = ChapterTreeItem(
                        chapter_data['title'],
                        chapter_data['content'],
                        chapter_data['word_count'],
                        vol_item
                    )
                    chapter_item.update_display_text()
                    total_chapters += 1
                total_words += vol_words
            root.setText(1, f"{len(chapters)}卷, {total_chapters}章, {total_words}字")
            root.setExpanded(True)
            for i in range(root.childCount()):
                root.child(i).setExpanded(True)
        finally:
            self.chapter_tree.setUpdatesEnabled(True)

    def show_content(self, item):
        if isinstance(item, ChapterTreeItem): 
            if self.summary_mode_btn.isChecked() or not item.is_summarized:
                self.content_display.setText(item.content)
            else:
                self.content_display.setText(item.summary)

    def toggle_display_mode(self):
        if self.summary_mode_btn.isChecked():
            self.summary_mode_btn.setText("显示要点")
        else:
            self.summary_mode_btn.setText("显示原文")
        current_item = self.chapter_tree.currentItem()
        if current_item: self.show_content(current_item)

    def get_current_model_name(self):
        current_data = self.model_combo.currentData()
        return current_data if current_data else self.model_combo.currentText().strip()

    def summarize_selected(self):
        current = self.chapter_tree.currentItem()
        if not current or not isinstance(current, ChapterTreeItem): 
            QMessageBox.warning(self, "提示", "请先选择要提炼的章节")
            return
        if not self.validate_config(): return
        context = ""
        parent = current.parent()
        if parent and isinstance(parent, ChapterTreeItem) and parent.is_summarized and parent.summary: 
            context = parent.summary
        try:
            api_config = {"url": self.api_url_input.text().strip(), "key": self.api_key_input.text().strip(), "model": self.get_current_model_name()}
            encoding_object = self.get_tiktoken_encoding(api_config['model'])
            if encoding_object is None:
                QMessageBox.critical(self, "编码器错误", f"无法为模型 '{api_config.get('model','未知')}' 初始化Token编码器。提炼中止。")
                self.status_label.setText(f"错误: 模型 '{api_config.get('model','未知')}' Token编码器初始化失败。")
                return
            current_llm_processor_instance = LLMProcessor(api_config, self.custom_prompt, encoding_object) 
            if self.worker_thread and self.worker_thread.isRunning():
                 QMessageBox.warning(self, "提示", "已有单章提炼任务在进行中。请等待其完成。")
                 return
            self.work_queue.put((current, context))
            self.worker_thread = WorkerThread(self.work_queue, current_llm_processor_instance) 
            self.worker_thread.update_signal.connect(self.handle_update)
            self.worker_thread.progress_signal.connect(self.update_progress)
            self.worker_thread.error_signal.connect(self.handle_error)
            self.worker_thread.finished.connect(self.processing_finished_single_task)
            self.summarize_btn.setEnabled(False)
            self.summarize_all_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.batch_size_spinbox.setEnabled(False)
            self.status_label.setText(f"正在处理章节: {current.original_title}...")
            self.progress_bar.setMaximum(1)
            self.progress_bar.setValue(0)
            self.worker_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动单章提炼失败: {str(e)}")
            self.status_label.setText("单章提炼启动失败。")
            self.summarize_btn.setEnabled(True); self.summarize_all_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.batch_size_spinbox.setEnabled(True)

    def prepare_content_store(self, all_task_structs):
        with self.content_store_lock:
            self.batch_content_store.clear()
            for task_data in all_task_structs:
                self.batch_content_store[task_data["identifier"]] = task_data["content"]

    def get_content_for_task(self, identifier):
        with self.content_store_lock:
            return self.batch_content_store.get(identifier)

    def clear_content_for_task(self, identifier):
        with self.content_store_lock:
            self.batch_content_store.pop(identifier, None)

    def summarize_all(self):
        if not self.book_data.get("title") or self.chapter_tree.topLevelItemCount() == 0:
            QMessageBox.information(self, "提示", "请先导入小说。")
            return
        if not self.validate_config(): return
        all_tasks_data_list = []
        book_item = self.chapter_tree.topLevelItem(0)
        if not book_item: return
        for i in range(book_item.childCount()):
            vol_item = book_item.child(i)
            for j in range(vol_item.childCount()):
                chapter_item = vol_item.child(j)
                if isinstance(chapter_item, ChapterTreeItem) and not chapter_item.is_summarized: 
                    identifier = (vol_item.text(0), chapter_item.original_title)
                    all_tasks_data_list.append({"identifier": identifier, "content": chapter_item.content, "context": ""})
        if not all_tasks_data_list:
            QMessageBox.information(self, "提示", "所有章节均已提炼或没有章节可供提炼。")
            return
        self.prepare_content_store(all_tasks_data_list)
        self.pending_batch_tasks_data = [{"identifier": td["identifier"], "context": td["context"]} for td in all_tasks_data_list]
        self.current_batch_chunk_offset = 0
        self.start_batch_processing(len(self.pending_batch_tasks_data))
        self.process_next_batch_chunk()

    def process_next_batch_chunk(self):
        if self.stop_batch_requested and self.active_batch_tasks == 0:
            self.processing_finished(stopped_by_user=True)
            return
        if self.stop_batch_requested: return

        current_chapters_per_batch = self.batch_size_spinbox.value()
        start_index = self.current_batch_chunk_offset
        end_index = start_index + current_chapters_per_batch
        current_chunk_tasks_data = self.pending_batch_tasks_data[start_index:end_index]

        if not current_chunk_tasks_data:
            if self.active_batch_tasks == 0: self.processing_finished(stopped_by_user=self.stop_batch_requested)
            return
        
        self.active_batch_tasks = len(current_chunk_tasks_data)
        try:
            api_config = {"url": self.api_url_input.text().strip(), "key": self.api_key_input.text().strip(), "model": self.get_current_model_name()}
            encoding_object = self.get_tiktoken_encoding(api_config['model'])
            if encoding_object is None:
                QMessageBox.critical(self, "编码器错误", f"无法为模型 {api_config['model']} 初始化Token编码器。批量处理中止。")
                self.processing_finished(stopped_by_user=True); return
        except Exception as e:
            QMessageBox.critical(self, "配置错误", f"API配置或编码器初始化无效: {str(e)}")
            self.processing_finished(stopped_by_user=True); return

        for task_data in current_chunk_tasks_data:
            if self.stop_batch_requested: break
            runnable_task = SummarizationTask(
                chapter_item_identifier=task_data["identifier"],
                chapter_context=task_data["context"],
                api_config=api_config,
                custom_prompt_text=self.custom_prompt,
                main_window_ref=self,
                encoding_object=encoding_object
            )
            runnable_task.signals.update_signal.connect(self.handle_chapter_summary_update)
            runnable_task.signals.progress_signal.connect(self.update_batch_progress)
            runnable_task.signals.error_signal.connect(self.handle_chapter_error)
            runnable_task.signals.finished_signal.connect(self.handle_task_finished)
            self.thread_pool.start(runnable_task)
        self.current_batch_chunk_offset = end_index

    def validate_config(self):
        if not self.api_url_input.text().strip(): QMessageBox.warning(self, "配置错误", "请输入API地址"); return False
        if not self.api_key_input.text().strip(): QMessageBox.warning(self, "配置错误", "请输入API密钥"); return False
        if not self.get_current_model_name(): QMessageBox.warning(self, "配置错误", "请选择或输入模型名称"); return False
        return True

    def stop_processing(self):
        self.stop_batch_requested = True
        if self.worker_thread and self.worker_thread.isRunning(): self.worker_thread.stop()
        self.status_label.setText("停止请求已发送... 等待当前活动任务完成或中止。")
        self.stop_btn.setEnabled(False)
        if self.active_batch_tasks == 0 and not (self.worker_thread and self.worker_thread.isRunning()):
             self.processing_finished(stopped_by_user=True)

    def processing_finished(self, stopped_by_user=False):
        with self.content_store_lock: self.batch_content_store.clear()
        self.summarize_btn.setEnabled(True); self.summarize_all_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        if stopped_by_user: self.status_label.setText(f"批量处理已由用户停止。已处理 {self.chapters_processed_count}/{self.chapters_to_process_total} 章节。")
        elif self.chapters_processed_count == self.chapters_to_process_total and self.chapters_to_process_total > 0: self.status_label.setText(f"批量提炼完成。共处理 {self.chapters_processed_count} 章节。")
        elif self.chapters_to_process_total == 0 : self.status_label.setText("就绪")
        else: self.status_label.setText(f"批量处理结束。已处理 {self.chapters_processed_count}/{self.chapters_to_process_total} 章节。")
        self.progress_bar.setValue(0); self.eta_label.setText("预计剩余时间: N/A"); self.metrics_label.setText("速率: N/A")
        self.active_batch_tasks = 0; self.chapters_to_process_total = 0; self.chapters_processed_count = 0
        self.total_input_tokens_batch = 0; self.total_output_tokens_batch = 0; self.batch_start_time = 0
        self.average_time_per_chapter = 0; self.stop_batch_requested = False
        self.pending_batch_tasks_data = []; self.current_batch_chunk_offset = 0
        for widget in [self.chapter_tree, self.model_combo, self.api_key_input, self.api_url_input, 
                       self.test_btn, self.load_btn, self.save_config_btn, self.load_config_btn, 
                       self.prompt_input, self.batch_size_spinbox, self.add_model_action, self.manage_models_action]:
            if widget: widget.setEnabled(True)
            
    def processing_finished_single_task(self):
        self.summarize_btn.setEnabled(True); self.summarize_all_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.batch_size_spinbox.setEnabled(True)
        current_item = self.chapter_tree.currentItem()
        if current_item and isinstance(current_item, ChapterTreeItem) and current_item.is_summarized: 
            self.status_label.setText(f"章节 '{current_item.original_title}' 处理完成。")
        elif current_item and isinstance(current_item, ChapterTreeItem): 
            self.status_label.setText(f"章节 '{current_item.original_title}' 处理结束。")
        else: self.status_label.setText("单章处理完成。")
        if self.progress_bar.maximum() == 1: self.progress_bar.setValue(1)

    def handle_update(self, update_type, data):
        if update_type == "summary":
            item, summary_text = data
            item.summary = summary_text; item.is_summarized = True; item.summary_timestamp = time.time()
            item.update_display_text()
            if self.chapter_tree.currentItem() == item and not self.summary_mode_btn.isChecked():
                self.content_display.setText(summary_text)
            if item.is_summarized: self.auto_export_novel_data()

    def handle_chapter_summary_update(self, identifier, summary_text):
        item = self.find_chapter_item_by_identifier(identifier)
        if item:
            item.summary = summary_text; item.is_summarized = True; item.summary_timestamp = time.time()
            item.update_display_text()
            if self.chapter_tree.currentItem() == item and not self.summary_mode_btn.isChecked():
                self.content_display.setText(summary_text)
            if item.is_summarized: self.auto_export_novel_data()

    def update_batch_progress(self, in_tokens, out_tokens, chapters_done_this_task):
        self.total_tokens[0] += in_tokens; self.total_tokens[1] += out_tokens
        self.total_input_tokens_batch += in_tokens; self.total_output_tokens_batch += out_tokens
        if self.chapters_to_process_total > 0 and self.chapters_processed_count > 0 and self.batch_start_time > 0:
            elapsed_time = max(1e-6, time.time() - self.batch_start_time)
            self.average_time_per_chapter = elapsed_time / self.chapters_processed_count
            remaining_chapters = self.chapters_to_process_total - self.chapters_processed_count
            self.eta_label.setText(f"ETA: {time.strftime('%H:%M:%S', time.gmtime(remaining_chapters * self.average_time_per_chapter))}" if remaining_chapters > 0 else "ETA: 完成")
            chapters_per_minute = self.chapters_processed_count / (elapsed_time / 60)
            tokens_per_second = (self.total_input_tokens_batch + self.total_output_tokens_batch) / elapsed_time
            self.metrics_label.setText(f"{chapters_per_minute:.2f} 章/分钟 | {tokens_per_second:.2f} Token/秒")
        self.token_label.setText(f"总消耗: 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}")

    def handle_chapter_error(self, identifier, error_msg):
        id_str = f'{identifier[0]}/{identifier[1]}' if isinstance(identifier, tuple) and len(identifier) == 2 else str(identifier)
        print(f"ERROR_BATCH_CHAPTER: 章节 '{id_str}' 处理失败: {error_msg}")
        self.status_label.setText(f"错误: 章节 '{id_str}' - {error_msg[:100]}...")

    def handle_task_finished(self, identifier):
        self.active_batch_tasks -= 1
        self.chapters_processed_count +=1
        if self.chapters_to_process_total > 0: self.progress_bar.setValue(self.chapters_processed_count)
        if self.chapters_processed_count > 0 and self.batch_start_time > 0:
            elapsed_time = max(1e-6, time.time() - self.batch_start_time)
            self.average_time_per_chapter = elapsed_time / self.chapters_processed_count
            remaining_chapters = self.chapters_to_process_total - self.chapters_processed_count
            if remaining_chapters > 0: self.eta_label.setText(f"ETA: {time.strftime('%H:%M:%S', time.gmtime(remaining_chapters * self.average_time_per_chapter))}")
            else: self.eta_label.setText("ETA: 完成")
        if self.active_batch_tasks == 0:
            if self.chapters_processed_count < self.chapters_to_process_total and not self.stop_batch_requested:
                self.process_next_batch_chunk()
            else:
                self.total_tokens[0] += self.total_input_tokens_batch; self.total_tokens[1] += self.total_output_tokens_batch
                self.token_label.setText(f"Token消耗: 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}")
                self.processing_finished(stopped_by_user=self.stop_batch_requested)

    def find_chapter_item_by_identifier(self, identifier):
        if not isinstance(identifier, tuple) or len(identifier) != 2: return None
        target_vol_title, target_chap_original_title = identifier
        if self.chapter_tree.topLevelItemCount() == 0: return None
        book_item = self.chapter_tree.topLevelItem(0)
        if not book_item: return None
        for i in range(book_item.childCount()):
            vol_item = book_item.child(i)
            if vol_item.text(0) == target_vol_title:
                for j in range(vol_item.childCount()):
                    chap_item = vol_item.child(j)
                    if isinstance(chap_item, ChapterTreeItem) and hasattr(chap_item, 'original_title') and chap_item.original_title == target_chap_original_title: 
                        return chap_item
                return None 
        return None

    def handle_error(self, error_msg):
        print(f"ERROR_SINGLE_CHAPTER: 处理错误 (单个章节): {error_msg}")
        self.status_label.setText(f"处理错误: {error_msg[:150]}...")

    def start_batch_processing(self, total_overall_tasks):
        self.chapters_to_process_total = total_overall_tasks; self.chapters_processed_count = 0
        self.total_input_tokens_batch = 0; self.total_output_tokens_batch = 0; self.batch_start_time = time.time()
        self.average_time_per_chapter = 0; self.active_batch_tasks = 0; self.stop_batch_requested = False
        self.progress_bar.setMaximum(total_overall_tasks if total_overall_tasks > 0 else 100); self.progress_bar.setValue(0)
        self.eta_label.setText("预计剩余时间: 计算中..."); self.metrics_label.setText("速率: 计算中...")
        self.status_label.setText(f"批量处理中... {total_overall_tasks}章节待处理")
        for widget in [self.chapter_tree, self.model_combo, self.api_key_input, self.api_url_input, 
                       self.test_btn, self.load_btn, self.save_config_btn, self.load_config_btn, 
                       self.prompt_input, self.batch_size_spinbox, self.add_model_action, self.manage_models_action,
                       self.summarize_btn, self.summarize_all_btn]:
            if widget: widget.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def update_progress(self, in_tokens, out_tokens, count):
        self.total_tokens[0] += in_tokens; self.total_tokens[1] += out_tokens
        self.token_label.setText(f"Token消耗: 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}")
        if self.progress_bar.maximum() == 1: self.progress_bar.setValue(self.progress_bar.value() + count)

    def export_results(self):
        if not self.default_export_path:
            if QMessageBox.question(self, '导出路径', '未设置默认导出路径，是否现在设置？', QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                self.set_export_path()
            else: return
        if not self.default_export_path: return 
        try:
            book_dir = os.path.join(self.default_export_path, f"{self.book_data['title']}_提炼结果")
            os.makedirs(book_dir, exist_ok=True)
            self.export_txt(book_dir); self.export_markdown(book_dir); self.export_json(book_dir)
            QMessageBox.information(self, "导出完成", f"结果已保存到:\n{book_dir}")
        except Exception as e: QMessageBox.critical(self, "导出错误", str(e))

    def export_txt(self, output_directory_path):
        with open(os.path.join(output_directory_path, f"{self.book_data['title']}_提炼总结.txt"), 'w', encoding='utf-8') as f:
            f.write(f"{self.book_data['title']} 提炼总结\n{'=' * 50}\n\n")
            if self.chapter_tree.topLevelItemCount() == 0: return
            book_item = self.chapter_tree.topLevelItem(0)
            if not book_item: return
            for i in range(book_item.childCount()):
                vol_item = book_item.child(i)
                f.write(f"{vol_item.text(0)}\n{'-' * 40}\n")
                for j in range(vol_item.childCount()):
                    chap_item = vol_item.child(j)
                    if isinstance(chap_item, ChapterTreeItem): 
                        f.write(f"\n### {chap_item.original_title} ###\n")
                        f.write(f"_(提炼后 - {time.strftime('%Y-%m-%d %H:%M', time.localtime(chap_item.summary_timestamp))})_\n" if chap_item.is_summarized else "(原文)\n")
                        f.write(f"{chap_item.summary if chap_item.is_summarized else chap_item.content}\n\n")
                f.write("\n\n")

    def export_markdown(self, output_directory_path):
        with open(os.path.join(output_directory_path, f"{self.book_data['title']}_提炼总结.md"), 'w', encoding='utf-8') as f:
            f.write(f"# {self.book_data['title']} 提炼总结\n\n")
            f.write(f"**生成时间:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Token消耗:** 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}\n\n---\n\n")
            if self.chapter_tree.topLevelItemCount() == 0: return
            book_item = self.chapter_tree.topLevelItem(0)
            if not book_item: return
            for i in range(book_item.childCount()):
                vol_item = book_item.child(i)
                f.write(f"## {vol_item.text(0)}\n\n")
                for j in range(vol_item.childCount()):
                    chap_item = vol_item.child(j)
                    if isinstance(chap_item, ChapterTreeItem): 
                        f.write(f"### {chap_item.original_title}\n\n")
                        f.write(f"_(提炼后 - {time.strftime('%Y-%m-%d %H:%M', time.localtime(chap_item.summary_timestamp))})_\n\n" if chap_item.is_summarized else "_(原文)_\n\n")
                        f.write(f"{chap_item.summary if chap_item.is_summarized else chap_item.content}\n\n")
                f.write("---\n\n")

    def export_json(self, output_directory_path):
        data = {"title": self.book_data["title"], "export_time": time.strftime('%Y-%m-%d %H:%M:%S'), "token_usage": {"input": self.total_tokens[0], "output": self.total_tokens[1]}, "volumes": []}
        if self.chapter_tree.topLevelItemCount() == 0: return
        book_item = self.chapter_tree.topLevelItem(0)
        if not book_item: return
        for i in range(book_item.childCount()):
            vol_item = book_item.child(i)
            volume_data = {"title": vol_item.text(0), "chapters": []}
            for j in range(vol_item.childCount()):
                chap_item = vol_item.child(j)
                if isinstance(chap_item, ChapterTreeItem): 
                    volume_data["chapters"].append({"title": chap_item.original_title, "original_length": chap_item.word_count, "is_summarized": chap_item.is_summarized, "summary": chap_item.summary if chap_item.is_summarized else "", "summary_length": len(chap_item.summary) if chap_item.is_summarized else 0, "summary_timestamp": chap_item.summary_timestamp if chap_item.is_summarized else 0, "content": chap_item.content})
            if volume_data["chapters"]: data["volumes"].append(volume_data)
        with open(os.path.join(output_directory_path, f"{self.book_data['title']}_数据.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def set_export_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择默认导出目录")
        if path: self.default_export_path = path; self.status_label.setText(f"导出目录设置为: {path}")

    def add_custom_model(self):
        dialog = QDialog(self); dialog.setWindowTitle("添加自定义模型"); dialog.setModal(True)
        layout = QFormLayout(); name_input = QLineEdit(); name_input.setPlaceholderText("例如: my-custom-model (唯一标识)")
        layout.addRow("模型ID (唯一):", name_input); display_name_input = QLineEdit(); display_name_input.setPlaceholderText("例如: 我的自定义模型")
        layout.addRow("显示名称:", display_name_input); url_input = QLineEdit(); url_input.setPlaceholderText("https://api.example.com/v1/chat/completions")
        layout.addRow("API地址:", url_input); buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addRow(buttons)
        dialog.setLayout(layout)
        if dialog.exec_() == QDialog.Accepted:
            model_id = name_input.text().strip(); display_name = display_name_input.text().strip(); api_url = url_input.text().strip()
            if model_id and display_name and api_url:
                if model_id in self.model_configs: QMessageBox.warning(self, "错误", f"模型ID '{model_id}' 已存在。"); return
                self.model_configs[model_id] = {"url": api_url, "display_name": display_name}
                self.model_combo.addItem(display_name, model_id); self.model_combo.setCurrentText(display_name)
                QMessageBox.information(self, "成功", f"已添加自定义模型: {display_name}")
            else: QMessageBox.warning(self, "错误", "请填写模型ID、显示名称和API地址。")

    def save_config(self, silent=False):
        if silent and not self.book_data.get("file_path"): return 
        config = {
            "model": self.get_current_model_name(), "api_url": self.api_url_input.text(),
            "api_key": self.api_key_input.text(), "export_path": self.default_export_path,
            "custom_models": {k: v for k, v in self.model_configs.items() if k not in self.initial_model_keys},
            "book_data": self.book_data, "summary_mode": self.summary_mode_btn.isChecked(),
            "custom_prompt": self.custom_prompt, "chapter_states": self.get_chapter_states(),
            "batch_size": self.batch_size_spinbox.value()
        }
        try:
            with open("config.json", 'w', encoding='utf-8') as f: json.dump(config, f, ensure_ascii=False, indent=2)
            if not silent: QMessageBox.information(self, "成功", "配置已保存")
            elif hasattr(self, 'status_label'): self.status_label.setText("配置已自动保存")
        except Exception as e:
            if not silent: QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")
            else: print(f"Error during silent save_config: {e}")

    def get_chapter_states(self):
        states = []
        if self.chapter_tree.topLevelItemCount() == 0: return states
        book_item = self.chapter_tree.topLevelItem(0)
        if not book_item: return states
        for i in range(book_item.childCount()):
            vol_item = book_item.child(i)
            for j in range(vol_item.childCount()):
                chap_item = vol_item.child(j)
                if isinstance(chap_item, ChapterTreeItem): 
                    states.append({"path": [vol_item.text(0), chap_item.original_title],"is_summarized": chap_item.is_summarized, "summary": chap_item.summary, "timestamp": chap_item.summary_timestamp, "content": chap_item.content, "word_count": chap_item.word_count})
        return states

    def load_config(self, silent=False):
        config_path = "config.json"; default_batch_size = 10
        if not os.path.exists(config_path):
            if not silent: QMessageBox.information(self, "提示", "未找到配置文件。")
            self.batch_size_spinbox.setValue(default_batch_size); return
        try:
            with open(config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
            if "custom_models" in config_data:
                for k, v in config_data["custom_models"].items():
                    if k not in self.initial_model_keys: self.model_configs[k] = v; self.model_combo.addItem(v.get("display_name", k), k)
            model_to_select = config_data.get("model")
            if model_to_select:
                idx = self.model_combo.findData(model_to_select)
                if idx == -1: idx = self.model_combo.findText(model_to_select)
                if idx != -1: self.model_combo.setCurrentIndex(idx)
            self.api_url_input.setText(config_data.get("api_url", "")); self.api_key_input.setText(config_data.get("api_key", ""))
            self.default_export_path = config_data.get("export_path", ""); self.custom_prompt = config_data.get("custom_prompt", DEFAULT_CUSTOM_PROMPT)
            self.prompt_input.setText(self.custom_prompt)
            try: loaded_batch_size = int(config_data.get("batch_size", default_batch_size))
            except (ValueError, TypeError): loaded_batch_size = default_batch_size
            if not (self.batch_size_spinbox.minimum() <= loaded_batch_size <= self.batch_size_spinbox.maximum()): loaded_batch_size = default_batch_size
            self.batch_size_spinbox.setValue(loaded_batch_size)
            reloaded_ok = False
            if "book_data" in config_data and config_data["book_data"].get("file_path"):
                self.book_data = config_data["book_data"]; reloaded_ok = self.reload_novel()
            if reloaded_ok and "chapter_states" in config_data: self.restore_chapter_states(config_data["chapter_states"])
            if "summary_mode" in config_data: self.summary_mode_btn.setChecked(config_data["summary_mode"]); self.toggle_display_mode()
            status_msg = "配置已加载"
            if reloaded_ok and self.book_data.get('title'): status_msg += f", 上次打开: {self.book_data['title']}"
            if not silent: QMessageBox.information(self, "成功", status_msg)
            elif hasattr(self, 'status_label'): self.status_label.setText(status_msg)
        except Exception as e:
            if not silent: QMessageBox.critical(self, "错误", f"加载配置失败: {str(e)}")
            else: print(f"Error during silent load_config: {e}")
            self.batch_size_spinbox.setValue(default_batch_size) 

    def reload_novel(self):
        file_path = self.book_data.get("file_path"); encoding = self.book_data.get("encoding")
        if not file_path or not os.path.exists(file_path) or not encoding:
            self.book_data = {"title": "", "volumes": []}; self.chapter_tree.clear(); return False
        if hasattr(self, 'status_label'): self.status_label.setText(f"重新加载: {self.book_data.get('title', '未知')}...")
        QApplication.processEvents()
        try:
            with open(file_path, 'r', encoding=encoding) as f: content = f.read()
            chapters = self.parse_chapters(content); self.build_chapter_tree(chapters)
            if hasattr(self, 'status_label'): self.status_label.setText(f"已重新加载: {self.book_data.get('title', '未知')} (编码: {encoding})")
            return True
        except Exception as e:
            self.book_data = {"title": "", "volumes": []}; self.chapter_tree.clear()
            if hasattr(self, 'status_label'): self.status_label.setText(f"重新加载失败: {str(e)}")
            return False

    def restore_chapter_states(self, states):
        if self.chapter_tree.topLevelItemCount() == 0: return
        book_item = self.chapter_tree.topLevelItem(0)
        if not book_item: return
        self.chapter_tree.setUpdatesEnabled(False)
        try:
            for state in states:
                path = state.get("path"); 
                if not path or len(path) != 2: continue
                vol_title, chap_title = path
                for i in range(book_item.childCount()):
                    vol_item = book_item.child(i)
                    if vol_item.text(0) == vol_title:
                        for j in range(vol_item.childCount()):
                            chap_item = vol_item.child(j)
                            if isinstance(chap_item, ChapterTreeItem) and chap_item.original_title == chap_title: 
                                chap_item.is_summarized = state.get("is_summarized", False)
                                chap_item.summary = state.get("summary", "")
                                chap_item.summary_timestamp = float(state.get("timestamp", 0))
                                chap_item.content = state.get("content", chap_item.content) 
                                chap_item.word_count = state.get("word_count", len(chap_item.content))
                                chap_item.setText(1, f"{chap_item.word_count}字")
                                chap_item.update_display_text()
                                break 
                        break 
        finally: self.chapter_tree.setUpdatesEnabled(True)

    def update_prompt(self): self.custom_prompt = self.prompt_input.text()

    def test_connection(self):
        if not self.validate_config(): return
        try:
            api_config = {"url": self.api_url_input.text().strip(), "key": self.api_key_input.text().strip(), "model": self.get_current_model_name()}
            encoding_object = self.get_tiktoken_encoding(api_config['model'])
            if encoding_object is None:
                QMessageBox.critical(self, "编码器错误", f"无法为模型 '{api_config.get('model','未知')}' 初始化Token编码器。测试中止。"); return
            processor = LLMProcessor(api_config, self.custom_prompt, encoding_object) 
            self.status_label.setText("正在测试连接..."); QApplication.processEvents()
            summary, _, _ = processor.summarize("这是一个连接测试。", max_retries=1)
            if summary or summary == "": 
                QMessageBox.information(self, "连接成功", f"API连接测试成功！\n模型: {api_config['model']}\n返回: {summary[:100]}...")
                self.status_label.setText("连接测试成功")
            else: raise ValueError("API返回内容为空或无效 (None)") 
        except Exception as e:
            QMessageBox.critical(self, "连接失败", f"API连接测试失败: {str(e)}")
            self.status_label.setText("连接测试失败")

    def auto_save(self): self.save_config(silent=True)

    def auto_export_novel_data(self):
        if not self.book_data.get("title"): return
        book_title_safe = re.sub(r'[\/*?:"<>|]', "_", self.book_data["title"])
        if not book_title_safe.strip(): book_title_safe = "Untitled_Novel"
        export_path = os.path.join(self.auto_export_base_dir, book_title_safe)
        try:
            os.makedirs(export_path, exist_ok=True)
            self.export_txt(export_path); self.export_markdown(export_path) 
            if hasattr(self, 'status_label'): self.status_label.setText(f"'{book_title_safe}' 已自动保存到桌面。")
        except Exception as e: 
            if hasattr(self, 'status_label'): self.status_label.setText(f"自动导出到桌面失败: {str(e)}")

    def save_chapter_edits(self): 
        pass 

    def get_tiktoken_encoding(self, model_name_from_config: str):
        effective_encoding_key = None
        try:
            _model_key_for_tiktoken = model_name_from_config.split('/')[-1].lower()
            encoding_map = {'gpt-4': 'cl100k_base', 'gpt-3.5-turbo': 'cl100k_base', 'deepseek-chat': 'cl100k_base', 'qwen': 'cl100k_base', 'chatglm': 'cl100k_base'}
            try:
                effective_encoding_key = _model_key_for_tiktoken
                with self.tiktoken_cache_lock:
                    if effective_encoding_key in self.tiktoken_encoding_cache: return self.tiktoken_encoding_cache[effective_encoding_key]
                    try:
                        encoding_obj = tiktoken.encoding_for_model(effective_encoding_key)
                        self.tiktoken_encoding_cache[effective_encoding_key] = encoding_obj
                        return encoding_obj
                    except KeyError: pass 
                derived_encoding_key = None
                for prefix, base_encoding_name in encoding_map.items():
                    if _model_key_for_tiktoken.startswith(prefix): derived_encoding_key = base_encoding_name; break
                if not derived_encoding_key: derived_encoding_key = 'cl100k_base'
                effective_encoding_key = derived_encoding_key
                with self.tiktoken_cache_lock:
                    if effective_encoding_key in self.tiktoken_encoding_cache: return self.tiktoken_encoding_cache[effective_encoding_key]
                    encoding_obj = tiktoken.get_encoding(effective_encoding_key)
                    self.tiktoken_encoding_cache[effective_encoding_key] = encoding_obj
                    return encoding_obj
            except Exception as e: 
                print(f"ERROR: Failed to get/create tiktoken encoding for '{model_name_from_config}'. Error: {e}. Using fallback 'cl100k_base'.")
                with self.tiktoken_cache_lock:
                    if 'cl100k_base' in self.tiktoken_encoding_cache: return self.tiktoken_encoding_cache['cl100k_base']
                    try:
                        encoding_obj = tiktoken.get_encoding('cl100k_base')
                        self.tiktoken_encoding_cache['cl100k_base'] = encoding_obj
                        return encoding_obj
                    except Exception as e_default:
                        print(f"CRITICAL ERROR: Failed to get default tiktoken encoder 'cl100k_base': {e_default}")
                        return None
    
    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(self, '确认退出', '正在处理任务，确定要退出吗？',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.worker_thread.stop()
                self.worker_thread.wait(3000) 
                try: self.save_config(silent=True)
                except Exception as e: print(f"Error saving config during forced close: {e}")
                event.accept()
                return 
            else:
                event.ignore()
                return 
        try: self.save_config(silent=True)
        except Exception as e: print(f"Error saving config on close: {e}")
        event.accept() 

# Main execution block
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("小说智能分析工具")
    app.setApplicationVersion("2.0")
    
    window = MainWindow() 
    window.show()
    
    sys.exit(app.exec_())
