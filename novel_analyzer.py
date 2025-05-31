import os
import re
import json
import threading
import queue
import time
import tiktoken
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem,
                             QTextEdit, QFileDialog, QPushButton, QComboBox,
                             QProgressBar, QLabel, QSplitter, QVBoxLayout,
                             QHBoxLayout, QWidget, QAction, QMessageBox, QLineEdit)
from PyQt5.QtCore import pyqtSignal, Qt, QThread, QTimer
import requests


class LLMProcessor:
    def __init__(self, api_config, main_window): # Added main_window
        self.api_url = api_config['url']
        self.api_key = api_config['key']
        self.model = api_config['model']
        self.main_window = main_window # Store main_window reference
        
        # 安全地初始化tiktoken编码器
        try:
            model_name = self.model.split('/')[-1].lower()
            # 映射一些常见模型到tiktoken支持的编码器
            encoding_map = {
                'gpt-4': 'cl100k_base',
                'gpt-3.5-turbo': 'cl100k_base',
                'deepseek-chat': 'cl100k_base',  # 使用通用编码器
                'qwen-turbo': 'cl100k_base',
                'chatglm-pro': 'cl100k_base',
                'claude': 'cl100k_base',
                'gemini': 'cl100k_base'
            }
            
            # 尝试获取模型专用编码器，失败则使用通用编码器
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # 使用通用编码器
                encoding_name = encoding_map.get(model_name.split('-')[0], 'cl100k_base')
                self.encoding = tiktoken.get_encoding(encoding_name)
                
        except Exception as e:
            print(f"编码器初始化警告: {e}, 使用默认编码器")
            self.encoding = tiktoken.get_encoding('cl100k_base')
            
        self.last_call = 0

    def calculate_tokens(self, text):
        """使用tiktoken精确计算token数"""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # 简单估算：中文约1.5字符/token，英文约4字符/token
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4)

    def summarize(self, text, context="", max_retries=3):
        """调用API进行内容提炼"""
        current_time = time.time()
        if current_time - self.last_call < 1.0:
            time.sleep(1.0 - (current_time - self.last_call))

        # 构造提示词
        prompt = f"{self.main_window.custom_prompt}\n{text}" if self.main_window.custom_prompt else f"作为专业编辑，请用原文语言提炼以下内容的核心要点（保留关键情节和人物关系，压缩至原文1%字数）：\n{text}"
        if context:
            prompt = f"上下文：{context}\n\n{prompt}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 构造请求数据
        data = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": min(4000, max(100, int(len(text) * 0.015)))
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code != 200:
                    raise requests.RequestException(f"HTTP {response.status_code}: {response.text}")
                
                result = response.json()

                # 处理API错误响应
                if 'error' in result:
                    error_msg = result['error'].get('message', '未知错误')
                    error_code = result['error'].get('code', 'unknown')
                    
                    if 'model' in error_msg.lower() or error_code == 'model_not_found':
                        raise ValueError(f"模型 {self.model} 不存在")
                    elif 'api_key' in error_msg.lower() or error_code == 'invalid_api_key':
                        raise PermissionError("API密钥无效")
                    else:
                        raise RuntimeError(f"API错误: {error_msg}")
                
                # 提取结果
                if 'choices' in result and len(result['choices']) > 0:
                    summary = result['choices'][0]['message']['content']
                else:
                    raise ValueError("API返回格式异常")
                
                # 计算token消耗
                input_tokens = result.get('usage', {}).get('prompt_tokens', 0)
                output_tokens = result.get('usage', {}).get('completion_tokens', 0)
                
                # 如果API没返回usage信息，使用本地计算
                if input_tokens == 0:
                    input_tokens = self.calculate_tokens(prompt)
                if output_tokens == 0:
                    output_tokens = self.calculate_tokens(summary)
                
                self.last_call = time.time()
                return summary, input_tokens, output_tokens

            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"API调用失败: {str(e)}")
                time.sleep(2 ** attempt)

        return "", 0, 0


class ChapterTreeItem(QTreeWidgetItem):
    """自定义树节点存储章节数据"""
    def __init__(self, title, content, word_count, parent=None):
        super().__init__(parent, [title, f"{word_count}字"])
        self.original_title = title # Store original title
        self.content = content
        self.summary = ""
        self.word_count = word_count
        self.is_summarized = False
        self.summary_timestamp = 0

    def update_display_text(self):
        """Updates the chapter title in the tree to reflect summary status."""
        current_title = self.text(0)
        marker = "* "
        if self.is_summarized:
            if not current_title.startswith(marker):
                self.setText(0, marker + self.original_title)
        else:
            if current_title.startswith(marker):
                self.setText(0, self.original_title)


class WorkerThread(QThread):
    """工作线程处理任务队列"""
    update_signal = pyqtSignal(str, object)
    progress_signal = pyqtSignal(int, int, int)
    error_signal = pyqtSignal(str)
    
    def __init__(self, work_queue, llm_processor):
        super().__init__()
        self.work_queue = work_queue
        self.llm_processor = llm_processor
        self.running = True
    
    def run(self):
        while self.running and not self.work_queue.empty():
            try:
                task = self.work_queue.get_nowait()
                item, context = task
                
                # 调用LLM处理
                summary, in_tokens, out_tokens = self.llm_processor.summarize(
                    item.content, context
                )
                
                # 发送更新信号
                self.update_signal.emit("summary", (item, summary))
                self.progress_signal.emit(in_tokens, out_tokens, 1)
                
                time.sleep(0.5)  # 避免API速率限制
                
            except queue.Empty:
                break
            except Exception as e:
                self.error_signal.emit(f"处理错误: {str(e)}")
                break
    
    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("小说智能分析工具 v2.0")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化变量
        self.book_data = {"title": "", "volumes": []}
        self.llm_processor = None
        self.work_queue = queue.Queue()
        self.worker_thread = None
        self.total_tokens = [0, 0]  # [input, output]
        self.default_export_path = ""
        self.custom_prompt = "提炼以下文本的核心要点，仅输出提炼后的内容，不要包含任何额外解释或与原文无关的文字。保留关键情节和人物关系，压缩至原文1%字数："

        # 预定义的模型配置
        self.model_configs = {
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
            
            # 自定义模型
            "custom": {
                "url": "",
                "display_name": "自定义模型"
            }
        }

        self.init_ui()
        self.auto_save_timer = QTimer()  # 新增自动保存定时器
        self.auto_save_timer.timeout.connect(self.auto_save)
        self.auto_save_timer.start(30000)  # 30秒自动保存

    def init_ui(self):
        # 创建主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # 顶部控制栏
        control_layout = QHBoxLayout()
        
        # 模型选择
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        
        # 添加预定义模型
        for model_key, config in self.model_configs.items():
            self.model_combo.addItem(config["display_name"], model_key)
        
        self.model_combo.setCurrentIndex(-1)
        self.model_combo.setPlaceholderText("选择或输入模型名称")
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        
        control_layout.addWidget(QLabel("选择模型:"))
        control_layout.addWidget(self.model_combo)

        # API密钥输入
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("输入API密钥")
        self.api_key_input.setEchoMode(QLineEdit.Password)  # 隐藏密钥显示
        control_layout.addWidget(QLabel("API密钥:"))
        control_layout.addWidget(self.api_key_input)

        # API地址输入
        self.api_url_input = QLineEdit()
        self.api_url_input.setPlaceholderText("API服务地址")
        control_layout.addWidget(QLabel("API地址:"))
        control_layout.addWidget(self.api_url_input)

        # 测试连接按钮
        self.test_btn = QPushButton("测试连接")
        self.test_btn.clicked.connect(self.test_connection)
        control_layout.addWidget(self.test_btn)

        main_layout.addLayout(control_layout)

        # 第二行控制栏
        control_layout2 = QHBoxLayout()
        
        self.load_btn = QPushButton("导入小说")
        self.load_btn.clicked.connect(self.load_novel)
        control_layout2.addWidget(self.load_btn)

        self.summary_mode_btn = QPushButton("显示原文") # Initial text: click to see original (implies summary is default)
        self.summary_mode_btn.setCheckable(True)
        # Default state is not checked. If an item is summarized, summary is shown. Button says "Show Original".
        # If button is clicked (becomes checked), original is shown. Button says "Show Summary".
        self.summary_mode_btn.clicked.connect(self.toggle_display_mode)
        control_layout2.addWidget(self.summary_mode_btn)

        # 添加保存/加载配置按钮
        self.save_config_btn = QPushButton("保存配置")
        self.save_config_btn.clicked.connect(self.save_config)
        control_layout2.addWidget(self.save_config_btn)
        
        self.load_config_btn = QPushButton("加载配置")
        self.load_config_btn.clicked.connect(self.load_config)
        control_layout2.addWidget(self.load_config_btn)

        # 在控制栏第二行添加提示词输入
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("输入自定义提示词")
        self.prompt_input.setText(self.custom_prompt)
        self.prompt_input.textChanged.connect(self.update_prompt)
        control_layout2.addWidget(QLabel("提示词:"))
        control_layout2.addWidget(self.prompt_input)

        main_layout.addLayout(control_layout2)

        # 分割视图
        splitter = QSplitter(Qt.Horizontal)

        # 章节树
        self.chapter_tree = QTreeWidget()
        self.chapter_tree.setHeaderLabels(["章节/卷", "字数"])
        self.chapter_tree.itemClicked.connect(self.show_content)

        # Configure column resizing
        header = self.chapter_tree.header()
        header.setSectionResizeMode(0, header.Stretch) # Chapter/Volume Name column
        header.setMinimumSectionSize(250) # Minimum width for chapter/volume column
        header.setSectionResizeMode(1, header.ResizeToContents) # Word Count column
        header.setStretchLastSection(False) # Add this line

        splitter.addWidget(self.chapter_tree)

        # 内容显示区
        right_panel = QVBoxLayout()
        self.content_display = QTextEdit()
        self.content_display.setReadOnly(True)
        right_panel.addWidget(self.content_display)

        # 操作按钮组
        btn_layout = QHBoxLayout()
        self.summarize_btn = QPushButton("提炼当前")
        self.summarize_btn.clicked.connect(self.summarize_selected)
        btn_layout.addWidget(self.summarize_btn)

        self.summarize_all_btn = QPushButton("一键提炼")
        self.summarize_all_btn.clicked.connect(self.summarize_all)
        btn_layout.addWidget(self.summarize_all_btn)

        self.stop_btn = QPushButton("停止处理")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        self.export_btn = QPushButton("导出结果")
        self.export_btn.clicked.connect(self.export_results)
        btn_layout.addWidget(self.export_btn)

        self.edit_chapter_btn = QPushButton("编辑章节")
        self.edit_chapter_btn.clicked.connect(self.enable_chapter_editing)
        self.edit_chapter_btn.setEnabled(False) # Initially disabled
        btn_layout.addWidget(self.edit_chapter_btn)

        self.save_chapter_btn = QPushButton("保存修改")
        self.save_chapter_btn.clicked.connect(self.save_chapter_edits)
        self.save_chapter_btn.setEnabled(False) # Initially disabled
        btn_layout.addWidget(self.save_chapter_btn)
        
        right_panel.addLayout(btn_layout)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])

        main_layout.addWidget(splitter, 4)

        # 底部状态栏
        status_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        status_layout.addWidget(self.progress_bar)

        self.token_label = QLabel("Token消耗: 输入 0 | 输出 0")
        status_layout.addWidget(self.token_label)

        self.status_label = QLabel("就绪")
        status_layout.addWidget(self.status_label)
        main_layout.addLayout(status_layout)

        # 菜单栏
        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件')

        export_action = QAction('设置导出路径', self)
        export_action.triggered.connect(self.set_export_path)
        file_menu.addAction(export_action)

        model_menu = menubar.addMenu('模型管理')
        add_model_action = QAction('添加自定义模型', self)
        add_model_action.triggered.connect(self.add_custom_model)
        model_menu.addAction(add_model_action)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def on_model_changed(self):
        """当模型选择改变时更新API地址"""
        try:
            current_text = self.model_combo.currentText()
            current_data = self.model_combo.currentData()
            
            # 如果是预定义模型
            if current_data and current_data in self.model_configs:
                config = self.model_configs[current_data]
                self.api_url_input.setText(config["url"])
            # 如果是自定义输入
            elif current_text and current_text not in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
                # 保持当前API地址不变，让用户手动输入
                pass
        except Exception as e:
            print(f"模型切换警告: {e}")

    def load_novel(self):
        """导入小说文件并解析章节结构"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择小说文件", "", 
            "文本文件 (*.txt);;所有文件 (*)"
        )
        if not file_path:
            return

        self.status_label.setText("解析文件中...")
        QApplication.processEvents()

        try:
            # 尝试多种编码格式读取文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
            content = None
            
            successful_encoding = None
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    successful_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
                    
            if content is None:
                raise ValueError("无法解码文件，请检查文件编码格式")

            # 智能章节分割
            chapters = self.parse_chapters(content)

            self.book_data["title"] = os.path.splitext(os.path.basename(file_path))[0]
            self.book_data["file_path"] = file_path
            self.book_data["encoding"] = successful_encoding # Save the encoding

            self.build_chapter_tree(chapters) # Build tree after book_data is set

            self.status_label.setText(f"已加载: {self.book_data['title']}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"文件加载失败: {str(e)}")

    def parse_chapters(self, content):
        """解析章节结构"""
        chapters = []
        
        # 多种章节标题模式
        patterns = [
            r'第([一二三四五六七八九十百千万零\d]+)[卷部][\s　]*(.+?)(?=\n|$)',
            r'([卷部])([一二三四五六七八九十百千万零\d]+)[\s　]*(.+?)(?=\n|$)',
            r'第([一二三四五六七八九十百千万零\d]+)[章节回][\s　]*(.+?)(?=\n|$)',
            r'([章节回])([一二三四五六七八九十百千万零\d]+)[\s　]*(.+?)(?=\n|$)',
            r'^\s*(\d+)\.(.+?)(?=\n|$)',  # 数字编号
        ]
        
        lines = content.split('\n')
        current_volume = None
        current_chapter = None
        content_buffer = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                content_buffer.append('')
                continue
                
            # 检查是否为卷标题
            volume_match = None
            for pattern in patterns[:2]:  # 只检查卷/部的模式
                volume_match = re.match(pattern, line)
                if volume_match:
                    break
                    
            if volume_match:
                # 保存当前章节
                if current_chapter:
                    current_chapter['content'] = '\n'.join(content_buffer).strip()
                    current_chapter['word_count'] = len(current_chapter['content'])
                    content_buffer = []
                
                # 保存当前卷
                if current_volume:
                    chapters.append(current_volume)
                
                # 创建新卷
                current_volume = {
                    'title': line,
                    'chapters': [],
                    'content': '',
                    'word_count': 0
                }
                current_chapter = None
                continue
                
            # 检查是否为章节标题
            chapter_match = None
            for pattern in patterns[2:]:  # 检查章/节/回的模式
                chapter_match = re.match(pattern, line)
                if chapter_match:
                    break
                    
            if chapter_match:
                # 保存当前章节
                if current_chapter:
                    current_chapter['content'] = '\n'.join(content_buffer).strip()
                    current_chapter['word_count'] = len(current_chapter['content'])
                    if current_volume:
                        current_volume['chapters'].append(current_chapter)
                
                # 创建新章节
                current_chapter = {
                    'title': line,
                    'content': '',
                    'word_count': 0
                }
                content_buffer = []
                
                # 如果没有卷，创建默认卷
                if not current_volume:
                    current_volume = {
                        'title': '正文',
                        'chapters': [],
                        'content': '',
                        'word_count': 0
                    }
                continue
                
            # 普通内容行
            content_buffer.append(line)
        
        # 处理最后的章节和卷
        if current_chapter:
            current_chapter['content'] = '\n'.join(content_buffer).strip()
            current_chapter['word_count'] = len(current_chapter['content'])
            if current_volume:
                current_volume['chapters'].append(current_chapter)
        
        if current_volume:
            chapters.append(current_volume)
            
        # 如果没有解析到任何章节，创建单一章节
        if not chapters:
            chapters = [{
                'title': '全文',
                'chapters': [{
                    'title': '内容',
                    'content': content,
                    'word_count': len(content)
                }],
                'content': '',
                'word_count': len(content)
            }]
        
        return chapters

    def build_chapter_tree(self, chapters):
        """构建章节树形结构"""
        self.chapter_tree.clear()
        root = QTreeWidgetItem(self.chapter_tree, [self.book_data["title"], ""])
        root.setExpanded(True)

        total_chapters = 0
        total_words = 0

        for volume in chapters:
            vol_words = sum(chapter['word_count'] for chapter in volume['chapters'])
            vol_item = QTreeWidgetItem(root, [volume['title'], f"{len(volume['chapters'])}章, {vol_words}字"])
            vol_item.setExpanded(True)

            for chapter in volume['chapters']:
                chapter_item = ChapterTreeItem(
                    chapter['title'],
                    chapter['content'],
                    chapter['word_count'],
                    vol_item
                )
                total_chapters += 1

            total_words += vol_words

        root.setText(1, f"{len(chapters)}卷, {total_chapters}章, {total_words}字")
        self.chapter_tree.expandAll()

    def show_content(self, item):
        """显示选中章节内容. Shows summary by default if available."""
        if not isinstance(item, ChapterTreeItem):
            self.content_display.clear()
            self.summary_mode_btn.setEnabled(False) # Disable button if no valid item selected
            self.summary_mode_btn.setText("显示原文") # Reset text for consistency
            self.summary_mode_btn.setText("显示原文") # Reset text for consistency
            self.summary_mode_btn.setChecked(False) # Ensure button is not left checked
            self.content_display.setReadOnly(True) # Ensure content display is read-only
            # Disable editing buttons if no valid chapter item is selected
            self.edit_chapter_btn.setEnabled(False)
            self.save_chapter_btn.setEnabled(False)
            # Re-enable other general buttons if they were disabled by editing
            self.summarize_btn.setEnabled(True) # Assuming a book is loaded
            self.summarize_all_btn.setEnabled(True) # Assuming a book is loaded
            self.export_btn.setEnabled(True) # Assuming a book is loaded
            self.load_btn.setEnabled(True)
            return

        # Item is a ChapterTreeItem or similar (e.g. volume root)
        self.content_display.setReadOnly(True) # Default to read-only
        self.save_chapter_btn.setEnabled(False) # Save button disabled by default

        is_chapter = isinstance(item, ChapterTreeItem)
        self.edit_chapter_btn.setEnabled(is_chapter) # Enable edit only for chapters

        # General buttons re-enabled when selection changes and not in edit mode
        self.summarize_btn.setEnabled(is_chapter) # Summarize current only for chapters
        self.summarize_all_btn.setEnabled(True) # Always enabled if a book is loaded
        self.export_btn.setEnabled(True) # Always enabled if a book is loaded
        self.load_btn.setEnabled(True)


        if item.is_summarized:
            self.summary_mode_btn.setEnabled(True)
            self.summary_mode_btn.setChecked(False)
            self.content_display.setText(item.summary)
            self.summary_mode_btn.setText("显示原文")
        else: # Not summarized (or not summarizable, e.g. a volume item)
            self.content_display.setText(item.content if is_chapter else "") # Show content for chapter, nothing for volume
            self.summary_mode_btn.setText("显示要点")
            self.summary_mode_btn.setChecked(False)
            self.summary_mode_btn.setEnabled(is_chapter and item.is_summarized) # Enable toggle only if it's a chapter AND has a summary

    def toggle_display_mode(self):
        """切换显示模式 (原文/提炼后). Called AFTER button state has changed by user click."""
        current_item = self.chapter_tree.currentItem()

        if current_item and isinstance(current_item, ChapterTreeItem):
            if current_item.is_summarized:
                # Button state has already been toggled by the click.
                if self.summary_mode_btn.isChecked(): # Now checked, so user wants to see Original
                    self.content_display.setText(current_item.content)
                    self.summary_mode_btn.setText("显示要点")
                else: # Now unchecked, so user wants to see Summary
                    self.content_display.setText(current_item.summary)
                    self.summary_mode_btn.setText("显示原文")
            else:
                # Should not happen if button is disabled for non-summarized items, but as a fallback:
                self.content_display.setText(current_item.content)
                self.summary_mode_btn.setText("显示要点")
                self.summary_mode_btn.setEnabled(False)
        elif not current_item: # No item selected
             # This case should ideally not be hit if button is disabled when no item selected,
             # but if it is, sync text with (now changed) check state.
            if self.summary_mode_btn.isChecked():
                 self.summary_mode_btn.setText("显示要点")
            else:
                 self.summary_mode_btn.setText("显示原文")

    def get_current_model_name(self):
        """获取当前选择的模型名称"""
        current_data = self.model_combo.currentData()
        if current_data:
            return current_data
        else:
            return self.model_combo.currentText().strip()

    def summarize_selected(self):
        """提炼当前选中章节"""
        current = self.chapter_tree.currentItem()
        if not current or not isinstance(current, ChapterTreeItem):
            QMessageBox.warning(self, "提示", "请先选择要提炼的章节")
            return

        if not self.validate_config():
            return

        # 获取上下文
        context = ""
        parent = current.parent()
        if parent and hasattr(parent, 'summary') and parent.summary:
            context = parent.summary

        # 初始化处理器
        try:
            api_config = {
                "url": self.api_url_input.text().strip(),
                "key": self.api_key_input.text().strip(),
                "model": self.get_current_model_name()
            }
            self.llm_processor = LLMProcessor(api_config, self) # Pass MainWindow instance
            
            # 添加任务到队列
            self.work_queue.put((current, context))
            self.start_processing()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化失败: {str(e)}")

    def summarize_all(self):
        """提炼所有章节"""
        if not self.validate_config():
            return
            
        # 收集所有未提炼的章节
        root = self.chapter_tree.invisibleRootItem()
        tasks = []
        
        for i in range(root.childCount()):
            volume = root.child(i)
            for j in range(volume.childCount()):
                chapter = volume.child(j)
                if isinstance(chapter, ChapterTreeItem) and not chapter.is_summarized:
                    tasks.append((chapter, ""))
        
        if not tasks:
            QMessageBox.information(self, "提示", "没有需要提炼的章节")
            return
            
        # 初始化处理器
        try:
            api_config = {
                "url": self.api_url_input.text().strip(),
                "key": self.api_key_input.text().strip(),
                "model": self.get_current_model_name()
            }
            self.llm_processor = LLMProcessor(api_config, self) # Pass MainWindow instance
            
            # 添加所有任务
            for task in tasks:
                self.work_queue.put(task)
                
            self.progress_bar.setMaximum(len(tasks))
            self.progress_bar.setValue(0)
            self.start_processing()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化失败: {str(e)}")

    def validate_config(self):
        """验证配置"""
        if not self.api_url_input.text().strip():
            QMessageBox.warning(self, "配置错误", "请输入API地址")
            return False
            
        if not self.api_key_input.text().strip():
            QMessageBox.warning(self, "配置错误", "请输入API密钥")
            return False
            
        if not self.get_current_model_name():
            QMessageBox.warning(self, "配置错误", "请选择或输入模型名称")
            return False
            
        return True

    def start_processing(self):
        """启动处理任务"""
        if self.worker_thread and self.worker_thread.isRunning():
            return
            
        self.worker_thread = WorkerThread(self.work_queue, self.llm_processor)
        self.worker_thread.update_signal.connect(self.handle_update)
        self.worker_thread.progress_signal.connect(self.update_progress)
        self.worker_thread.error_signal.connect(self.handle_error)
        self.worker_thread.finished.connect(self.processing_finished)
        
        self.worker_thread.start()
        self.summarize_btn.setEnabled(False)
        self.summarize_all_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("处理中...")

    def stop_processing(self):
        """停止处理"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait(3000)  # 等待3秒
            
        self.processing_finished()

    def processing_finished(self):
        """处理完成"""
        self.summarize_btn.setEnabled(True)
        self.summarize_all_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("处理完成")

    def handle_update(self, update_type, data):
        """处理UI更新"""
        if update_type == "summary":
            item, summary = data
            item.summary = summary
            item.is_summarized = True
            item.summary_timestamp = time.time()
            item.update_display_text() # Update visual marker
            
            # Refresh content display for the updated item if it's the current one
            current_item = self.chapter_tree.currentItem()
            if current_item == item:
                self.show_content(item) # This will now apply the default logic

    def handle_error(self, error_msg):
        """处理错误"""
        QMessageBox.critical(self, "处理错误", error_msg)
        self.processing_finished()

    def update_progress(self, in_tokens, out_tokens, count):
        """更新进度和Token统计"""
        self.total_tokens[0] += in_tokens
        self.total_tokens[1] += out_tokens
        self.token_label.setText(
            f"Token消耗: 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}"
        )

        if count > 0:
            self.progress_bar.setValue(self.progress_bar.value() + count)

    def export_results(self):
        """导出提炼结果"""
        if not self.book_data.get("title") or self.chapter_tree.topLevelItemCount() == 0:
            QMessageBox.warning(self, "导出错误", "没有加载小说或小说内容为空，无法导出。")
            return

        if not self.default_export_path:
            reply = QMessageBox.question(
                self, '设置导出路径',
                '未设置默认导出路径，是否现在设置？',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No # Default to No
            )
            if reply == QMessageBox.Yes:
                self.set_export_path()
                if not self.default_export_path: # Check again if path was actually set
                    QMessageBox.warning(self, "导出取消", "未选择导出路径，导出操作已取消。")
                    return
            else:
                QMessageBox.information(self, "导出取消", "未设置导出路径，导出操作已取消。")
                return

        # Ensure book_data and title are available (redundant with earlier check, but safe)
        if not self.book_data or not self.book_data.get("title"):
            QMessageBox.warning(self, "错误", "没有加载小说信息，无法导出。") # Should be caught by first check
            return

        try:
            book_title = self.book_data.get("title", "未命名小说")
            book_dir = os.path.join(self.default_export_path, f"{book_title}_提炼结果")
            os.makedirs(book_dir, exist_ok=True) # Ensure directory is created before writing files

            # 导出不同格式
            self.export_txt(book_dir)
            self.export_markdown(book_dir)
            self.export_json(book_dir)

            QMessageBox.information(self, "导出完成", f"结果已保存到:\n{book_dir}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", str(e))

    def export_txt(self, path):
        """导出TXT格式"""
        file_name = os.path.join(path, f"{self.book_data.get('title', '未命名小说')}_提炼总结.txt")
        try:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(f"{self.book_data.get('title', '未命名小说')} 提炼总结\n")
                f.write("=" * 50 + "\n\n")

                book_item = self.chapter_tree.topLevelItem(0)
                if not book_item: return # Should be caught by export_results

                for i in range(book_item.childCount()): # Iterate through volumes
                    vol_item = book_item.child(i)
                    f.write(f"{vol_item.text(0)}\n") # Volume title
                    f.write("-" * 40 + "\n")

                    for j in range(vol_item.childCount()): # Iterate through chapters
                        chapter_item = vol_item.child(j)
                        if not isinstance(chapter_item, ChapterTreeItem): continue

                        title_to_write = chapter_item.original_title if hasattr(chapter_item, 'original_title') else chapter_item.text(0)

                        if chapter_item.is_summarized and chapter_item.summary:
                            f.write(f"【提炼总结】 {title_to_write}\n")
                            f.write("-" * 20 + "\n")
                            f.write(f"{chapter_item.summary}\n\n")
                        else:
                            f.write(f"【原文】 {title_to_write}\n")
                            f.write("-" * 20 + "\n")
                            f.write(f"{chapter_item.content}\n\n")
                    f.write("\n") # Extra newline after each volume's content
        except IOError as e:
            QMessageBox.critical(self, "导出错误", f"写入TXT文件失败: {file_name}\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出TXT时发生未知错误: {str(e)}")

    def export_markdown(self, path):
        """导出Markdown格式"""
        file_name = os.path.join(path, f"{self.book_data.get('title', '未命名小说')}_提炼总结.md")
        try:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(f"# {self.book_data.get('title', '未命名小说')} 提炼总结\n\n")
                f.write(f"**生成时间:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Token消耗:** 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}\n\n")
                f.write("---\n\n")

                book_item = self.chapter_tree.topLevelItem(0)
                if not book_item: return

            for i in range(book_item.childCount()): # Iterate through volumes
                vol_item = book_item.child(i)
                f.write(f"## {vol_item.text(0)}\n\n") # Volume title

                for j in range(vol_item.childCount()): # Iterate through chapters
                    chapter_item = vol_item.child(j)
                    if not isinstance(chapter_item, ChapterTreeItem): continue

                    title_to_write = chapter_item.original_title if hasattr(chapter_item, 'original_title') else chapter_item.text(0)

                    if chapter_item.is_summarized and chapter_item.summary:
                        f.write(f"### {title_to_write} (提炼后)\n\n")
                        f.write(f"{chapter_item.summary}\n\n")
                    else:
                        f.write(f"### {title_to_write} (原文)\n\n")
                        f.write(f"{chapter_item.content}\n\n")
        except IOError as e:
            QMessageBox.critical(self, "导出错误", f"写入Markdown文件失败: {file_name}\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出Markdown时发生未知错误: {str(e)}")

    def export_json(self, path):
        """导出JSON格式"""
        data = {
            "title": self.book_data.get('title', '未命名小说'),
            "export_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "token_usage": {
                "input": self.total_tokens[0],
                "output": self.total_tokens[1]
            },
            "volumes": []
        }

        book_item = self.chapter_tree.topLevelItem(0)
        if not book_item: return

        for i in range(book_item.childCount()): # Iterate through volumes
            vol_item = book_item.child(i)
            volume_data = {
                "title": vol_item.text(0), # Use volume item's text
                "chapters": []
            }

            for j in range(vol_item.childCount()): # Iterate through chapters
                chapter_item = vol_item.child(j)
                if not isinstance(chapter_item, ChapterTreeItem): continue

                title_to_write = chapter_item.original_title if hasattr(chapter_item, 'original_title') else chapter_item.text(0)
                content_to_export = ""
                status = "original"
                content_length = chapter_item.word_count

                if chapter_item.is_summarized and chapter_item.summary:
                    content_to_export = chapter_item.summary
                    status = "refined"
                    content_length = len(chapter_item.summary)
                else:
                    content_to_export = chapter_item.content
                    # status and content_length already set for original

                chapter_data = {
                    "title": title_to_write,
                    "status": status,
                    "content": content_to_export,
                    "length": content_length
                }
                volume_data["chapters"].append(chapter_data)

            if volume_data["chapters"]:
                data["volumes"].append(volume_data)

        file_name = os.path.join(path, f"{self.book_data.get('title', '未命名小说')}_数据.json")
        try:
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            QMessageBox.critical(self, "导出错误", f"写入JSON文件失败: {file_name}\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出JSON时发生未知错误: {str(e)}")


    def set_export_path(self):
        """设置默认导出路径"""
        path = QFileDialog.getExistingDirectory(self, "选择默认导出目录")
        if path:
            self.default_export_path = path
            self.status_label.setText(f"导出目录已设置为: {path}")
            QMessageBox.information(self, "设置成功", f"默认导出目录已设置为:\n{path}")
        else:
            self.status_label.setText("导出目录未更改")
            # QMessageBox.information(self, "设置取消", "未选择目录，导出路径未更改。") # Optional: can be noisy

    def add_custom_model(self):
        """添加自定义模型"""
        from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("添加自定义模型")
        dialog.setModal(True)
        
        layout = QFormLayout()
        
        name_input = QLineEdit()
        name_input.setPlaceholderText("例如: my-custom-model")
        layout.addRow("模型名称:", name_input)
        
        display_name_input = QLineEdit()
        display_name_input.setPlaceholderText("例如: 我的自定义模型")
        layout.addRow("显示名称:", display_name_input)
        
        url_input = QLineEdit()
        url_input.setPlaceholderText("https://api.example.com/v1/chat/completions")
        layout.addRow("API地址:", url_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            model_name = name_input.text().strip()
            display_name = display_name_input.text().strip()
            api_url = url_input.text().strip()
            
            if model_name and display_name and api_url:
                # 添加到配置
                self.model_configs[model_name] = {
                    "url": api_url,
                    "display_name": display_name
                }
                
                # 添加到下拉框
                self.model_combo.addItem(display_name, model_name)
                self.model_combo.setCurrentText(display_name)
                
                QMessageBox.information(self, "成功", f"已添加自定义模型: {display_name}")
            else:
                QMessageBox.warning(self, "错误", "请填写完整信息")

    def save_config(self):
        """保存配置到文件"""
        config = {
            "model": self.get_current_model_name(),
            "api_url": self.api_url_input.text(),
            "api_key": self.api_key_input.text(),
            "export_path": self.default_export_path,
            "custom_models": {k: v for k, v in self.model_configs.items() if k.startswith('custom_')}, # Save only custom models explicitly added by user
            "book_data": { # Ensure all necessary book_data fields are saved
                "title": self.book_data.get("title"),
                "file_path": self.book_data.get("file_path"),
                "encoding": self.book_data.get("encoding"),
                "volumes": self.book_data.get("volumes", []) # Retain volumes if they exist
            },
            "summary_mode": self.summary_mode_btn.isChecked(),
            "custom_prompt": self.custom_prompt_input.text(), # Get current prompt from input field
            "chapter_states": self.get_chapter_states()
        }
        
        try:
            with open("config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "成功", "配置已保存")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")

    def get_chapter_states(self):
        """获取所有章节处理状态"""
        states = []
        root = self.chapter_tree.invisibleRootItem()
        for i in range(root.childCount()):
            vol = root.child(i)
            for j in range(vol.childCount()):
                chap = vol.child(j)
                if isinstance(chap, ChapterTreeItem):
                    states.append({
                        "path": [vol.text(0), chap.text(0)],
                        "is_summarized": chap.is_summarized,
                        "summary": chap.summary,
                        "timestamp": chap.summary_timestamp,
                        "content": chap.content, # Save content
                        "word_count": chap.word_count # Save word count
                    })
        return states

    def load_config(self):
        """从文件加载配置"""
        try:
            if os.path.exists("config.json"):
                with open("config.json", 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 加载自定义模型
                if "custom_models" in config:
                    self.model_configs.update(config["custom_models"])
                    
                    # 更新下拉框
                    for model_key, model_config in config["custom_models"].items():
                        self.model_combo.addItem(model_config["display_name"], model_key)
                
                # 恢复配置
                if config.get("model"):
                    # 查找对应的显示名称
                    for i in range(self.model_combo.count()):
                        if self.model_combo.itemData(i) == config["model"]:
                            self.model_combo.setCurrentIndex(i)
                            break
                
                self.api_url_input.setText(config.get("api_url", ""))
                self.api_key_input.setText(config.get("api_key", ""))
                self.default_export_path = config.get("export_path", "")
                self.custom_prompt = config.get("custom_prompt", self.custom_prompt) # Restore custom_prompt
                self.prompt_input.setText(self.custom_prompt) # Update prompt input field

                loaded_book_data = config.get("book_data")
                if loaded_book_data and loaded_book_data.get("file_path") and loaded_book_data.get("encoding"):
                    self.book_data = loaded_book_data
                    self.reload_novel()
        
                # 恢复章节状态 (should be after reload_novel if novel is loaded)
                if "chapter_states" in config:
                    self.restore_chapter_states(config["chapter_states"])
                
                # 恢复显示模式 - This needs to be after items are potentially reloaded and states restored
                if "summary_mode" in config and isinstance(config["summary_mode"], bool):
                    self.summary_mode_btn.setChecked(config["summary_mode"])

                # Update button text based on loaded state, especially if no item is selected after load.
                # If an item IS selected (e.g. after reload_novel), show_content would have set this.
                if not self.chapter_tree.currentItem(): # Only adjust if no item is actively displayed
                    if self.summary_mode_btn.isChecked(): # Checked means "user wants original"
                        self.summary_mode_btn.setText("显示要点")
                    else: # Not checked means "user wants summary" (or default)
                        self.summary_mode_btn.setText("显示原文")

                # Update status bar or show message only if not reloading novel (to avoid double messages)
                if not (loaded_book_data and loaded_book_data.get("file_path")):
                    QMessageBox.information(self, "成功", "配置已加载")
            else:
                QMessageBox.information(self, "提示", "未找到配置文件")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载配置失败: {str(e)}")

    def reload_novel(self):
        """重新加载上次打开的小说"""
        file_path = self.book_data.get("file_path")
        encoding = self.book_data.get("encoding")

        if not file_path or not encoding:
            print("重新加载小说失败: 文件路径或编码未在book_data中设置")
            return

        self.status_label.setText(f"重新加载: {self.book_data.get('title', '未知标题')}...")
        QApplication.processEvents()
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            # Parse chapters and rebuild tree
            chapters = self.parse_chapters(content)
            self.build_chapter_tree(chapters) # This will set the book title in the tree

            self.status_label.setText(f"已重新加载: {self.book_data.get('title', '未知标题')}")
            # QMessageBox.information(self, "小说已重新加载", f"{self.book_data.get('title', '未知标题')} 已成功加载。")

        except FileNotFoundError:
            QMessageBox.critical(self, "错误", f"重新加载小说失败: 文件未找到 {file_path}")
            self.status_label.setText("重新加载失败: 文件未找到")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重新加载小说失败: {str(e)}")
            self.status_label.setText(f"重新加载失败: {str(e)}")


    def restore_chapter_states(self, states):
        """恢复章节处理状态"""
        root = self.chapter_tree.invisibleRootItem()
        for state in states:
            vol_title, chap_title = state["path"]
            for i in range(root.childCount()):
                vol = root.child(i)
                if vol.text(0) == vol_title:
                    for j in range(vol.childCount()):
                        chap = vol.child(j)
                        if chap.text(0) == chap_title and isinstance(chap, ChapterTreeItem):
                            chap.is_summarized = state.get("is_summarized", False)
                            chap.summary = state.get("summary", "")
                            chap.summary_timestamp = state.get("timestamp", 0)
                            # Restore content and word count
                            if "content" in state: # Check for backward compatibility
                                chap.content = state["content"]
                            if "word_count" in state:
                                chap.word_count = state["word_count"]
                                chap.setText(1, f"{chap.word_count}字") # Update tree display for word count
                            else: # Recalculate if not found (older config)
                                chap.word_count = len(chap.content)
                                chap.setText(1, f"{chap.word_count}字")
                            chap.update_display_text() # Add marker if loaded state is summarized

    def update_prompt(self):
        """更新自定义提示词"""
        self.custom_prompt = self.prompt_input.text()

    def test_connection(self):
        """测试API连接"""
        if not self.validate_config():
            return
            
        try:
            api_config = {
                "url": self.api_url_input.text().strip(),
                "key": self.api_key_input.text().strip(),
                "model": self.get_current_model_name()
            }
            
            # 创建处理器并测试
            # For test_connection, we might not have a fully initialized LLMProcessor instance member yet,
            # so create a local one and pass 'self' (MainWindow) for custom_prompt access.
            processor = LLMProcessor(api_config, self) # Pass MainWindow instance
            test_text = "这是一个连接测试。"
            
            self.status_label.setText("正在测试连接...")
            QApplication.processEvents()
            
            # 计算token
            token_count = processor.calculate_tokens(test_text)
            
            # 调用API测试
            summary, in_tokens, out_tokens = processor.summarize(test_text, max_retries=1)
            
            if summary:
                QMessageBox.information(
                    self,
                    "连接成功",
                    f"API连接测试成功！\n\n"
                    f"模型: {api_config['model']}\n"
                    f"API地址: {api_config['url']}\n"
                    f"测试文本Token: {token_count}\n"
                    f"输入Token: {in_tokens}\n"
                    f"输出Token: {out_tokens}\n\n"
                    f"返回内容: {summary[:100]}..."
                )
                self.status_label.setText("连接测试成功")
            else:
                raise ValueError("API返回内容为空")
                
        except Exception as e:
            error_msg = str(e)
            
            # 提供更详细的错误诊断
            suggestions = []
            if "model" in error_msg.lower():
                suggestions.append("• 检查模型名称是否正确")
            if "api_key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                suggestions.append("• 检查API密钥是否有效")
            if "timeout" in error_msg.lower():
                suggestions.append("• 检查网络连接")
            if "url" in error_msg.lower() or "connection" in error_msg.lower():
                suggestions.append("• 检查API地址是否正确")
                
            suggestion_text = "\n".join(suggestions) if suggestions else "• 检查网络连接和配置信息"
            
            QMessageBox.critical(
                self,
                "连接失败",
                f"API连接测试失败:\n\n{error_msg}\n\n建议:\n{suggestion_text}"
            )
            self.status_label.setText("连接测试失败")

    def auto_save(self):
        """自动保存配置"""
        if self.book_data.get("file_path"): # Only save if a book is loaded
            try:
                # Ensure not in editing mode before auto-saving
                if not self.content_display.isReadOnly():
                    print("自动保存已跳过：章节正在编辑中。") # Auto-save skipped: chapter editing in progress.
                    return
                self.save_config()
                # self.status_label.setText(f"自动保存于 {time.strftime('%H:%M:%S')}") # Optional: feedback for auto-save
            except Exception as e:
                print(f"自动保存失败: {str(e)}")

    def enable_chapter_editing(self):
        """Enables editing for the currently selected chapter's original content."""
        current_item = self.chapter_tree.currentItem()
        if not isinstance(current_item, ChapterTreeItem):
            QMessageBox.warning(self, "无法编辑", "请先选择一个章节进行编辑。")
            return

        # Check if summary is displayed (and item is summarized)
        # summary_mode_btn isChecked means "user wants original" (text is "Show Summary")
        # summary_mode_btn not isChecked means "user wants summary" (text is "Show Original")
        if current_item.is_summarized and not self.summary_mode_btn.isChecked():
             QMessageBox.warning(self, "编辑冲突", "正在显示摘要，请切换到原文以进行编辑。")
             return

        self.content_display.setReadOnly(False)
        # Ensure original content is loaded for editing, even if summary was somehow displayed
        self.content_display.setText(current_item.content)

        self.edit_chapter_btn.setEnabled(False)
        self.save_chapter_btn.setEnabled(True)

        # Disable other actions that might interfere
        self.summarize_btn.setEnabled(False)
        self.summarize_all_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.load_btn.setEnabled(False) # Disable load button during editing
        self.summary_mode_btn.setEnabled(False) # Disable mode toggle during edit

        self.status_label.setText("章节编辑模式...")


    def save_chapter_edits(self):
        """Saves the edited content of the current chapter."""
        current_item = self.chapter_tree.currentItem()
        if not isinstance(current_item, ChapterTreeItem):
            # This case should ideally not be reached if button states are managed well
            QMessageBox.critical(self, "错误", "没有选中有效章节以保存。")
            return

        current_item.content = self.content_display.toPlainText()
        current_item.word_count = len(current_item.content)
        current_item.setText(1, f"{current_item.word_count}字") # Update word count in tree

        # If the chapter was summarized, its summary is now potentially outdated.
        # We could clear the summary, or mark it as needing re-summary, or leave it.
        # For now, let's consider the summary stale. We'll clear the visual marker.
        # The user would need to re-summarize.
        if current_item.is_summarized:
            # current_item.is_summarized = False # Option 1: Mark as not summarized
            # current_item.summary = "" # Option 2: Clear summary
            # For now, just update display text if it had marker, user can re-summarize
            pass # The existing summary remains, user responsibility to re-summarize if needed

        self.content_display.setReadOnly(True)
        self.edit_chapter_btn.setEnabled(True)
        self.save_chapter_btn.setEnabled(False)

        # Re-enable other actions
        self.summarize_btn.setEnabled(True)
        self.summarize_all_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        if current_item.is_summarized : # Only re-enable toggle if there is a summary
             self.summary_mode_btn.setEnabled(True)


        self.status_label.setText("章节修改已保存。")

        # Explicitly call auto_save to persist the change to chapter content
        # This is important because get_chapter_states now includes content
        self.auto_save()


    def closeEvent(self, event):
        """程序关闭事件"""
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self, '确认退出', 
                '正在处理任务，确定要退出吗？',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker_thread.stop()
                self.worker_thread.wait(3000)
            self.save_config() # Save config before exiting
            event.accept()
        else:
            self.save_config() # Save config before exiting
            event.accept()


if __name__ == "__main__":
    import sys
    
    app = QApplication(sys.argv)
    app.setApplicationName("小说智能分析工具")
    app.setApplicationVersion("2.0")
    
    window = MainWindow()
    window.load_config() # Load config on startup
    window.show()
    
    sys.exit(app.exec_())