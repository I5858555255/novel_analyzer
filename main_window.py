# main_window.py
import os
import re
import json
import threading # Ensure threading is imported
import queue
import time
import copy
import tiktoken

from PyQt5.QtWidgets import (QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem,
                             QTextEdit, QFileDialog, QPushButton, QComboBox,
                             QProgressBar, QLabel, QSplitter, QVBoxLayout,
                             QHBoxLayout, QWidget, QAction, QMessageBox, QLineEdit,
                             QHeaderView, QSpinBox)
from PyQt5.QtCore import pyqtSignal, Qt, QThread, QTimer, QThreadPool

import requests

# Project module imports
from constants import DEFAULT_MODEL_CONFIGS, DEFAULT_CUSTOM_PROMPT
from llm_processor import LLMProcessor
from custom_widgets import ChapterTreeItem
from threading_utils import SummarizationSignals, SummarizationTask, WorkerThread
from dialogs import ManageModelsDialog

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
            if cpu_cores:
                pass
        except Exception:
            pass
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

        self.batch_content_store = {} # Added for on-demand content fetching
        self.content_store_lock = threading.Lock() # Added

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
        print(f"DEBUG: Entering MainWindow.build_chapter_tree (processing {len(chapters)} volumes/groups)")
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
            print("DEBUG: Exiting MainWindow.build_chapter_tree")

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
        print("DEBUG: summarize_selected called")
        current = self.chapter_tree.currentItem()
        if not current or not isinstance(current, ChapterTreeItem):
            QMessageBox.warning(self, "提示", "请先选择要提炼的章节")
            return

        if not self.validate_config():
            print("DEBUG: summarize_selected - validate_config failed")
            return

        context = ""
        parent = current.parent()
        if parent and isinstance(parent, ChapterTreeItem) and parent.is_summarized and parent.summary:
            context = parent.summary
        elif parent and not isinstance(parent, ChapterTreeItem) and parent.text(0) != self.book_data.get("title"):
             pass

        try:
            api_config = {
                "url": self.api_url_input.text().strip(),
                "key": self.api_key_input.text().strip(),
                "model": self.get_current_model_name()
            }
            print(f"DEBUG: summarize_selected - API Config: {api_config}")

            encoding_object = self.get_tiktoken_encoding(api_config['model'])
            print(f"DEBUG: summarize_selected - Encoding Object: {type(encoding_object)}")

            if encoding_object is None:
                QMessageBox.critical(self, "编码器错误", f"无法为模型 '{api_config.get('model','未知')}' 初始化Token编码器。提炼中止。")
                self.status_label.setText(f"错误: 模型 '{api_config.get('model','未知')}' Token编码器初始化失败。")
                print("DEBUG: summarize_selected - encoding_object is None, aborting.")
                return

            print("DEBUG: summarize_selected - Instantiating LLMProcessor for WorkerThread")
            current_llm_processor_instance = LLMProcessor(api_config, self.custom_prompt, encoding_object)
            print("DEBUG: summarize_selected - LLMProcessor for WorkerThread instantiated")

            if self.worker_thread and self.worker_thread.isRunning():
                 QMessageBox.warning(self, "提示", "已有单章提炼任务在进行中。请等待其完成。")
                 print("DEBUG: summarize_selected - WorkerThread already running.")
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

            print(f"DEBUG: summarize_selected - Starting WorkerThread for chapter: {current.original_title}")
            self.progress_bar.setMaximum(1)
            self.progress_bar.setValue(0)
            self.worker_thread.start()

        except Exception as e:
            error_msg = f"启动单章提炼失败: {str(e)}"
            print(f"DEBUG: summarize_selected - EXCEPTION: {error_msg}")
            QMessageBox.critical(self, "错误", error_msg)
            self.status_label.setText("单章提炼启动失败。")
            self.summarize_btn.setEnabled(True)
            self.summarize_all_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.batch_size_spinbox.setEnabled(True)

    def prepare_content_store(self, all_task_structs):
        print(f"DEBUG: MainWindow.prepare_content_store - Preparing content for {len(all_task_structs)} tasks.")
        with self.content_store_lock:
            self.batch_content_store.clear()
            for task_data in all_task_structs:
                self.batch_content_store[task_data["identifier"]] = task_data["content"]
        print(f"DEBUG: MainWindow.prepare_content_store - Content store populated with {len(self.batch_content_store)} items.")

    def get_content_for_task(self, identifier):
        with self.content_store_lock:
            content = self.batch_content_store.get(identifier)
        return content

    def clear_content_for_task(self, identifier):
        with self.content_store_lock:
            self.batch_content_store.pop(identifier, None)

    def summarize_all(self):
        print("DEBUG: summarize_all called")
        if not self.book_data.get("title") or self.chapter_tree.topLevelItemCount() == 0:
            QMessageBox.information(self, "提示", "请先导入小说。")
            print("DEBUG: summarize_all returned - no book loaded")
            return

        if not self.validate_config():
            print("DEBUG: summarize_all returned - validate_config failed")
            return

        all_tasks_data_list = []
        book_item = self.chapter_tree.topLevelItem(0)
        if not book_item:
            QMessageBox.information(self, "提示", "书籍未正确加载到章节树。")
            return

        for i in range(book_item.childCount()):
            vol_item = book_item.child(i)
            for j in range(vol_item.childCount()):
                chapter_item = vol_item.child(j)
                if isinstance(chapter_item, ChapterTreeItem) and not chapter_item.is_summarized:
                    identifier = (vol_item.text(0), chapter_item.original_title)
                    content = chapter_item.content
                    current_context = ""
                    all_tasks_data_list.append({
                        "identifier": identifier, "content": content, "context": current_context
                    })

        if not all_tasks_data_list:
            QMessageBox.information(self, "提示", "所有章节均已提炼或没有章节可供提炼。")
            print("DEBUG: summarize_all returned - no tasks to run")
            return

        self.prepare_content_store(all_tasks_data_list) # Populate content store

        # self.pending_batch_tasks_data will now store dicts without 'content'
        # as content is fetched on demand by SummarizationTask
        self.pending_batch_tasks_data = [
            {"identifier": td["identifier"], "context": td["context"]} for td in all_tasks_data_list
        ]
        self.current_batch_chunk_offset = 0

        self.start_batch_processing(len(self.pending_batch_tasks_data))
        self.process_next_batch_chunk()

    def process_next_batch_chunk(self):
        print(f"DEBUG: process_next_batch_chunk called. Offset: {self.current_batch_chunk_offset}, Stop: {self.stop_batch_requested}")
        if self.stop_batch_requested:
            if self.active_batch_tasks == 0 :
                 self.processing_finished(stopped_by_user=True)
            return

        current_chapters_per_batch = self.batch_size_spinbox.value()
        if current_chapters_per_batch <= 0:
            current_chapters_per_batch = 10

        start_index = self.current_batch_chunk_offset
        end_index = start_index + current_chapters_per_batch
        current_chunk_tasks_data = self.pending_batch_tasks_data[start_index:end_index]

        if not current_chunk_tasks_data:
            if self.active_batch_tasks == 0:
                print("DEBUG: process_next_batch_chunk - No more tasks data and no active tasks. Finishing.")
                self.processing_finished(stopped_by_user=self.stop_batch_requested)
            else:
                print("DEBUG: process_next_batch_chunk - No more tasks data, but waiting for active tasks to complete.")
            return

        self.active_batch_tasks = len(current_chunk_tasks_data)
        print(f"DEBUG: process_next_batch_chunk - Processing chunk: {len(current_chunk_tasks_data)} tasks. Active tasks set to: {self.active_batch_tasks}")

        try:
            api_config = {
                "url": self.api_url_input.text().strip(),
                "key": self.api_key_input.text().strip(),
                "model": self.get_current_model_name()
            }
            encoding_object = self.get_tiktoken_encoding(api_config['model'])
            if encoding_object is None:
                QMessageBox.critical(self, "编码器错误", f"无法为模型 {api_config['model']} 初始化Token编码器。批量处理中止。")
                self.processing_finished(stopped_by_user=True)
                return
        except Exception as e:
            QMessageBox.critical(self, "配置错误", f"API配置或编码器初始化无效: {str(e)}")
            self.processing_finished(stopped_by_user=True)
            return

        for task_data in current_chunk_tasks_data: # task_data here does not have 'content'
            if self.stop_batch_requested:
                print(f"DEBUG: process_next_batch_chunk - Stop requested, breaking task creation for current chunk.")
                break

            runnable_task = SummarizationTask(
                chapter_item_identifier=task_data["identifier"],
                # chapter_content is no longer passed here
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
            print(f"DEBUG: MainWindow.process_next_batch_chunk - About to start task in thread_pool for: {task_data['identifier']}")
            self.thread_pool.start(runnable_task)
            print(f"DEBUG: MainWindow.process_next_batch_chunk - Task for {task_data['identifier']} SUBMITTED to thread_pool.")

        self.current_batch_chunk_offset = end_index
        print(f"DEBUG: process_next_batch_chunk - Next offset: {self.current_batch_chunk_offset}")

    def validate_config(self):
        print("DEBUG: validate_config called")
        if not self.api_url_input.text().strip():
            QMessageBox.warning(self, "配置错误", "请输入API地址"); print("DEBUG: validate_config failed - no API URL"); return False
        if not self.api_key_input.text().strip():
            QMessageBox.warning(self, "配置错误", "请输入API密钥"); print("DEBUG: validate_config failed - no API Key"); return False
        model_name = self.get_current_model_name()
        print(f"DEBUG: validate_config - model name: {model_name}")
        if not model_name:
            QMessageBox.warning(self, "配置错误", "请选择或输入模型名称"); print("DEBUG: validate_config failed - no model name"); return False
        print("DEBUG: validate_config succeeded")
        return True

    def start_processing(self):
        pass

    def stop_processing(self):
        print("DEBUG: stop_processing called by user.")
        self.stop_batch_requested = True

        if self.worker_thread and self.worker_thread.isRunning():
            print("DEBUG: stop_processing - Stopping single WorkerThread")
            self.worker_thread.stop()

        self.status_label.setText("停止请求已发送... 等待当前活动任务完成或中止。")
        self.stop_btn.setEnabled(False)

        if self.active_batch_tasks == 0 and not (self.worker_thread and self.worker_thread.isRunning()):
             print("DEBUG: stop_processing - No active tasks found. Calling processing_finished.")
             self.processing_finished(stopped_by_user=True)

    def processing_finished(self, stopped_by_user=False):
        print(f"DEBUG: processing_finished (batch) called. Stopped by user: {stopped_by_user}")

        with self.content_store_lock: # Clear content store on batch finish/stop
            self.batch_content_store.clear()
        print("DEBUG: MainWindow.processing_finished - Batch content store cleared.")

        self.summarize_btn.setEnabled(True)
        self.summarize_all_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if stopped_by_user:
            self.status_label.setText(f"批量处理已由用户停止。已处理 {self.chapters_processed_count}/{self.chapters_to_process_total} 章节。")
        elif self.chapters_processed_count == self.chapters_to_process_total and self.chapters_to_process_total > 0:
            self.status_label.setText(f"批量提炼完成。共处理 {self.chapters_processed_count} 章节。")
        elif self.chapters_to_process_total == 0 :
             self.status_label.setText("就绪")
        else:
            self.status_label.setText(f"批量处理结束。已处理 {self.chapters_processed_count}/{self.chapters_to_process_total} 章节。")

        self.progress_bar.setValue(0)
        self.eta_label.setText("预计剩余时间: N/A")
        self.metrics_label.setText("速率: N/A")

        self.active_batch_tasks = 0
        self.chapters_to_process_total = 0
        self.chapters_processed_count = 0
        self.total_input_tokens_batch = 0
        self.total_output_tokens_batch = 0
        self.batch_start_time = 0
        self.average_time_per_chapter = 0
        self.stop_batch_requested = False
        self.pending_batch_tasks_data = []
        self.current_batch_chunk_offset = 0

        self.chapter_tree.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.api_key_input.setEnabled(True)
        self.api_url_input.setEnabled(True)
        self.test_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.save_config_btn.setEnabled(True)
        self.load_config_btn.setEnabled(True)
        self.prompt_input.setEnabled(True)
        self.batch_size_spinbox.setEnabled(True)
        if hasattr(self, 'add_model_action') and self.add_model_action:
            self.add_model_action.setEnabled(True)
        if hasattr(self, 'manage_models_action') and self.manage_models_action:
            self.manage_models_action.setEnabled(True)
        print("DEBUG: processing_finished (batch) - UI re-enabled.")

    def processing_finished_single_task(self):
        print("DEBUG: processing_finished_single_task called")
        self.summarize_btn.setEnabled(True)
        self.summarize_all_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.batch_size_spinbox.setEnabled(True)

        current_item = self.chapter_tree.currentItem()
        if current_item and isinstance(current_item, ChapterTreeItem) and current_item.is_summarized:
            self.status_label.setText(f"章节 '{current_item.original_title}' 处理完成。")
        elif current_item and isinstance(current_item, ChapterTreeItem):
            self.status_label.setText(f"章节 '{current_item.original_title}' 处理结束。")
        else:
            self.status_label.setText("单章处理完成。")

        if self.progress_bar.maximum() == 1:
            self.progress_bar.setValue(1)
        print("DEBUG: processing_finished_single_task - UI re-enabled.")

    def handle_update(self, update_type, data):
        if update_type == "summary":
            item, summary_text = data
            item.summary = summary_text
            item.is_summarized = True
            item.summary_timestamp = time.time()
            item.update_display_text()
            current_tree_item = self.chapter_tree.currentItem()
            if current_tree_item == item and not self.summary_mode_btn.isChecked():
                self.content_display.setText(summary_text)
            if item.is_summarized:
                self.auto_export_novel_data()

    def handle_chapter_summary_update(self, identifier, summary_text):
        item = self.find_chapter_item_by_identifier(identifier)
        if item:
            item.summary = summary_text
            item.is_summarized = True
            item.summary_timestamp = time.time()
            item.update_display_text()
            current_selected_item = self.chapter_tree.currentItem()
            if current_selected_item == item and not self.summary_mode_btn.isChecked():
                self.content_display.setText(summary_text)
            if item.is_summarized:
                self.auto_export_novel_data()

    def update_batch_progress(self, in_tokens, out_tokens, chapters_done_this_task):
        self.total_tokens[0] += in_tokens
        self.total_tokens[1] += out_tokens

        self.total_input_tokens_batch += in_tokens
        self.total_output_tokens_batch += out_tokens

        if self.chapters_to_process_total > 0 and self.chapters_processed_count > 0:
            if self.batch_start_time == 0:
                print("Warning: batch_start_time is 0 in update_batch_progress. ETA/metrics might be incorrect.")
                return

            elapsed_time = time.time() - self.batch_start_time
            if elapsed_time <= 0:
                elapsed_time = 1e-6

            self.average_time_per_chapter = elapsed_time / self.chapters_processed_count
            remaining_chapters = self.chapters_to_process_total - self.chapters_processed_count

            if remaining_chapters > 0:
                eta_seconds = remaining_chapters * self.average_time_per_chapter
                self.eta_label.setText(f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}")
            else:
                self.eta_label.setText("ETA: 完成")

            chapters_per_minute = self.chapters_processed_count / (elapsed_time / 60)
            tokens_per_second = (self.total_input_tokens_batch + self.total_output_tokens_batch) / elapsed_time
            self.metrics_label.setText(f"{chapters_per_minute:.2f} 章/分钟 | {tokens_per_second:.2f} Token/秒")

        self.token_label.setText(f"总消耗: 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}")

    def handle_chapter_error(self, identifier, error_msg):
        if isinstance(identifier, tuple) and len(identifier) == 2:
            vol_title, chap_original_title = identifier
            item = self.find_chapter_item_by_identifier(identifier)
            chapter_display_title = item.original_title if item and hasattr(item, 'original_title') else chap_original_title
            id_str = f'{vol_title}/{chapter_display_title}'
        else:
            id_str = str(identifier)
        full_error_message = f"错误: 章节 '{id_str}' 处理失败: {error_msg}"
        print(f"ERROR_BATCH_CHAPTER: {full_error_message}")
        self.status_label.setText(f"错误: 章节 '{id_str}' - {error_msg[:100]}...")

    def handle_task_finished(self, identifier):
        self.active_batch_tasks -= 1
        self.chapters_processed_count +=1
        print(f"DEBUG: Task finished for {identifier}. Active tasks remaining in chunk: {self.active_batch_tasks}. Total processed: {self.chapters_processed_count}/{self.chapters_to_process_total}")

        if self.chapters_to_process_total > 0:
            self.progress_bar.setValue(self.chapters_processed_count)

            if self.chapters_processed_count > 0 and self.batch_start_time > 0:
                elapsed_time = time.time() - self.batch_start_time
                if elapsed_time <= 0: elapsed_time = 1e-6
                self.average_time_per_chapter = elapsed_time / self.chapters_processed_count
                remaining_chapters = self.chapters_to_process_total - self.chapters_processed_count
                if remaining_chapters > 0:
                    eta_seconds = remaining_chapters * self.average_time_per_chapter
                    self.eta_label.setText(f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}")
                else:
                    self.eta_label.setText("ETA: 完成")

        if self.active_batch_tasks == 0:
            if self.chapters_processed_count < self.chapters_to_process_total and not self.stop_batch_requested:
                print(f"DEBUG: Chunk finished. Processed: {self.chapters_processed_count}/{self.chapters_to_process_total}. Starting next chunk.")
                self.process_next_batch_chunk()
            else:
                print(f"DEBUG: All tasks finished or stop requested. Processed: {self.chapters_processed_count}/{self.chapters_to_process_total}.")
                self.total_tokens[0] += self.total_input_tokens_batch
                self.total_tokens[1] += self.total_output_tokens_batch
                self.token_label.setText(f"Token消耗: 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}")
                self.processing_finished(stopped_by_user=self.stop_batch_requested)

    def find_chapter_item_by_identifier(self, identifier):
        if not isinstance(identifier, tuple) or len(identifier) != 2:
            print(f"Warning: Invalid identifier format received by find_chapter_item_by_identifier: {identifier}")
            return None
        target_vol_title, target_chap_original_title = identifier
        if self.chapter_tree.topLevelItemCount() == 0:
            return None
        book_item = self.chapter_tree.topLevelItem(0)
        if not book_item:
            return None
        for i in range(book_item.childCount()):
            vol_item = book_item.child(i)
            if vol_item.text(0) == target_vol_title:
                for j in range(vol_item.childCount()):
                    chapter_item = vol_item.child(j)
                    if isinstance(chap_item, ChapterTreeItem) and \
                       hasattr(chap_item, 'original_title') and \
                       chap_item.original_title == target_chap_original_title:
                        return chapter_item
                # If the volume was found, but the chapter was not, return None.
                # This means the inner loop completed without returning a chapter_item.
                return None
        # If the volume itself was not found after checking all volumes, return None.
        return None

    def handle_error(self, error_msg):
        full_error_message = f"处理错误 (单个章节): {error_msg}"
        print(f"ERROR_SINGLE_CHAPTER: {full_error_message}")
        self.status_label.setText(f"处理错误: {error_msg[:150]}...")

    def start_batch_processing(self, total_overall_tasks):
        print(f"DEBUG: start_batch_processing called with total_overall_tasks: {total_overall_tasks}")

        self.chapters_to_process_total = total_overall_tasks
        self.chapters_processed_count = 0
        self.total_input_tokens_batch = 0
        self.total_output_tokens_batch = 0
        self.batch_start_time = time.time()
        self.average_time_per_chapter = 0
        self.active_batch_tasks = 0
        self.stop_batch_requested = False
        # self.pending_batch_tasks_data is set by summarize_all before calling this
        # self.current_batch_chunk_offset is set by summarize_all before calling this

        self.progress_bar.setMaximum(total_overall_tasks if total_overall_tasks > 0 else 100)
        self.progress_bar.setValue(0)
        self.eta_label.setText("预计剩余时间: 计算中...")
        self.metrics_label.setText("速率: 计算中...")
        self.status_label.setText(f"批量处理中... {total_overall_tasks}章节待处理")

        self.chapter_tree.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.api_key_input.setEnabled(False)
        self.api_url_input.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.save_config_btn.setEnabled(False)
        self.load_config_btn.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self.batch_size_spinbox.setEnabled(False)

        if hasattr(self, 'add_model_action') and self.add_model_action:
            self.add_model_action.setEnabled(False)
        if hasattr(self, 'manage_models_action') and self.manage_models_action:
            self.manage_models_action.setEnabled(False)

        self.summarize_btn.setEnabled(False)
        self.summarize_all_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        print("DEBUG: start_batch_processing finished UI disabling")

    def update_progress(self, in_tokens, out_tokens, count):
        self.total_tokens[0] += in_tokens
        self.total_tokens[1] += out_tokens
        self.token_label.setText(
            f"Token消耗: 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}"
        )
        if self.progress_bar.maximum() == 1:
             self.progress_bar.setValue(self.progress_bar.value() + count)

    def export_results(self):
        if not self.default_export_path:
            reply = QMessageBox.question(self, '导出路径', '未设置默认导出路径，是否现在设置？', QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes: self.set_export_path()
            else: return
        try:
            book_dir = os.path.join(self.default_export_path, f"{self.book_data['title']}_提炼结果")
            os.makedirs(book_dir, exist_ok=True)
            self.export_txt(book_dir)
            self.export_markdown(book_dir)
            self.export_json(book_dir)
            QMessageBox.information(self, "导出完成", f"结果已保存到:\n{book_dir}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", str(e))

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
                        if chap_item.is_summarized:
                            f.write(f"(提炼后 - {time.strftime('%Y-%m-%d %H:%M', time.localtime(chap_item.summary_timestamp))})\n")
                            f.write(f"{chap_item.summary}\n\n")
                        else:
                            f.write("(原文)\n")
                            f.write(f"{chap_item.content}\n\n")
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
                        if chap_item.is_summarized:
                            f.write(f"_(提炼后 - {time.strftime('%Y-%m-%d %H:%M', time.localtime(chap_item.summary_timestamp))})_\n\n")
                            f.write(f"{chap_item.summary}\n\n")
                        else:
                            f.write("_(原文)_\n\n")
                            f.write(f"{chap_item.content}\n\n")
                f.write("---\n\n")

    def export_json(self, output_directory_path):
        data = {"title": self.book_data["title"], "export_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                "token_usage": {"input": self.total_tokens[0], "output": self.total_tokens[1]}, "volumes": []}
        if self.chapter_tree.topLevelItemCount() == 0: return
        book_item = self.chapter_tree.topLevelItem(0)
        if not book_item: return

        for i in range(book_item.childCount()):
            vol_item = book_item.child(i)
            volume_data = {"title": vol_item.text(0), "chapters": []}
            for j in range(vol_item.childCount()):
                chap_item = vol_item.child(j)
                if isinstance(chap_item, ChapterTreeItem):
                    chapter_data = {
                        "title": chap_item.original_title,
                        "original_length": chap_item.word_count,
                        "is_summarized": chap_item.is_summarized,
                        "summary": chap_item.summary if chap_item.is_summarized else "",
                        "summary_length": len(chap_item.summary) if chap_item.is_summarized else 0,
                        "summary_timestamp": chap_item.summary_timestamp if chap_item.is_summarized else 0,
                        "content": chap_item.content
                    }
                    volume_data["chapters"].append(chapter_data)
            if volume_data["chapters"]: data["volumes"].append(volume_data)
        with open(os.path.join(output_directory_path, f"{self.book_data['title']}_数据.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def set_export_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择默认导出目录")
        if path:
            self.default_export_path = path
            self.status_label.setText(f"导出目录设置为: {path}")

    def add_custom_model(self):
        from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
        dialog = QDialog(self)
        dialog.setWindowTitle("添加自定义模型")
        dialog.setModal(True)
        layout = QFormLayout()
        name_input = QLineEdit(); name_input.setPlaceholderText("例如: my-custom-model (唯一标识)")
        layout.addRow("模型ID (唯一):", name_input)
        display_name_input = QLineEdit(); display_name_input.setPlaceholderText("例如: 我的自定义模型")
        layout.addRow("显示名称:", display_name_input)
        url_input = QLineEdit(); url_input.setPlaceholderText("https://api.example.com/v1/chat/completions")
        layout.addRow("API地址:", url_input)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        dialog.setLayout(layout)
        if dialog.exec_() == QDialog.Accepted:
            model_id = name_input.text().strip()
            display_name = display_name_input.text().strip()
            api_url = url_input.text().strip()
            if model_id and display_name and api_url:
                if model_id in self.model_configs:
                    QMessageBox.warning(self, "错误", f"模型ID '{model_id}' 已存在。"); return
                self.model_configs[model_id] = {"url": api_url, "display_name": display_name}
                self.model_combo.addItem(display_name, model_id)
                self.model_combo.setCurrentText(display_name)
                QMessageBox.information(self, "成功", f"已添加自定义模型: {display_name}")
            else:
                QMessageBox.warning(self, "错误", "请填写模型ID、显示名称和API地址。")

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
            else: self.status_label.setText("配置已自动保存")
        except Exception as e:
            if not silent: QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")
            else: print(f"Error during silent save_config: {e}")

    def get_chapter_states(self):
        states = []
        root = self.chapter_tree.invisibleRootItem()
        if root.childCount() == 0: return states
        book_item = root.child(0)

        for i in range(book_item.childCount()):
            vol_item = book_item.child(i)
            for j in range(vol_item.childCount()):
                chap_item = vol_item.child(j)
                if isinstance(chap_item, ChapterTreeItem):
                    states.append({
                        "path": [vol_item.text(0), chap_item.original_title if hasattr(chap_item, 'original_title') else chap_item.text(0)],
                        "is_summarized": chap_item.is_summarized, "summary": chap_item.summary,
                        "timestamp": chap_item.summary_timestamp, "content": chap_item.content,
                        "word_count": chap_item.word_count
                    })
        return states

    def load_config(self, silent=False):
        config_path = "config.json"
        default_batch_size = 10
        if not os.path.exists(config_path):
            if not silent: QMessageBox.information(self, "提示", "未找到配置文件。")
            else: print("Config file not found on startup. Skipping load.")
            self.batch_size_spinbox.setValue(default_batch_size)
            return
        try:
            with open(config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)

            if "custom_models" in config_data:
                for model_key, model_data_val in config_data["custom_models"].items():
                    if model_key not in self.initial_model_keys:
                         self.model_configs[model_key] = model_data_val
                         if self.model_combo.findData(model_key) == -1:
                              self.model_combo.addItem(model_data_val.get("display_name", model_key), model_key)

            if config_data.get("model"):
                model_to_select = config_data["model"]
                index = self.model_combo.findData(model_to_select)
                if index == -1:
                    index = self.model_combo.findText(model_to_select)
                if index != -1: self.model_combo.setCurrentIndex(index)

            self.api_url_input.setText(config_data.get("api_url", ""))
            self.api_key_input.setText(config_data.get("api_key", ""))
            self.default_export_path = config_data.get("export_path", "")
            self.custom_prompt = config_data.get("custom_prompt", DEFAULT_CUSTOM_PROMPT)
            self.prompt_input.setText(self.custom_prompt)

            loaded_batch_size_val = config_data.get("batch_size", default_batch_size)
            try:
                loaded_batch_size_int = int(loaded_batch_size_val)
                if not (self.batch_size_spinbox.minimum() <= loaded_batch_size_int <= self.batch_size_spinbox.maximum()):
                    loaded_batch_size_int = default_batch_size
                    print(f"Warning: Loaded batch_size '{config_data.get('batch_size')}' out of range, reset to {loaded_batch_size_int}.")
            except (ValueError, TypeError):
                loaded_batch_size_int = default_batch_size
                print(f"Warning: Invalid batch_size '{config_data.get('batch_size')}' in config, reset to {loaded_batch_size_int}.")
            self.batch_size_spinbox.setValue(loaded_batch_size_int)

            reloaded_successfully = False
            if "book_data" in config_data and config_data["book_data"].get("file_path"):
                self.book_data = config_data["book_data"]
                reloaded_successfully = self.reload_novel()

            if reloaded_successfully and "chapter_states" in config_data:
                self.restore_chapter_states(config_data["chapter_states"])

            if "summary_mode" in config_data:
                self.summary_mode_btn.setChecked(config_data["summary_mode"])
                self.toggle_display_mode()

            if not silent: QMessageBox.information(self, "成功", "配置已加载")
            else:
                status_msg = "配置已加载"
                if reloaded_successfully: status_msg += f", 上次打开: {self.book_data.get('title','未知标题')}"
                self.status_label.setText(status_msg)
                print("Config loaded silently.")
        except json.JSONDecodeError as e:
            if not silent: QMessageBox.critical(self, "错误", f"加载配置失败: 配置文件格式错误。\n{str(e)}")
            else: print(f"Error decoding config.json: {e}")
            self.batch_size_spinbox.setValue(default_batch_size)
        except Exception as e:
            if not silent: QMessageBox.critical(self, "错误", f"加载配置失败: {str(e)}")
            else: print(f"Error during silent load_config: {e}")
            self.batch_size_spinbox.setValue(default_batch_size)

    def reload_novel(self):
        file_path = self.book_data.get("file_path")
        encoding = self.book_data.get("encoding")
        if not file_path or not os.path.exists(file_path) or not encoding:
            msg = f"配置文件中的小说路径不存在或信息不完整：\n{file_path}" if file_path else "配置文件中小说路径或编码信息不完整。"
            if hasattr(self, '_is_ui_ready') and self._is_ui_ready: QMessageBox.warning(self, "警告", msg)
            else: print(f"Startup Warning: {msg}")
            self.book_data = {"title": "", "volumes": [], "file_path": None, "encoding": None}
            self.chapter_tree.clear()
            return False
        if hasattr(self, 'status_label'): self.status_label.setText(f"重新加载: {self.book_data.get('title', '未知标题')}...")
        QApplication.processEvents()
        try:
            with open(file_path, 'r', encoding=encoding) as f: content = f.read()
            chapters = self.parse_chapters(content)
            self.build_chapter_tree(chapters)
            if hasattr(self, 'status_label'): self.status_label.setText(f"已重新加载: {self.book_data.get('title', '未知标题')} (编码: {encoding})")
            return True
        except Exception as e:
            msg = f"重新加载小说 '{self.book_data.get('title', file_path)}' 失败: {str(e)}"
            if hasattr(self, '_is_ui_ready') and self._is_ui_ready: QMessageBox.critical(self, "错误", msg)
            else: print(f"Startup Error: {msg}")
            self.book_data = {"title": "", "volumes": [], "file_path": None, "encoding": None}
            self.chapter_tree.clear()
            return False

    def restore_chapter_states(self, states):
        print(f"DEBUG: Entering MainWindow.restore_chapter_states (processing {len(states)} states)")
        if self.chapter_tree.topLevelItemCount() == 0:
            print("DEBUG: MainWindow.restore_chapter_states - Chapter tree is empty, exiting.")
            return
        book_item = self.chapter_tree.topLevelItem(0)
        if not book_item:
            print("DEBUG: MainWindow.restore_chapter_states - Book item not found, exiting.")
            return

        self.chapter_tree.setUpdatesEnabled(False)
        try:
            for state_data in states:
                if not isinstance(state_data, dict): continue
                path_info = state_data.get("path")
                if not isinstance(path_info, list) or len(path_info) != 2: continue
                vol_title, chap_title = path_info
                for i in range(book_item.childCount()):
                    vol_item = book_item.child(i)
                    if vol_item.text(0) == vol_title:
                        for j in range(vol_item.childCount()):
                            chap_item = vol_item.child(j)
                            if isinstance(chap_item, ChapterTreeItem) and chap_item.original_title == chap_title:
                                chap_item.is_summarized = state_data.get("is_summarized", False)
                                chap_item.summary = state_data.get("summary", "")
                                chap_item.summary_timestamp = float(state_data.get("timestamp", 0))
                                chap_item.content = state_data.get("content", chap_item.content)
                                chap_item.word_count = state_data.get("word_count", len(chap_item.content))
                                chap_item.setText(1, f"{chap_item.word_count}字")
                                chap_item.update_display_text()
                                break
                        break
        finally:
            self.chapter_tree.setUpdatesEnabled(True)
            print("DEBUG: Exiting MainWindow.restore_chapter_states")

    def update_prompt(self):
        self.custom_prompt = self.prompt_input.text()

    def test_connection(self):
        print("DEBUG: test_connection called")
        if not self.validate_config():
            print("DEBUG: test_connection - validate_config failed")
            return

        try:
            api_config = {
                "url": self.api_url_input.text().strip(),
                "key": self.api_key_input.text().strip(),
                "model": self.get_current_model_name()
            }
            print(f"DEBUG: test_connection - API Config: {api_config}")

            encoding_object = self.get_tiktoken_encoding(api_config['model'])
            print(f"DEBUG: test_connection - Encoding Object: {type(encoding_object)}")

            if encoding_object is None:
                QMessageBox.critical(self, "编码器错误", f"无法为模型 '{api_config.get('model','未知')}' 初始化Token编码器。测试中止。")
                self.status_label.setText(f"错误: 模型 '{api_config.get('model','未知')}' Token编码器初始化失败。")
                print("DEBUG: test_connection - encoding_object is None, aborting.")
                return

            print("DEBUG: test_connection - Instantiating LLMProcessor for test")
            processor = LLMProcessor(api_config, self.custom_prompt, encoding_object)
            print("DEBUG: test_connection - LLMProcessor instantiated for test")

            test_text = "这是一个连接测试。"
            self.status_label.setText("正在测试连接..."); QApplication.processEvents()

            print("DEBUG: test_connection - Calling processor.summarize for test")
            summary, in_tokens, out_tokens = processor.summarize(test_text, max_retries=1)
            print(f"DEBUG: test_connection - processor.summarize returned. Summary empty: {not bool(summary)}")

            if summary or summary == "":
                QMessageBox.information(self, "连接成功", f"API连接测试成功！\n模型: {api_config['model']}\n返回: {summary[:100]}...")
                self.status_label.setText("连接测试成功")
            else:
                raise ValueError("API返回内容为空或无效 (None)")

        except Exception as e:
            error_msg = f"API连接测试失败: {str(e)}"
            print(f"DEBUG: test_connection - EXCEPTION: {error_msg}")
            QMessageBox.critical(self, "连接失败", error_msg)
            self.status_label.setText("连接测试失败")

    def auto_save(self):
        self.save_config(silent=True)

    def auto_export_novel_data(self):
        if not self.book_data.get("title"):
            return

        book_title_for_folder = re.sub(r'[\/*?:"<>|]', "_", self.book_data["title"])
        if not book_title_for_folder.strip():
            book_title_for_folder = "Untitled_Novel"

        specific_book_export_path = os.path.join(self.auto_export_base_dir, book_title_for_folder)

        try:
            os.makedirs(specific_book_export_path, exist_ok=True)
            self.export_txt(specific_book_export_path)
            self.export_markdown(specific_book_export_path)
            self.status_label.setText(f"'{book_title_for_folder}' 已自动保存到桌面。")
        except Exception as e:
            error_message = f"自动导出到桌面失败: {str(e)}"
            print(f"ERROR: Auto-export failed: {error_message}")
            self.status_label.setText(error_message)

    def save_chapter_edits(self):
        current_item = self.chapter_tree.currentItem()
        if isinstance(current_item, ChapterTreeItem) and not self.content_display.isReadOnly():
            new_content = self.content_display.toPlainText()
            current_item.content = new_content
            current_item.word_count = len(new_content)
            current_item.setText(1, f"{current_item.word_count}字")

            current_item.is_summarized = False
            current_item.summary = ""
            current_item.summary_timestamp = 0
            current_item.update_display_text()

            self.content_display.setReadOnly(True)
            self.status_label.setText(f"章节 '{current_item.original_title}' 修改已保存。")

            print("DEBUG: Triggering auto-export from save_chapter_edits")
            self.auto_export_novel_data()
        else:
            self.status_label.setText("没有可保存的章节修改或当前非编辑模式。")

    def get_tiktoken_encoding(self, model_name_from_config: str):
        effective_encoding_key = None
        try:
            _model_key_for_tiktoken = model_name_from_config.split('/')[-1].lower()
            encoding_map = {
                'gpt-4': 'cl100k_base',
                'gpt-3.5-turbo': 'cl100k_base',
                'deepseek-chat': 'cl100k_base',
                'qwen': 'cl100k_base',
                'chatglm': 'cl100k_base',
            }
            try:
                # Attempt 1: Try direct model name (which might be a specific version like gpt-4-0613)
                effective_encoding_key = _model_key_for_tiktoken
                with self.tiktoken_cache_lock:
                    if effective_encoding_key in self.tiktoken_encoding_cache:
                        return self.tiktoken_encoding_cache[effective_encoding_key]
                    try:
                        # Try encoding_for_model first
                        encoding_obj = tiktoken.encoding_for_model(effective_encoding_key)
                        self.tiktoken_encoding_cache[effective_encoding_key] = encoding_obj
                        print(f"DEBUG: Tiktoken encoding for '{effective_encoding_key}' (model specific) cached and returned.")
                        return encoding_obj
                    except KeyError:
                        # This specific model name is not directly known by tiktoken.encoding_for_model
                        # Proceed to mapped/generic logic below.
                        pass # Fall through to the next mechanism (mapped or generic cl100k_base)

                # Attempt 2: Check mapped prefixes if direct model name failed
                # This section is reached if encoding_for_model raised KeyError
                derived_encoding_key = None
                for prefix, base_encoding_name in encoding_map.items():
                    if _model_key_for_tiktoken.startswith(prefix):
                        derived_encoding_key = base_encoding_name
                        break

                if not derived_encoding_key: # If no prefix matched, default to cl100k_base
                    derived_encoding_key = 'cl100k_base'

                effective_encoding_key = derived_encoding_key # Use the derived key for caching

                with self.tiktoken_cache_lock:
                    if effective_encoding_key in self.tiktoken_encoding_cache:
                        return self.tiktoken_encoding_cache[effective_encoding_key]

                    # If not in cache, create using get_encoding and store it
                    print(f"DEBUG: Initializing tiktoken encoding for: '{effective_encoding_key}' (derived/default for '{model_name_from_config}')")
                    encoding_obj = tiktoken.get_encoding(effective_encoding_key)
                    self.tiktoken_encoding_cache[effective_encoding_key] = encoding_obj
                    print(f"DEBUG: Tiktoken encoding for '{effective_encoding_key}' cached and returned.")
                    return encoding_obj

            except Exception as e: # Broad exception for any failure in the above logic
                print(f"ERROR: Failed to get/create tiktoken encoding for '{model_name_from_config}'. Error: {e}. Using fallback 'cl100k_base'.")
                with self.tiktoken_cache_lock:
                    if 'cl100k_base' in self.tiktoken_encoding_cache:
                        return self.tiktoken_encoding_cache['cl100k_base']
                    try:
                        # Create, store, and return the default 'cl100k_base' encoder
                        encoding_obj = tiktoken.get_encoding('cl100k_base')
                        self.tiktoken_encoding_cache['cl100k_base'] = encoding_obj
                        print("DEBUG: Tiktoken encoding for 'cl100k_base' (fallback) cached and returned.")
                        return encoding_obj
                    except Exception as e_default:
                        print(f"CRITICAL ERROR: Failed to get default tiktoken encoder 'cl100k_base': {e_default}")
                        return None

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(self, '确认退出', '正在处理任务，确定要退出吗？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.worker_thread.stop()
                self.worker_thread.wait(3000)
                try: self.save_config(silent=True)
                except Exception as e: print(f"Error saving config during forced close: {e}")
                event.accept()
            else:
                event.ignore()
                return
        # This code path is for when the `if self.worker_thread and self.worker_thread.isRunning():` is false initially,
        # or when the user chose 'Yes' to exit, the worker thread was stopped, and config was saved (though that path also calls event.accept() and returns).
        # The primary purpose here is to save config if no worker was running.
        try:
            self.save_config(silent=True)
        except Exception as e:
            print(f"Error saving config on close: {e}")
        event.accept()

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setApplicationName("小说智能分析工具")
    app.setApplicationVersion("2.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
