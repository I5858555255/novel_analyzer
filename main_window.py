# main_window.py
import os
import re
import json
import threading # Though direct use in MainWindow might be minimal after refactor
import queue
import time
import copy # For deepcopy

from PyQt5.QtWidgets import (QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem,
                             QTextEdit, QFileDialog, QPushButton, QComboBox,
                             QProgressBar, QLabel, QSplitter, QVBoxLayout,
                             QHBoxLayout, QWidget, QAction, QMessageBox, QLineEdit,
                             QHeaderView) # QHeaderView added as per prompt
from PyQt5.QtCore import pyqtSignal, Qt, QThread, QTimer, QThreadPool # QThreadPool added as per prompt

import requests # Used for test_connection, though that might be refactored later

# Project module imports
from constants import DEFAULT_MODEL_CONFIGS, DEFAULT_CUSTOM_PROMPT
from llm_processor import LLMProcessor
from custom_widgets import ChapterTreeItem
from threading_utils import SummarizationSignals, SummarizationTask, WorkerThread # SummarizationTask/Signals might not be used directly by MainWindow yet
from dialogs import ManageModelsDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("小说智能分析工具 v2.0")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化变量
        self.book_data = {"title": "", "volumes": []}
        self.llm_processor = None # This might be primarily for the old WorkerThread in summarize_selected
        self.work_queue = queue.Queue() # Used by old WorkerThread
        self.worker_thread = None # Instance of old WorkerThread for summarize_selected
        self.total_tokens = [0, 0]  # [input, output]
        self.default_export_path = ""
        self.custom_prompt = DEFAULT_CUSTOM_PROMPT

        self.model_configs = copy.deepcopy(DEFAULT_MODEL_CONFIGS)
        self.initial_model_keys = set(self.model_configs.keys())

        # For batch processing with QThreadPool (setup for future refactor of summarize_all)
        self.thread_pool = QThreadPool()
        self.active_batch_tasks = 0
        self.batch_start_time = 0
        self.chapters_to_process_total = 0
        self.chapters_processed_count = 0
        self.total_input_tokens_batch = 0
        self.total_output_tokens_batch = 0
        self.average_time_per_chapter = 0


        self.init_ui()
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(lambda: self.save_config(silent=True))
        self.auto_save_timer.start(30000)
        self.load_config(silent=True)

    def init_ui(self):
        # 创建主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # 顶部控制栏
        control_layout = QHBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        for model_key, config in self.model_configs.items():
            self.model_combo.addItem(config["display_name"], model_key)
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
        splitter.addWidget(self.chapter_tree)

        right_panel_layout = QVBoxLayout()
        self.content_display = QTextEdit()
        self.content_display.setReadOnly(True)
        right_panel_layout.addWidget(self.content_display)

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
        self.status_label = QLabel("就绪") # General status
        status_layout.addWidget(self.status_label)
        # ETA and Metrics labels for batch processing (can be initially empty or hidden)
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
        add_model_action = QAction('添加自定义模型', self)
        add_model_action.triggered.connect(self.add_custom_model) # Keep this for now, or change to open_manage_models_dialog
        model_menu.addAction(add_model_action)

        manage_models_action = QAction("管理自定义模型", self)
        manage_models_action.triggered.connect(self.open_manage_models_dialog)
        model_menu.addAction(manage_models_action)

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
                config = self.model_configs[current_data]
                self.api_url_input.setText(config["url"])
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
        self.chapter_tree.clear()
        root_title = self.book_data.get("title", "未命名书籍")
        root = QTreeWidgetItem(self.chapter_tree, [root_title, ""])
        root.setExpanded(True)
        total_chapters = 0
        total_words = 0
        for volume_data in chapters:
            vol_words = sum(c['word_count'] for c in volume_data['chapters'])
            vol_item = QTreeWidgetItem(root, [volume_data['title'], f"{len(volume_data['chapters'])}章, {vol_words}字"])
            vol_item.setExpanded(True)
            for chapter_data in volume_data['chapters']:
                chapter_item = ChapterTreeItem(chapter_data['title'], chapter_data['content'], chapter_data['word_count'], vol_item)
                chapter_item.update_display_text()
                total_chapters += 1
            total_words += vol_words
        root.setText(1, f"{len(chapters)}卷, {total_chapters}章, {total_words}字")
        self.chapter_tree.expandAll()

    def show_content(self, item):
        if isinstance(item, ChapterTreeItem):
            if self.summary_mode_btn.isChecked() or not item.is_summarized: # isChecked means "Display Original"
                self.content_display.setText(item.content)
            else: # Not checked (Display Summary mode) and item IS summarized
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
        if parent and hasattr(parent, 'summary') and parent.summary: context = parent.summary
        try:
            api_config = {"url": self.api_url_input.text().strip(), "key": self.api_key_input.text().strip(), "model": self.get_current_model_name()}
            current_llm_processor_instance = LLMProcessor(api_config, self.custom_prompt)
            self.work_queue.put((current, context))
            self.worker_thread = WorkerThread(self.work_queue, current_llm_processor_instance)
            if self.worker_thread and self.worker_thread.isRunning():
                 QMessageBox.warning(self, "提示", "已有提炼任务在进行中。")
                 return
            self.worker_thread.update_signal.connect(self.handle_update)
            self.worker_thread.progress_signal.connect(self.update_progress)
            self.worker_thread.error_signal.connect(self.handle_error)
            self.worker_thread.finished.connect(self.processing_finished)
            self.worker_thread.start()
            self.summarize_btn.setEnabled(False)
            self.summarize_all_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("正在处理当前章节...")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化失败: {str(e)}")

    def summarize_all(self): # Placeholder for QThreadPool refactor
        QMessageBox.information(self, "提示", "一键提炼功能正在使用新的线程模型重构，请稍后。")
        return
        # The old WorkerThread based logic for summarize_all is now fully removed
        # It will be replaced by QThreadPool logic in a subsequent step.

    def validate_config(self):
        if not self.api_url_input.text().strip():
            QMessageBox.warning(self, "配置错误", "请输入API地址"); return False
        if not self.api_key_input.text().strip():
            QMessageBox.warning(self, "配置错误", "请输入API密钥"); return False
        if not self.get_current_model_name():
            QMessageBox.warning(self, "配置错误", "请选择或输入模型名称"); return False
        return True

    def start_processing(self): # Effectively a no-op now for summarize_selected
        pass

    def stop_processing(self):
        if self.worker_thread and self.worker_thread.isRunning(): # For old WorkerThread
            self.worker_thread.stop()
            self.worker_thread.wait(3000)
        # Add logic here to stop QThreadPool tasks if implemented for summarize_all
        # For now, this only affects summarize_selected
        self.processing_finished() # Reset UI elements

    def processing_finished(self): # Called by WorkerThread or QThreadPool completion logic
        self.summarize_btn.setEnabled(True)
        self.summarize_all_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("处理完成")
        # Reset progress for next operation (WorkerThread doesn't manage batch progress)
        self.progress_bar.setValue(0)
        # For QThreadPool, reset batch-specific counters here too
        self.active_batch_tasks = 0
        self.eta_label.setText("")
        self.metrics_label.setText("")


    def handle_update(self, update_type, data): # For WorkerThread
        if update_type == "summary":
            item, summary_text = data
            item.summary = summary_text
            item.is_summarized = True
            item.summary_timestamp = time.time()
            item.update_display_text()
            current_item = self.chapter_tree.currentItem()
            if current_item == item and not self.summary_mode_btn.isChecked():
                self.content_display.setText(summary_text)

    # Methods for QThreadPool based summarize_all (to be fully wired up later)
    def handle_chapter_summary_update(self, identifier, summary_text):
        item = self.find_chapter_item_by_identifier(identifier) # Helper needed
        if item:
            item.summary = summary_text
            item.is_summarized = True
            item.summary_timestamp = time.time()
            item.update_display_text()
            current_selected_item = self.chapter_tree.currentItem()
            if current_selected_item == item and not self.summary_mode_btn.isChecked():
                self.content_display.setText(summary_text)

    def update_batch_progress(self, in_tokens, out_tokens, chapters_done_this_task):
        self.total_input_tokens_batch += in_tokens
        self.total_output_tokens_batch += out_tokens
        self.chapters_processed_count += chapters_done_this_task # Should be 1 per task

        if self.chapters_to_process_total > 0:
            progress_value = int((self.chapters_processed_count / self.chapters_to_process_total) * 100)
            self.progress_bar.setValue(progress_value)

            if self.chapters_processed_count > 0:
                elapsed_time = time.time() - self.batch_start_time
                self.average_time_per_chapter = elapsed_time / self.chapters_processed_count
                remaining_chapters = self.chapters_to_process_total - self.chapters_processed_count
                eta_seconds = remaining_chapters * self.average_time_per_chapter
                self.eta_label.setText(f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_seconds)) if eta_seconds > 0 else '完成'}")

                chapters_per_minute = self.chapters_processed_count / (elapsed_time / 60) if elapsed_time > 0 else 0
                tokens_per_second = (self.total_input_tokens_batch + self.total_output_tokens_batch) / elapsed_time if elapsed_time > 0 else 0
                self.metrics_label.setText(f"{chapters_per_minute:.2f} 章/分钟 | {tokens_per_second:.2f} Token/秒")

        self.token_label.setText(f"总消耗: 输入 {self.total_tokens[0] + self.total_input_tokens_batch} | 输出 {self.total_tokens[1] + self.total_output_tokens_batch}")


    def handle_chapter_error(self, identifier, error_msg):
        item = self.find_chapter_item_by_identifier(identifier) # Helper needed
        chapter_title = item.original_title if item else str(identifier)
        QMessageBox.warning(self, "提炼错误", f"章节 '{chapter_title}' 处理失败: {error_msg}")
        # Error does not stop other tasks in QThreadPool

    def handle_task_finished(self, identifier): # For QThreadPool tasks
        self.active_batch_tasks -= 1
        if self.active_batch_tasks == 0:
            self.total_tokens[0] += self.total_input_tokens_batch
            self.total_tokens[1] += self.total_output_tokens_batch
            self.token_label.setText(f"Token消耗: 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}")
            self.processing_finished() # Generic UI reset
            QMessageBox.information(self, "完成", "所有章节处理完毕。")
            self.eta_label.setText("全部完成")

    def find_chapter_item_by_identifier(self, identifier):
        # Identifier could be the original_title or a tuple (vol_title, chap_title)
        # This needs to be robust. For now, assuming identifier is unique original_title.
        root = self.chapter_tree.invisibleRootItem()
        for i in range(root.childCount()):
            vol_item = root.child(i)
            for j in range(vol_item.childCount()):
                chap_item = vol_item.child(j)
                if isinstance(chap_item, ChapterTreeItem) and chap_item.original_title == identifier:
                    return chap_item
        return None


    def handle_error(self, error_msg): # For old WorkerThread
        QMessageBox.critical(self, "处理错误", error_msg)
        self.processing_finished()

    def update_progress(self, in_tokens, out_tokens, count): # For old WorkerThread
        self.total_tokens[0] += in_tokens
        self.total_tokens[1] += out_tokens
        self.token_label.setText(
            f"Token消耗: 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}"
        )
        # Old WorkerThread progress was simpler, just based on count of single tasks
        if self.progress_bar.maximum() == 1 or self.progress_bar.maximum() == 0 : # Single task
             self.progress_bar.setMaximum(1) # Ensure it's set for single task
             self.progress_bar.setValue(self.progress_bar.value() + count)
        # If summarize_all were to use this (which it shouldn't long-term),
        # it would need to set progress_bar.setMaximum(len(tasks)) before starting.

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

    def export_txt(self, path):
        with open(os.path.join(path, f"{self.book_data['title']}_提炼总结.txt"), 'w', encoding='utf-8') as f:
            f.write(f"{self.book_data['title']} 提炼总结\n{'=' * 50}\n\n")
            root = self.chapter_tree.invisibleRootItem()
            for i in range(root.childCount()):
                vol = root.child(i)
                f.write(f"{vol.text(0)}\n{'-' * 40}\n")
                for j in range(vol.childCount()):
                    chap = vol.child(j)
                    if chap.is_summarized:
                        f.write(f"\n{chap.text(0)}\n{'-' * 20}\n{chap.summary}\n")
                f.write("\n\n")

    def export_markdown(self, path):
        with open(os.path.join(path, f"{self.book_data['title']}_提炼总结.md"), 'w', encoding='utf-8') as f:
            f.write(f"# {self.book_data['title']} 提炼总结\n\n")
            f.write(f"**生成时间:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Token消耗:** 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}\n\n---\n\n")
            root = self.chapter_tree.invisibleRootItem()
            for i in range(root.childCount()):
                vol = root.child(i)
                f.write(f"## {vol.text(0)}\n\n")
                for j in range(vol.childCount()):
                    chap = vol.child(j)
                    if chap.is_summarized:
                        f.write(f"### {chap.text(0)}\n\n{chap.summary}\n\n")

    def export_json(self, path):
        data = {"title": self.book_data["title"], "export_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                "token_usage": {"input": self.total_tokens[0], "output": self.total_tokens[1]}, "volumes": []}
        root = self.chapter_tree.invisibleRootItem()
        for i in range(root.childCount()):
            vol = root.child(i)
            volume_data = {"title": vol.text(0), "chapters": []}
            for j in range(vol.childCount()):
                chap = vol.child(j)
                if chap.is_summarized:
                    chapter_data = {"title": chap.text(0), "original_length": chap.word_count,
                                    "summary": chap.summary, "summary_length": len(chap.summary)}
                    volume_data["chapters"].append(chapter_data)
            if volume_data["chapters"]: data["volumes"].append(volume_data)
        with open(os.path.join(path, f"{self.book_data['title']}_数据.json"), 'w', encoding='utf-8') as f:
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
            "custom_models": {k: v for k, v in self.model_configs.items() if k not in self.initial_model_keys}, # Save only non-default
            "book_data": self.book_data, "summary_mode": self.summary_mode_btn.isChecked(),
            "custom_prompt": self.custom_prompt, "chapter_states": self.get_chapter_states()
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
        for i in range(root.childCount()):
            vol = root.child(i)
            for j in range(vol.childCount()):
                chap = vol.child(j)
                if isinstance(chap, ChapterTreeItem):
                    states.append({
                        "path": [vol.text(0), chap.original_title if hasattr(chap, 'original_title') else chap.text(0)],
                        "is_summarized": chap.is_summarized, "summary": chap.summary,
                        "timestamp": chap.summary_timestamp, "content": chap.content,
                        "word_count": chap.word_count
                    })
        return states

    def load_config(self, silent=False):
        config_path = "config.json"
        if not os.path.exists(config_path):
            if not silent: QMessageBox.information(self, "提示", "未找到配置文件。")
            else: print("Config file not found on startup. Skipping load.")
            return
        try:
            with open(config_path, 'r', encoding='utf-8') as f: config = json.load(f)

            # Load custom models from config and add to self.model_configs
            # Ensure they don't overwrite initial_model_keys if IDs clash, though custom should be unique
            if "custom_models" in config:
                for model_key, model_data in config["custom_models"].items():
                    if model_key not in self.initial_model_keys: # Only add if truly custom
                         self.model_configs[model_key] = model_data
                         # Add to combo box if not already there (e.g. if config was edited manually)
                         if self.model_combo.findData(model_key) == -1:
                              self.model_combo.addItem(model_data.get("display_name", model_key), model_key)

            if config.get("model"):
                model_to_select = config["model"]
                index = self.model_combo.findData(model_to_select)
                if index == -1: # If not found by data (e.g. old config or custom name entered)
                    index = self.model_combo.findText(model_to_select) # Try by text
                if index != -1: self.model_combo.setCurrentIndex(index)

            self.api_url_input.setText(config.get("api_url", ""))
            self.api_key_input.setText(config.get("api_key", ""))
            self.default_export_path = config.get("export_path", "")
            self.custom_prompt = config.get("custom_prompt", DEFAULT_CUSTOM_PROMPT)
            self.prompt_input.setText(self.custom_prompt)

            reloaded_successfully = False
            if "book_data" in config and config["book_data"].get("file_path"):
                self.book_data = config["book_data"]
                reloaded_successfully = self.reload_novel()

            if reloaded_successfully and "chapter_states" in config:
                self.restore_chapter_states(config["chapter_states"])

            if "summary_mode" in config:
                self.summary_mode_btn.setChecked(config["summary_mode"])
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
        except Exception as e:
            if not silent: QMessageBox.critical(self, "错误", f"加载配置失败: {str(e)}")
            else: print(f"Error during silent load_config: {e}")

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
        if self.chapter_tree.topLevelItemCount() == 0: return
        book_item = self.chapter_tree.topLevelItem(0)
        if not book_item: return
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

    def update_prompt(self):
        self.custom_prompt = self.prompt_input.text()

    def test_connection(self):
        if not self.validate_config(): return
        try:
            api_config = {"url": self.api_url_input.text().strip(), "key": self.api_key_input.text().strip(), "model": self.get_current_model_name()}
            processor = LLMProcessor(api_config, self.custom_prompt) # Pass custom_prompt here
            test_text = "这是一个连接测试。"
            self.status_label.setText("正在测试连接..."); QApplication.processEvents()
            summary, in_tokens, out_tokens = processor.summarize(test_text, max_retries=1)
            if summary:
                QMessageBox.information(self, "连接成功", f"API连接测试成功！\n模型: {api_config['model']}\n返回: {summary[:100]}...")
                self.status_label.setText("连接测试成功")
            else: raise ValueError("API返回内容为空")
        except Exception as e:
            QMessageBox.critical(self, "连接失败", f"API连接测试失败:\n{str(e)}")
            self.status_label.setText("连接测试失败")

    def auto_save(self):
        self.save_config(silent=True)

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
        try: self.save_config(silent=True)
        except Exception as e: print(f"Error saving config on close: {e}")
        event.accept()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setApplicationName("小说智能分析工具")
    app.setApplicationVersion("2.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
[end of novel_analyzer.py]
