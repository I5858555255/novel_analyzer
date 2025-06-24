"""
Main application module for the Novel Analyzer.

This module contains the MainWindow class, which defines the main user interface
and orchestrates the application's functionality, including loading novels,
managing API configurations, processing text for summarization using LLMs,
and handling user interactions. It integrates various components like custom
widgets, LLM processors, threading utilities, and dialogs.
"""
import os
import re
import json
import threading
# import queue # Removed
import time
import copy
import tiktoken
import csv
import sys  # For sys.argv and sys.exit
import datetime # Added for LoggingService
import traceback # Added for LoggingService error logging

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem, QTextEdit,
    QFileDialog, QPushButton, QComboBox, QProgressBar, QLabel, QSplitter,
    QVBoxLayout, QHBoxLayout, QWidget, QAction, QMessageBox, QLineEdit,
    QHeaderView, QSpinBox, QDialog, QFormLayout, QDialogButtonBox,
    QMenu, QInputDialog
)
from PyQt5.QtCore import (
    pyqtSignal, Qt, QThread, QTimer, QThreadPool, QObject, QRunnable
)

# Import constants
from constants import DEFAULT_MODEL_CONFIGS
# Import PerformanceLogger
from performance_logger import PerformanceLogger
# Import ChapterTreeItem
from custom_widgets import ChapterTreeItem
# Import LLMProcessor
from llm_processor import LLMProcessor
# Import Dialogs
from dialogs import ManageModelsDialog, LogViewerDialog # Added LogViewerDialog

# New Imports
from batch_analysis_processor import BatchAnalysisProcessor # This should provide AutoFindTask and Signals
from llm_requirements_data import REQUIREMENTS_STRUCTURE


# Logging Service Implementation
class LogLevel:
    INFO = "INFO"
    ERROR = "ERROR"
    API_REQUEST = "API_REQUEST"
    API_RESPONSE = "API_RESPONSE"
    DATA_CHANGE = "DATA_CHANGE"
    DEBUG = "DEBUG"

class LoggingService:
    def __init__(self, max_entries=1000): # Limit entries to prevent memory issues
        self.log_entries = []
        self.max_entries = max_entries

    def _add_log(self, level, message, details=None):
        if len(self.log_entries) >= self.max_entries:
            self.log_entries.pop(0) # Remove oldest entry if limit reached

        entry = {
            'timestamp': datetime.datetime.now(),
            'level': level,
            'message': str(message)
        }
        if details:
            entry['details'] = details # details can be a string or a dict
        self.log_entries.append(entry)

    def info(self, message, details=None):
        self._add_log(LogLevel.INFO, message, details)

    def error(self, message, details=None, exc_info=False): # exc_info for traceback
        if exc_info:
            if details is None:
                details = {}
            elif not isinstance(details, dict):
                details = {'original_details': str(details)}
            details['traceback'] = traceback.format_exc()
        self._add_log(LogLevel.ERROR, message, details)

    def api_request(self, message, details=None):
        self._add_log(LogLevel.API_REQUEST, message, details)

    def api_response(self, message, details=None):
        self._add_log(LogLevel.API_RESPONSE, message, details)

    def data_change(self, message, details=None):
        self._add_log(LogLevel.DATA_CHANGE, message, details)

    def debug(self, message, details=None):
        self._add_log(LogLevel.DEBUG, message, details)

    def get_logs(self, level_filter=None):
        if level_filter:
            return [entry for entry in self.log_entries if entry['level'] == level_filter]
        return list(self.log_entries) # Return a copy

    def clear_logs(self):
        self.log_entries = []

class MainWindow(QMainWindow):
    CHAPTER_PARSE_PATTERNS = [
        r'第([一二三四五六七八九十百千万零\d]+)[卷部][\s　]*(.+?)(?=\n|$)',
        r'([卷部])([一二三四五六七八九十百千万零\d]+)[\s　]*(.+?)(?=\n|$)',
        r'第([一二三四五六七八九十百千万零\d]+)[章节回][\s　]*(.+?)(?=\n|$)',
        r'([章节回])([一二三四五六七八九十百千万零\d]+)[\s　]*(.+?)(?=\n|$)',
        r'^\s*(\d+)\.(.+?)(?=\n|$)',
    ]

    def __init__(self):
        super().__init__()
        self.logging_service = LoggingService() # Instantiate LoggingService
        self.logging_service.info("Application MainWindow initialized.")

        self.setWindowTitle("小说结构化分析工具 v2.3") # Version Update for logging integration
        self.setGeometry(100, 100, 1400, 900)

        self.book_data = {"title": "", "volumes": []}
        self.analysis_data = {}
        self.custom_requirements_overrides = {}
        self.custom_created_requirements = {}
        self.current_selected_requirement_id = None
        self.CHAPTER_OUTLINE_REQ_ID = "AG_3_2_3" # ID for "章节大纲"

        self.llm_processor = None
        self.total_tokens = [0, 0]
        self.default_export_path = ""
        self.model_configs = copy.deepcopy(DEFAULT_MODEL_CONFIGS)
        self.initial_model_keys = set(self.model_configs.keys())

        self.tiktoken_encoding_cache = {}
        self.tiktoken_cache_lock = threading.Lock()

        self.thread_pool = QThreadPool()
        desired_thread_count = 16 # Increased from 8
        try:
            cpu_cores = os.cpu_count()
            if cpu_cores:
                # You might want to base this on cpu_cores, e.g., cpu_cores * 2 or cpu_cores * 4
                # For now, directly setting to 16 as an initial test.
                pass
        except Exception:
            pass
        self.thread_pool.setMaxThreadCount(desired_thread_count)
        self.logging_service.debug(f"Thread pool max threads set to: {self.thread_pool.maxThreadCount()}")

        self.stop_batch_requested = False
        self.is_chapter_analysis_view_active = False

        self.batch_analyzer = BatchAnalysisProcessor(self)

        self._is_ui_ready = False
        self.auto_export_base_dir = os.path.join(
            os.path.expanduser("~"), "Desktop", "NovelAnalyzer_Exports")

        self.init_ui()
        self.populate_requirements_tree()
        self._is_ui_ready = True
        self.logging_service.info("UI initialized and requirements tree populated.")

        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(
            lambda: self.save_config(silent=True))
        self.auto_save_timer.start(1000)

        self.load_config(silent=True)

    def get_chapter_global_index(self, target_chapter_item: ChapterTreeItem):
        """Calculates the global index of a chapter item in the novel."""
        global_idx = 0
        if self.chapter_tree.topLevelItemCount() > 0:
            book_item = self.chapter_tree.topLevelItem(0)
            for i in range(book_item.childCount()): # Volumes
                vol_item = book_item.child(i)
                for j in range(vol_item.childCount()): # Chapters
                    chap_item = vol_item.child(j)
                    if isinstance(chap_item, ChapterTreeItem): # Count only actual chapters
                        if chap_item == target_chapter_item:
                            return global_idx
                        global_idx += 1
        return -1 # Should not happen if item is from the tree

    def _get_current_config_for_saving(self):
        return {
            "model": self.get_current_model_name(),
            "api_url": self.api_url_input.text(),
            "api_key": self.api_key_input.text(),
            "export_path": self.default_export_path,
            "custom_models": {k: v for k, v in self.model_configs.items() if k not in self.initial_model_keys},
            "book_data": {**self.book_data, "analysis": self.analysis_data if hasattr(self, 'analysis_data') else {}},
            "custom_requirements_overrides": self.custom_requirements_overrides if hasattr(self, 'custom_requirements_overrides') else {},
            "custom_created_requirements": self.custom_created_requirements if hasattr(self, 'custom_created_requirements') else {},
            "chapter_states": self.get_chapter_states(),
        }

    def _create_control_panel_api(self):
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
        return control_layout

    def _create_control_panel_file(self):
        control_layout2 = QHBoxLayout()
        self.load_btn = QPushButton("导入小说")
        self.load_btn.clicked.connect(self.load_novel)
        control_layout2.addWidget(self.load_btn)

        self.save_config_btn = QPushButton("保存配置")
        self.save_config_btn.clicked.connect(self.save_config)
        control_layout2.addWidget(self.save_config_btn)

        self.load_config_btn = QPushButton("加载配置")
        self.load_config_btn.clicked.connect(self.load_config)
        control_layout2.addWidget(self.load_config_btn)

        self.run_full_analysis_btn = QPushButton("执行全面分析")
        self.run_full_analysis_btn.setToolTip("对当前选定小说的所有章节，依次处理每一章节的所有分析项。警告：此操作可能耗时较长且API费用较高！")
        self.run_full_analysis_btn.clicked.connect(self.start_full_novel_analysis)
        control_layout2.addWidget(self.run_full_analysis_btn)
        return control_layout2

    def _create_summarization_controls(self):
        btn_layout = QHBoxLayout()
        self.stop_btn = QPushButton("停止处理")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        return btn_layout

    def _create_requirements_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.requirements_tree = QTreeWidget()
        self.requirements_tree.setHeaderLabels(["设定/资料"])
        self.requirements_tree.itemClicked.connect(self.on_requirement_selected)
        self.requirements_tree.itemChanged.connect(self.on_requirement_item_changed) # Connect new handler
        self.requirements_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.requirements_tree.customContextMenuRequested.connect(self.show_requirements_context_menu)
        layout.addWidget(self.requirements_tree)
        return panel

    def show_requirements_context_menu(self, position):
        selected_item = self.requirements_tree.itemAt(position)
        menu = QMenu()
        add_top_level_action = menu.addAction("新增分析项 (顶级)")
        edit_title_action = None
        edit_description_action = None
        add_sub_item_action = None
        delete_action = None
        edit_processing_type_action = None # Initialize

        # Common actions if an item is selected
        if selected_item:
            edit_title_action = menu.addAction("编辑标题")
            edit_description_action = menu.addAction("编辑描述")
            add_sub_item_action = menu.addAction("新增子分析项")

            # Add "Edit Processing Type" for any selected item.
            # The method edit_requirement_processing_type will handle logic based on item type.
            # menu.addSeparator() # Optional: consider placement if grouping with other edits
            # edit_processing_type_action = menu.addAction("编辑处理类型") # Commented out as per requirement

            menu.addSeparator()
            delete_action = menu.addAction("删除选定分析项")
            delete_action.setEnabled(True)
            # Direct connection for delete, as per existing pattern for it
            delete_action.triggered.connect(lambda: self.delete_requirement_item(selected_item))
        else:
            # No item selected, actions requiring selection are not added or are disabled
            pass

        action = menu.exec_(self.requirements_tree.mapToGlobal(position))

        if action == add_top_level_action:
            self.add_new_requirement_item(parent_item=None)
        # Check actions that require a selected_item and were not directly connected
        # (or were connected but also need to be checked if using a mixed approach, though less ideal)
        elif selected_item:
            if action == edit_title_action:
                self.edit_requirement_title(selected_item)
            elif action == edit_description_action:
                self.edit_requirement_description(selected_item)
            elif action == add_sub_item_action:
                self.add_new_requirement_item(parent_item=selected_item)
            # elif action == edit_processing_type_action: # New action handling # Commented out
            #     self.edit_requirement_processing_type(selected_item)
            # delete_action is handled by its direct connection, so no 'elif action == delete_action' needed here.

    def add_new_requirement_item(self, parent_item=None):
        title, ok = QInputDialog.getText(self, "新增分析项", "标题:")
        if not ok or not title.strip():
            self.status_label.setText("新增分析项已取消。")
            self.logging_service.info("Add new requirement cancelled by user (title input).")
            return
        title = title.strip()

        description, ok = QInputDialog.getMultiLineText(self, "新增分析项", "描述 (LLM提示部分):")
        if not ok:
            self.status_label.setText("新增分析项描述输入已取消。")
            self.logging_service.info("Add new requirement description input cancelled.", details={'title': title})
            return

        processing_type_choices = ["(Default/Inherit)", "aggregate", "prompt_only", "chapter_specific"]
        selected_type_str, type_ok = QInputDialog.getItem(self, "新增分析项", "选择处理类型:", processing_type_choices, 0, False)

        if not type_ok:
            self.status_label.setText("新增分析项已取消 (处理类型选择)。")
            self.logging_service.info("Add new requirement cancelled at processing_type selection.", details={'title': title})
            return

        actual_processing_type = None
        if selected_type_str == "aggregate":
            actual_processing_type = "aggregate"
        elif selected_type_str == "prompt_only":
            actual_processing_type = "prompt_only"
        elif selected_type_str == "chapter_specific":
            actual_processing_type = "chapter_specific"
        # If "(Default/Inherit)", actual_processing_type remains None

        req_id = f"custom_{int(time.time())}_{threading.get_ident()}"
        while req_id in self.custom_created_requirements or self._find_req_data_recursive(REQUIREMENTS_STRUCTURE, req_id):
             req_id = f"custom_{int(time.time())}_{threading.get_ident()}_{os.urandom(2).hex()}"

        parent_id_str = None
        if parent_item:
            parent_id_str = parent_item.data(0, Qt.UserRole)

        new_item_data = {
            'id': req_id, 'title': title, 'description': description,
            'parent_id': parent_id_str, 'sub_item_ids': []
        }
        if actual_processing_type: # Only add the key if a specific type was chosen
            new_item_data['processing_type'] = actual_processing_type

        self.custom_created_requirements[req_id] = new_item_data

        if parent_id_str and parent_id_str in self.custom_created_requirements:
            self.custom_created_requirements[parent_id_str]['sub_item_ids'].append(req_id)

        if req_id not in self.custom_requirements_overrides:
            self.custom_requirements_overrides[req_id] = {}
        self.custom_requirements_overrides[req_id]['title'] = title
        self.custom_requirements_overrides[req_id]['description'] = description
        # Ensure new custom items participate by default
        self.custom_requirements_overrides[req_id]['participates_in_extraction'] = True

        self.logging_service.data_change("Custom requirement added.", details={'req_id': req_id, 'title': title, 'parent_id': parent_id_str, 'participates': True})
        self.populate_requirements_tree()
        self.status_label.setText(f"已新增分析项: {title}")

    def _delete_recursive_from_data_stores(self, req_id_to_delete):
        if req_id_to_delete in self.custom_created_requirements:
            item_to_delete_data = self.custom_created_requirements[req_id_to_delete]
            children_ids_copy = list(item_to_delete_data.get('sub_item_ids', []))
            for child_id in children_ids_copy:
                self._delete_recursive_from_data_stores(child_id)
            parent_id = item_to_delete_data.get('parent_id')
            if parent_id and parent_id in self.custom_created_requirements:
                if req_id_to_delete in self.custom_created_requirements[parent_id]['sub_item_ids']:
                    self.custom_created_requirements[parent_id]['sub_item_ids'].remove(req_id_to_delete)
            del self.custom_created_requirements[req_id_to_delete]
            self.logging_service.data_change("Custom requirement structure deleted.", details={'req_id': req_id_to_delete})


        self.custom_requirements_overrides.pop(req_id_to_delete, None)
        self.analysis_data.pop(req_id_to_delete, None)
        self.logging_service.data_change("Requirement data deleted (analysis_data and overrides).", details={'req_id': req_id_to_delete})

    def delete_requirement_item(self, item): # item is the QTreeWidgetItem
        if not item: return
        req_id = item.data(0, Qt.UserRole)
        item_title = item.text(0) # Get title for messages before item is potentially removed

        if not req_id:
            self.logging_service.error("Delete item called with no req_id.", details={'item_text': item_title})
            QMessageBox.warning(self, "错误", "无法确定所选项目的ID。")
            return

        is_custom = req_id in self.custom_created_requirements

        if is_custom:
            reply = QMessageBox.question(self, "确认删除自定义项",
                                       f"确定要永久删除自定义分析项 '{item_title}' 及其所有子项和相关笔记吗？此操作无法撤销。",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.logging_service.info(f"Deletion confirmed for custom requirement: {item_title}", {'req_id': req_id})
                self._delete_recursive_from_data_stores(req_id)
                # _delete_recursive_from_data_stores already handles custom_overrides.pop and analysis_data.pop for the req_id and its custom children.
                self.populate_requirements_tree()
                self.status_label.setText(f"自定义分析项 '{item_title}' 已删除。")
                self.logging_service.data_change("Custom requirement deleted.", details={'req_id': req_id, 'title': item_title})
                if self.current_selected_requirement_id == req_id:
                    self.current_selected_requirement_id = None
                    self.refresh_analysis_input_area_for_selected_req()
            else:
                self.logging_service.info(f"Deletion cancelled for custom requirement: {item_title}", {'req_id': req_id})
        else: # Default item
            reply = QMessageBox.question(self, "确认隐藏默认项",
                                       f"确定要隐藏默认分析项 '{item_title}' 吗？其分析笔记也将被清除。之后可统一恢复所有隐藏的默认项。", # TODO: Implement recovery mechanism later
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                if req_id not in self.custom_requirements_overrides:
                    self.custom_requirements_overrides[req_id] = {}
                self.custom_requirements_overrides[req_id]['hidden'] = True
                # Also set participates_in_extraction to False when hiding a default item
                self.custom_requirements_overrides[req_id]['participates_in_extraction'] = False
                self.analysis_data.pop(req_id, None) # Clear associated analysis notes for the default item

                self.logging_service.data_change("Default requirement hidden and set to non-participating.", details={'req_id': req_id, 'title': item_title})
                self.populate_requirements_tree()
                self.status_label.setText(f"默认分析项 '{item_title}' 已隐藏。")
                if self.current_selected_requirement_id == req_id: # If the hidden item was selected
                    self.current_selected_requirement_id = None
                    self.refresh_analysis_input_area_for_selected_req()
            else:
                self.logging_service.info(f"Hiding cancelled for default requirement: {item_title}", {'req_id': req_id})

    def edit_requirement_title(self, item):
        original_title = item.text(0)
        requirement_id = item.data(0, Qt.UserRole)
        if requirement_id is None: return

        new_title, ok = QInputDialog.getText(self, "编辑分析项标题", "标题:", QLineEdit.Normal, original_title)

        if ok and new_title and new_title.strip():
            new_title = new_title.strip()
            if requirement_id not in self.custom_requirements_overrides:
                self.custom_requirements_overrides[requirement_id] = {}
            self.custom_requirements_overrides[requirement_id]['title'] = new_title
            item.setText(0, new_title)
            self.logging_service.data_change("Requirement title changed.", details={'req_id': requirement_id, 'new_title': new_title})
            self.status_label.setText(f"分析项 '{new_title}' 标题已更新。")
            if self.current_selected_requirement_id == requirement_id:
                self.refresh_analysis_input_area_for_selected_req()
        elif ok and not new_title.strip():
            QMessageBox.warning(self, "无效标题", "分析项标题不能为空。")

    def edit_requirement_description(self, item):
        original_description = item.data(0, Qt.UserRole + 1) or ""
        requirement_id = item.data(0, Qt.UserRole)
        if requirement_id is None: return

        new_description, ok = QInputDialog.getMultiLineText(self, "编辑分析项描述", "描述 (LLM提示部分):", original_description)

        if ok:
            if requirement_id not in self.custom_requirements_overrides:
                self.custom_requirements_overrides[requirement_id] = {}
            self.custom_requirements_overrides[requirement_id]['description'] = new_description
            item.setData(0, Qt.UserRole + 1, new_description)
            self.logging_service.data_change("Requirement description changed.", details={'req_id': requirement_id, 'desc_length': len(new_description)})
            self.status_label.setText(f"分析项 '{item.text(0)}' 描述已更新。")
            if self.current_selected_requirement_id == requirement_id:
                self.refresh_analysis_input_area_for_selected_req()

    def edit_requirement_processing_type(self, item):
        if not item:
            self.logging_service.error("edit_requirement_processing_type called with no item.")
            return

        req_id = item.data(0, Qt.UserRole)
        item_title_for_dialog = item.text(0) # Get title for dialog

        if not req_id:
            self.logging_service.error("edit_requirement_processing_type called with invalid req_id.", details={'item_title': item_title_for_dialog})
            QMessageBox.warning(self, "错误", "无法确定所选项目的ID。")
            return

        is_custom = req_id in self.custom_created_requirements
        current_processing_type = None
        base_processing_type_for_default = None

        if is_custom:
            current_processing_type = self.custom_created_requirements[req_id].get('processing_type')
        else: # Default item
            default_item_data_from_structure = self._find_req_data_recursive(REQUIREMENTS_STRUCTURE, req_id)
            if default_item_data_from_structure:
                base_processing_type_for_default = default_item_data_from_structure.get('processing_type')

            # Effective current type for a default item is its override, or its base type if no override
            current_processing_type = self.custom_requirements_overrides.get(req_id, {}).get('processing_type', base_processing_type_for_default)

        choices = ["(Default/Inherit)", "aggregate", "prompt_only", "chapter_specific"]
        current_choice_index = 0 # Default to "(Default/Inherit)"
        if current_processing_type == "aggregate":
            current_choice_index = 1
        elif current_processing_type == "prompt_only":
            current_choice_index = 2
        elif current_processing_type == "chapter_specific":
            current_choice_index = 3

        selected_type_str, ok = QInputDialog.getItem(
            self, "编辑处理类型", f"为 '{item_title_for_dialog}' 选择处理类型:",
            choices, current_choice_index, False
        )

        if not ok:
            self.logging_service.info(
                f"Edit processing_type cancelled for req_id: {req_id}",
                details={'item_title': item_title_for_dialog}
            )
            self.status_label.setText("编辑处理类型已取消。")
            return

        new_processing_type_value = None
        if selected_type_str == "aggregate":
            new_processing_type_value = "aggregate"
        elif selected_type_str == "prompt_only":
            new_processing_type_value = "prompt_only"
        elif selected_type_str == "chapter_specific":
            new_processing_type_value = "chapter_specific"

        if new_processing_type_value == current_processing_type: # This checks effective types
            # For a default item, if user selected "(Default/Inherit)" and it was already inheriting (no override or override matches base), this is true.
            # If user selected a specific type and it matches an existing override (or base if no override), this is true.
            # One edge case: if user selects "(Default/Inherit)" for a default item that HAD an override, this IS a change.
            # The current_processing_type already reflects the *effective* type.
            # The new_processing_type_value is what the user *wants* it to be effectively.
            # The actual storage change is what matters.

            # Let's refine the "no change" logic based on actual storage impact.
            no_actual_change = False
            if is_custom:
                if current_processing_type == new_processing_type_value: # Covers None == None too
                    no_actual_change = True
            else: # Default item
                override_exists = req_id in self.custom_requirements_overrides and \
                                  'processing_type' in self.custom_requirements_overrides[req_id]

                if new_processing_type_value is None: # User wants to revert to base
                    if not override_exists: # Was already using base, no change
                        no_actual_change = True
                else: # User wants to set a specific override
                    if override_exists and self.custom_requirements_overrides[req_id].get('processing_type') == new_processing_type_value:
                        no_actual_change = True
                    elif not override_exists and base_processing_type_for_default == new_processing_type_value:
                        # User selected the same type as base, and there was no override.
                        # This means we don't need to create an override.
                        no_actual_change = True

            if no_actual_change:
                self.status_label.setText(f"处理类型未更改: '{item_title_for_dialog}' 的有效类型仍为 '{selected_type_str}'.")
                return

        # Update the stored data
        if is_custom:
            if new_processing_type_value is None:
                self.custom_created_requirements[req_id].pop('processing_type', None)
            else:
                self.custom_created_requirements[req_id]['processing_type'] = new_processing_type_value
        else: # Default item
            if new_processing_type_value is None: # Revert to base type
                if req_id in self.custom_requirements_overrides:
                    self.custom_requirements_overrides[req_id].pop('processing_type', None)
                    if not self.custom_requirements_overrides[req_id]: # If dict for this req_id is now empty
                        self.custom_requirements_overrides.pop(req_id)
            else: # Set or change an override
                self.custom_requirements_overrides.setdefault(req_id, {})['processing_type'] = new_processing_type_value

        self.logging_service.data_change(
            "Requirement processing_type changed.",
            details={'req_id': req_id, 'item_title': item_title_for_dialog, 'new_effective_type': new_processing_type_value if new_processing_type_value else "Default/Inherit", 'is_custom': is_custom}
        )
        self.status_label.setText(f"分析项 '{item_title_for_dialog}' 的处理类型已更新为 '{selected_type_str}'.")

        if self.current_selected_requirement_id == req_id:
            # Re-trigger selection logic to update UI based on the new type
            self.on_requirement_selected(item, 0)
            # refresh_analysis_input_area_for_selected_req is called within on_requirement_selected if type is aggregate

    def _create_novel_display_splitter(self):
        center_pane_widget = QWidget()
        center_pane_layout = QVBoxLayout(center_pane_widget)
        center_pane_layout.setContentsMargins(0,0,0,0)

        self.chapter_tree = QTreeWidget()
        self.chapter_tree.setHeaderLabels(["章节/卷", "字数"])
        self.chapter_tree.itemClicked.connect(self.show_content)
        header = self.chapter_tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setMinimumSectionSize(200)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)

        content_area_widget = QWidget()
        content_area_layout = QVBoxLayout(content_area_widget)
        content_area_layout.setContentsMargins(0,0,0,0)

        view_controls_layout = QHBoxLayout()
        self.toggle_chapter_view_btn = QPushButton("查看章节大纲")
        self.toggle_chapter_view_btn.setToolTip("查看当前选定章节的“章节大纲”分析结果。")
        self.toggle_chapter_view_btn.setCheckable(True)
        self.toggle_chapter_view_btn.clicked.connect(self.toggle_chapter_analysis_view)
        view_controls_layout.addWidget(self.toggle_chapter_view_btn)
        view_controls_layout.addStretch()
        content_area_layout.addLayout(view_controls_layout)

        self.content_display = QTextEdit()
        self.content_display.setReadOnly(True)
        content_area_layout.addWidget(self.content_display)

        display_splitter = QSplitter(Qt.Horizontal)
        display_splitter.addWidget(self.chapter_tree)
        display_splitter.addWidget(content_area_widget)
        display_splitter.setSizes([140, 420])

        center_pane_layout.addWidget(display_splitter)
        return center_pane_widget

    def _create_analysis_input_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        analysis_input_label = QLabel("分析笔记/摘录 (Markdown):")
        layout.addWidget(analysis_input_label)
        self.analysis_input_area = QTextEdit()
        self.analysis_input_area.setReadOnly(True)
        layout.addWidget(self.analysis_input_area)
        return panel

    def _create_status_bar(self):
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
        return status_layout

    def _create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件')
        export_action = QAction('设置导出路径', self)
        export_action.triggered.connect(self.set_export_path)
        file_menu.addAction(export_action)

        view_menu = menubar.addMenu("视图") # New View Menu
        self.show_log_viewer_action = QAction("查看系统日志", self)
        self.show_log_viewer_action.triggered.connect(self.open_log_viewer_dialog)
        view_menu.addAction(self.show_log_viewer_action)

        model_menu = menubar.addMenu('模型管理')
        self.add_model_action = QAction('添加自定义模型', self)
        self.add_model_action.triggered.connect(self.add_custom_model)
        model_menu.addAction(self.add_model_action)
        self.manage_models_action = QAction("管理自定义模型", self)
        self.manage_models_action.triggered.connect(self.open_manage_models_dialog)
        model_menu.addAction(self.manage_models_action)

        save_analysis_as_action = QAction('另存分析...', self)
        save_analysis_as_action.triggered.connect(self.save_analysis_as)
        file_menu.addAction(save_analysis_as_action)

    def open_log_viewer_dialog(self):
        # Pass LogLevel class itself, not an instance
        dialog = LogViewerDialog(self.logging_service, LogLevel, self)
        dialog.exec_()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        api_controls = self._create_control_panel_api()
        main_layout.addLayout(api_controls)

        file_controls = self._create_control_panel_file()
        main_layout.addLayout(file_controls)

        self.main_splitter = QSplitter(Qt.Horizontal)
        novel_display_panel = self._create_novel_display_splitter()
        self.main_splitter.addWidget(novel_display_panel)
        requirements_panel = self._create_requirements_panel()
        self.main_splitter.addWidget(requirements_panel)
        analysis_input_panel = self._create_analysis_input_panel()
        self.main_splitter.addWidget(analysis_input_panel)
        self.main_splitter.setSizes([560, 420, 420])
        main_layout.addWidget(self.main_splitter, 1)

        summarization_controls_layout = self._create_summarization_controls()
        main_layout.addLayout(summarization_controls_layout)

        status_bar = self._create_status_bar()
        main_layout.addLayout(status_bar)
        self._create_menu_bar()

    def populate_requirements_tree(self):
        self.requirements_tree.clear()
        self.requirements_tree.setUpdatesEnabled(False)
        temp_item_map = {}

        def add_default_items_recursive(parent_widget_item, default_items_list):
            for item_data in default_items_list:
                current_req_id = item_data['id']
                # Check if default item should be hidden
                if self.custom_requirements_overrides.get(current_req_id, {}).get('hidden', False):
                    self.logging_service.debug(f"Skipping hidden default item: {current_req_id} in populate_requirements_tree")
                    continue # Skip this item and its default children (as they are part of the default structure)

                # req_id = item_data['id'] # This is same as current_req_id
                override_data = self.custom_requirements_overrides.get(current_req_id, {})
                title = override_data.get('title', item_data['title'])
                description = override_data.get('description', item_data.get('description', ''))
                tree_item = QTreeWidgetItem(parent_widget_item, [title])
                tree_item.setData(0, Qt.UserRole, current_req_id) # Use current_req_id consistently
                tree_item.setData(0, Qt.UserRole + 1, description)
                tree_item.setToolTip(0, title) # Add tooltip for default items

                # Set checkable and initial state for default items
                tree_item.setFlags(tree_item.flags() | Qt.ItemIsUserCheckable)
                # Default to checked, unless 'participates_in_extraction' is explicitly False in overrides
                participates = override_data.get('participates_in_extraction', True)
                tree_item.setCheckState(0, Qt.Checked if participates else Qt.Unchecked)

                temp_item_map[current_req_id] = tree_item # Use current_req_id consistently
                if 'sub_items' in item_data and item_data['sub_items']:
                    add_default_items_recursive(tree_item, item_data['sub_items'])

        add_default_items_recursive(self.requirements_tree.invisibleRootItem(), REQUIREMENTS_STRUCTURE)

        custom_items_to_add = list(self.custom_created_requirements.values())
        added_custom_ids_this_pass = set()
        max_passes = len(custom_items_to_add) + 1
        for _ in range(max_passes):
            items_added_in_this_pass = 0
            for item_data_custom in custom_items_to_add:
                req_id = item_data_custom['id']
                if req_id in added_custom_ids_this_pass: continue
                parent_id = item_data_custom.get('parent_id')
                parent_widget_node = None
                if parent_id:
                    parent_widget_node = temp_item_map.get(parent_id)
                else:
                    parent_widget_node = self.requirements_tree.invisibleRootItem()
                if parent_widget_node:
                    override_data = self.custom_requirements_overrides.get(req_id, {})
                    title = override_data.get('title', item_data_custom['title'])
                    description = override_data.get('description', item_data_custom.get('description', ''))
                    display_title = f"[自定义] {title}" if not title.startswith("[自定义]") else title
                    tree_item = QTreeWidgetItem(parent_widget_node, [display_title])
                    tree_item.setData(0, Qt.UserRole, req_id)
                    tree_item.setData(0, Qt.UserRole + 1, description)
                    tree_item.setToolTip(0, display_title) # Add tooltip for custom items

                    # Set checkable and initial state for custom items
                    tree_item.setFlags(tree_item.flags() | Qt.ItemIsUserCheckable)
                    # Default to checked, unless 'participates_in_extraction' is explicitly False in overrides
                    participates = override_data.get('participates_in_extraction', True)
                    tree_item.setCheckState(0, Qt.Checked if participates else Qt.Unchecked)

                    temp_item_map[req_id] = tree_item
                    added_custom_ids_this_pass.add(req_id)
                    items_added_in_this_pass +=1
            if items_added_in_this_pass == 0 and len(added_custom_ids_this_pass) == len(custom_items_to_add):
                break
        self.requirements_tree.expandToDepth(1)
        self.requirements_tree.setUpdatesEnabled(True)

    def on_requirement_item_changed(self, item, column):
        if not item or column != 0: # Only react to changes in the first column (checkbox)
            return

        requirement_id = item.data(0, Qt.UserRole)
        if not requirement_id:
            self.logging_service.error("on_requirement_item_changed: No requirement_id found for item.",
                                       details={'item_text': item.text(0)})
            return

        is_checked = item.checkState(0) == Qt.Checked

        # Update self.custom_requirements_overrides
        if requirement_id not in self.custom_requirements_overrides:
            self.custom_requirements_overrides[requirement_id] = {}

        self.custom_requirements_overrides[requirement_id]['participates_in_extraction'] = is_checked

        self.logging_service.data_change(
            "Requirement participation changed.",
            details={
                'req_id': requirement_id,
                'title': item.text(0),
                'participates_in_extraction': is_checked
            }
        )
        # If the currently selected item's participation changed, we might need to update the UI
        # if its processing behavior is affected by this flag.
        if self.current_selected_requirement_id == requirement_id:
            # Potentially re-evaluate display or available actions if participation affects them
            # For now, re-triggering selection logic might be enough if it considers participation
            self.on_requirement_selected(item, 0) # Re-trigger selection logic


    def on_requirement_selected(self, item, column):
        if not item: return
        requirement_id = item.data(0, Qt.UserRole)
        description = item.data(0, Qt.UserRole + 1)
        title_display = item.text(0)
        if requirement_id:
            self.current_selected_requirement_id = requirement_id # Always set this
            req_data = self._find_any_req_data(requirement_id)
            processing_type = req_data.get('processing_type') if req_data else None
            display_title = req_data.get('title', title_display) if req_data else title_display

            if processing_type == 'non_participating':
                actual_type = req_data.get('original_processing_type', 'N/A') if req_data else 'N/A'
                if actual_type == 'N/A' or actual_type is None: # Try to get it from base if not in original_processing_type
                    actual_type = self._get_base_processing_type(requirement_id) or '未指定'
                self.status_label.setText(f"未参与提取: {display_title} (ID: {requirement_id})")
                self.analysis_input_area.setMarkdown(f"# {display_title}\n\n此分析项当前未选中参与提取（复选框未勾选）。\n其实际处理类型 **{actual_type}** 将不生效，直到重新勾选。")
            elif processing_type == 'chapter_specific':
                self.status_label.setText(f"章节大纲模式: {display_title} (ID: {requirement_id})")
                self.analysis_input_area.setMarkdown(f"# {display_title}\n\n章节大纲在此处生成。\n\n请在左上方选择一个章节，并使用“查看章节大纲”按钮查看具体内容。")
            # Always refresh the consolidated view when a selection is made.
            self.refresh_analysis_input_area_for_selected_req()
            # Update status label to show what is selected, even if the view is consolidated.
            self.status_label.setText(f"已选定: {display_title} (ID: {requirement_id}, 类型: {processing_type if processing_type else 'N/A'})")

        else: # No item selected
            self.current_selected_requirement_id = None
            self.analysis_input_area.setMarkdown("请在左侧选择一个分析项。选择后，此处将显示所有可参与分析项的整合脉络视图。")
            self.status_label.setText("未选择分析项。")

    def refresh_analysis_input_area_for_selected_req(self):
        self.analysis_input_area.clear()
        markdown_parts = ["# 全书分析脉络（整合视图）\n\n"]
        
        # Helper function to recursively traverse the QTreeWidget
        def traverse_tree_items(parent_item):
            for i in range(parent_item.childCount()):
                tree_item = parent_item.child(i)
                req_id = tree_item.data(0, Qt.UserRole)
                if not req_id:
                    continue

                req_data = self._find_any_req_data(req_id)
                if not req_data:
                    continue

                effective_processing_type = req_data.get('processing_type')
                item_title = req_data.get('title', req_id)

                # Filter out non-participating, prompt_only, and chapter_specific items
                if effective_processing_type == 'non_participating' or \
                   effective_processing_type == 'prompt_only' or \
                   effective_processing_type == self.CHAPTER_OUTLINE_REQ_ID or \
                   effective_processing_type == 'chapter_specific': # General chapter_specific check
                    # Recursively process children even if parent is skipped for display
                    traverse_tree_items(tree_item)
                    continue

                # Process 'aggregate' items (and any other types that should be displayed similarly)
                if effective_processing_type == 'aggregate': # Or other relevant types
                    snippets_list = self.analysis_data.get(req_id, [])
                    if snippets_list:
                        markdown_parts.append(f"## {item_title} (ID: {req_id})\n\n")

                        # Sort snippets by chapter_order
                        sorted_snippets = sorted(snippets_list, key=lambda x: x.get('chapter_order', float('inf')))

                        consolidated_item_snippets = []
                        for snippet_data in sorted_snippets:
                            chapter_title = snippet_data.get('chapter_title', '未知章节')
                            snippet_text = snippet_data.get('snippet', '[无内容]').strip()
                            # chapter_order_display = snippet_data.get('chapter_order', 'N/A') # Not used in this new format per snippet

                            # Prepend each snippet with its chapter title
                            consolidated_item_snippets.append(f"### 章节: {chapter_title}\n{snippet_text}\n")

                        markdown_parts.append("\n".join(consolidated_item_snippets) + "\n---\n\n") # Add horizontal rule after each item's content
                    else:
                        markdown_parts.append(f"## {item_title} (ID: {req_id})\n\n_此分析项无聚合笔记。_\n\n---\n\n")

                # Recursively process children
                traverse_tree_items(tree_item)

        # Start traversal from the invisible root item of the requirements_tree
        traverse_tree_items(self.requirements_tree.invisibleRootItem())

        if len(markdown_parts) == 1: # Only the initial title was added
            markdown_parts.append("当前没有可显示的聚合分析内容。请确保相关分析项已勾选参与提取，并且已经处理生成了笔记。\n")

        self.analysis_input_area.setMarkdown("".join(markdown_parts))
        self.status_label.setText("整合分析脉络视图已刷新。")


    def _get_base_processing_type(self, req_id_to_find):
        """Helper to get the original processing type from REQUIREMENTS_STRUCTURE or custom_created_requirements."""
        # Check custom created first
        if req_id_to_find in self.custom_created_requirements:
            # For custom items, 'processing_type' is the base.
            # 'original_processing_type' is only added by _find_any_req_data if it becomes non_participating.
            return self.custom_created_requirements[req_id_to_find].get('processing_type')

        # Then check default structure
        default_item_base = self._find_req_data_recursive(REQUIREMENTS_STRUCTURE, req_id_to_find)
        if default_item_base:
            # For default items, the type in the structure is the base.
            # Overrides are handled by _find_any_req_data, which might also set original_processing_type.
            return default_item_base.get('processing_type')

        # Fallback for orphan overrides, try to get it from custom_requirements_overrides
        # This is a bit less certain as 'processing_type' might be an override itself.
        if req_id_to_find in self.custom_requirements_overrides:
            return self.custom_requirements_overrides[req_id_to_find].get('processing_type')

        return None

        if not req_data_list:
            self.analysis_input_area.setMarkdown(f"# {title_to_display}\n\n分析项 **{title_to_display}** 尚无任何聚合笔记。")
            return

        sorted_snippets = sorted(req_data_list, key=lambda x: x.get('chapter_order', float('inf')))
        markdown_parts = [f"# {title_to_display}\n"]
        for snippet_data in sorted_snippets:
            chapter_title = snippet_data.get('chapter_title', '未知章节')
            safe_chapter_title = chapter_title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            snippet_text = snippet_data.get('snippet', '[无内容]').strip()
            chapter_order_display = snippet_data.get('chapter_order', 'N/A')
            markdown_parts.append(f"## 章节: {safe_chapter_title} (顺序: {chapter_order_display})\n")
            markdown_parts.append(f"```text\n{snippet_text}\n```\n")
        self.analysis_input_area.setMarkdown("\n".join(markdown_parts))

    def _find_req_data_recursive(self, items_list, req_id_to_find):
        for item_data in items_list:
            if item_data['id'] == req_id_to_find:
                return item_data
            if 'sub_items' in item_data and item_data['sub_items']:
                found = self._find_req_data_recursive(item_data['sub_items'], req_id_to_find)
                if found: return found
        return None

    def _find_any_req_data(self, req_id_to_find):
        # Precedence for data retrieval:
        # 1. Custom-created item data (from self.custom_created_requirements) is primary for custom items.
        # 2. Default item data (from REQUIREMENTS_STRUCTURE) is primary for default items.
        # 3. Overrides (from self.custom_requirements_overrides) can augment or replace fields from either of the above.
        #    - For 'title' and 'description', overrides always take precedence if they exist.
        #    - For 'processing_type':
        #        - Custom items: 'processing_type' is stored directly in custom_created_requirements. Overrides are not typically used for this.
        #        - Default items: 'processing_type' from REQUIREMENTS_STRUCTURE is the base. An override can specify a different type.
        #                       If the override for 'processing_type' is removed, it reverts to the base.

        if req_id_to_find in self.custom_created_requirements:
            # This is a custom-defined requirement item.
            custom_data = self.custom_created_requirements[req_id_to_find] # Base data for custom item
            # Overrides for title, description etc. can still apply to custom items.
            # processing_type for custom items is primarily stored in custom_data itself.
            # If custom_requirements_overrides also had a processing_type for a custom_req_id, it would override,
            # but current edit logic puts it in custom_created_requirements.
            item_data = {**custom_data} # Start with base custom data
            # Store original type before override, in case participation changes it
            original_processing_type = item_data.get('processing_type')

            if req_id_to_find in self.custom_requirements_overrides:
                item_data.update(self.custom_requirements_overrides[req_id_to_find]) # Apply overrides

            # Check participation status
            participates = self.custom_requirements_overrides.get(req_id_to_find, {}).get('participates_in_extraction', True)
            if not participates:
                item_data['original_processing_type'] = original_processing_type # Store the type it would have had
                item_data['processing_type'] = 'non_participating' # Mark as non-participating
            return item_data

        default_item_base = self._find_req_data_recursive(REQUIREMENTS_STRUCTURE, req_id_to_find)
        if default_item_base:
            # This is a default requirement item.
            # Base data comes from REQUIREMENTS_STRUCTURE.
            # Overrides (including for 'processing_type') are in self.custom_requirements_overrides.
            item_data = {**default_item_base} # Start with base default data
            original_processing_type = item_data.get('processing_type') # Get base processing type

            if req_id_to_find in self.custom_requirements_overrides:
                # Apply overrides. If 'processing_type' is in overrides, it will overwrite the base one.
                item_data.update(self.custom_requirements_overrides[req_id_to_find])
                # If override changed processing_type, that's the new "original" before participation check
                if 'processing_type' in self.custom_requirements_overrides[req_id_to_find]:
                    original_processing_type = self.custom_requirements_overrides[req_id_to_find]['processing_type']

            # Check participation status
            participates = self.custom_requirements_overrides.get(req_id_to_find, {}).get('participates_in_extraction', True)
            if not participates:
                item_data['original_processing_type'] = original_processing_type # Store the type it would have had
                item_data['processing_type'] = 'non_participating'
            return item_data

        if req_id_to_find in self.custom_requirements_overrides:
            # This case handles "orphan" overrides, where an item might have been a default item,
            # was overridden, and then somehow its base definition was removed from REQUIREMENTS_STRUCTURE.
            # Or, if a custom item was deleted but its overrides remained (though cleanup should prevent this).
            # It primarily returns what's in the overrides.
            self.logging_service.debug(f"_find_any_req_data: req_id '{req_id_to_find}' not in custom_created or default_structure, but found in overrides.")
            override_data = self.custom_requirements_overrides[req_id_to_find]
            # Construct a minimal item structure from override data.
            # 'processing_type' will be included if it's in override_data.
            return {'id': req_id_to_find,
                    'title': override_data.get('title', req_id_to_find), # Default title to req_id if not in override
                    'description': override_data.get('description','No description in orphan override.'),
                    'processing_type': override_data.get('processing_type'), # Will be None if not set
                    'parent_id': None, # Orphan overrides don't have structural parent/child info from base.
                    'sub_item_ids': [], # No sub-items known for orphan overrides from base.
                    **override_data # Ensure all other fields from override_data are included.
                    }
            # For orphan overrides, participation status also matters.
            item_data_for_orphan = {'id': req_id_to_find,
                                    'title': override_data.get('title', req_id_to_find),
                                    'description': override_data.get('description','No description in orphan override.'),
                                    'parent_id': None, 'sub_item_ids': [],
                                    **override_data}
            original_processing_type = item_data_for_orphan.get('processing_type') # Get its type before participation check

            participates = override_data.get('participates_in_extraction', True)
            if not participates:
                item_data_for_orphan['original_processing_type'] = original_processing_type
                item_data_for_orphan['processing_type'] = 'non_participating'
            return item_data_for_orphan

        self.logging_service.debug(f"_find_any_req_data: req_id '{req_id_to_find}' not found in any data source.")
        return None

    def handle_auto_find_snippet(self, req_id, chapter_order_index, chapter_title, snippet_text):
        if req_id not in self.analysis_data:
            self.analysis_data[req_id] = []
        new_entry = {
            'chapter_order': chapter_order_index,
            'chapter_title': chapter_title,
            'snippet': snippet_text.strip()
        }
        self.analysis_data[req_id].append(new_entry)
        self.logging_service.data_change("Analysis snippet added.", details={'req_id': req_id, 'chapter': chapter_title, 'snippet_length': len(snippet_text)})
        if self.current_selected_requirement_id == req_id:
            self.refresh_analysis_input_area_for_selected_req()
        if self.is_chapter_analysis_view_active:
            current_tree_item = self.chapter_tree.currentItem()
            if isinstance(current_tree_item, ChapterTreeItem):
                if current_tree_item.original_title == chapter_title:
                    self.display_chapter_specific_analysis(current_tree_item)

    def handle_auto_find_error(self, req_id, chapter_title, error_msg):
        error_to_display = f"分析项 '{req_id}' / 章节 '{chapter_title}' 智能查找失败: {error_msg}"
        self.logging_service.error(f"AutoFindTask error for req '{req_id}', chapter '{chapter_title}'.", details={'error_message': error_msg})
        self.status_label.setText(error_to_display[:150] + "...")

    def toggle_chapter_analysis_view(self):
        self.is_chapter_analysis_view_active = self.toggle_chapter_view_btn.isChecked()
        if self.is_chapter_analysis_view_active:
            self.toggle_chapter_view_btn.setText("查看原文")
            # When switching to "chapter analysis view", we might want to ensure
            # the main analysis area (right side) reflects that a chapter-specific
            # view is active in the content_display area (middle-right).
            # However, the main analysis area is driven by requirement selection.
            # This button primarily affects the content_display.
        else:
            self.toggle_chapter_view_btn.setText("查看章节大纲")
        self.refresh_content_display_area()

    def refresh_content_display_area(self):
        current_item = self.chapter_tree.currentItem()
        if not current_item:
            self.content_display.clear()
            if self.is_chapter_analysis_view_active:
                 self.toggle_chapter_view_btn.setChecked(False) # Uncheck if no chapter selected
                 self.is_chapter_analysis_view_active = False
                 self.toggle_chapter_view_btn.setText("查看章节大纲")
            return

        if self.is_chapter_analysis_view_active and isinstance(current_item, ChapterTreeItem):
            # This will now display analysis for the self.CHAPTER_OUTLINE_REQ_ID
            self.display_chapter_specific_analysis(current_item)
        else: # Display original chapter content
            if isinstance(current_item, ChapterTreeItem):
                self.content_display.setPlainText(current_item.content)
            elif hasattr(current_item, 'text'): # Fallback for non-ChapterTreeItems if any
                self.content_display.setPlainText(f"选中: {current_item.text(0)}\n(此类项目无正文内容可显示)")
            else:
                self.content_display.clear()


    def display_chapter_specific_analysis(self, chapter_item):
        # This function now specifically displays the "章节大纲" (self.CHAPTER_OUTLINE_REQ_ID)
        if not isinstance(chapter_item, ChapterTreeItem):
            self.content_display.setHtml("<p><i>请选择一个章节以查看其章节大纲。</i></p>")
            return

        self.content_display.clear()
        target_chapter_title = chapter_item.original_title
        safe_chapter_title_html = target_chapter_title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        req_id_for_outline = self.CHAPTER_OUTLINE_REQ_ID
        outline_req_data = self._find_any_req_data(req_id_for_outline)
        outline_title_text = outline_req_data.get('title', "章节大纲") if outline_req_data else "章节大纲"
        safe_outline_title_html = outline_title_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        html_parts = [f"<h1>章节: {safe_chapter_title_html} - {safe_outline_title_html}</h1>"]
        
        snippets_list = self.analysis_data.get(req_id_for_outline, [])
        chapter_specific_snippets_content = []

        for snippet_entry in snippets_list:
            if snippet_entry.get('chapter_title') == target_chapter_title:
                chapter_specific_snippets_content.append(snippet_entry.get('snippet', '').strip())
        
        if chapter_specific_snippets_content:
            full_content = "\n\n".join(chapter_specific_snippets_content) # Join if multiple snippets (ideally one)
            # Basic conversion for display: replace newlines with <br> and escape HTML
            escaped_content = full_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
            html_parts.append(f"<p>{escaped_content}</p>")
        else:
            html_parts.append(f"<hr><p><b>当前章节的“{safe_outline_title_html}”尚未生成或无内容。</b></p>")
            html_parts.append("<p><i>提示：可尝试通过“批处理分析”中的“智能查找章节片段并生成对应分析”功能，或针对此章节和“章节大纲”分析项进行单项处理来生成。</i></p>")

        self.content_display.setHtml("\n".join(html_parts))

    def start_full_novel_analysis(self):
        if self.batch_analyzer:
            self.logging_service.info("Full novel analysis requested via button.")
            self.batch_analyzer.start_full_analysis()

    def stop_processing(self):
        self.logging_service.info("Stop processing requested by user.")
        self.stop_batch_requested = True
        if hasattr(self, 'batch_analyzer') and self.batch_analyzer:
            self.batch_analyzer.request_stop()
        self.status_label.setText("停止请求已发送... 当前批处理任务将在完成后或下一检查点中止。")
        self.stop_btn.setEnabled(False)

    def open_manage_models_dialog(self):
        dialog = ManageModelsDialog(self, self.logging_service, self)
        dialog.exec_()

    def on_model_changed(self):
        try:
            current_text = self.model_combo.currentText()
            current_data = self.model_combo.currentData()
            if current_data and current_data in self.model_configs:
                config_data = self.model_configs[current_data]
                self.api_url_input.setText(config_data.get("url", ""))
            elif current_text and current_text not in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
                pass
        except Exception as e:
            self.logging_service.error("Error during model change.", details={'error': str(e)}, exc_info=True)
            # print(f"模型切换警告: {e}") # Replaced by logging

    def load_novel(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择小说文件", "", "文本文件 (*.txt);;所有文件 (*)")
        if not file_path:
            self.logging_service.info("Novel loading cancelled by user.")
            return
        self.logging_service.info(f"Attempting to load novel: {file_path}")
        self.status_label.setText("解析文件中...")
        QApplication.processEvents()
        try:
            encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'utf-16']
            content = None
            detected_encoding = None
            for encoding_attempt in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding_attempt) as f:
                        content = f.read()
                    detected_encoding = encoding_attempt
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                raise ValueError("无法解码文件。请确保文件编码是UTF-8, GBK, GB2312或UTF-16之一。")
            chapters = self.parse_chapters(content)
            self.book_data["title"] = os.path.splitext(os.path.basename(file_path))[0]
            self.book_data["file_path"] = file_path
            self.book_data["encoding"] = detected_encoding
            self.analysis_data = {}
            self.custom_requirements_overrides = {}
            self.populate_requirements_tree()
            self.build_chapter_tree(chapters)
            self.status_label.setText(
                f"已加载: {self.book_data['title']} (编码: {detected_encoding}) - 分析数据已重置")
            self.logging_service.info(f"Novel loaded: {self.book_data['title']}", details={'file_path': file_path, 'encoding': detected_encoding})
        except Exception as e:
            self.logging_service.error(f"Failed to load novel: {file_path}", details={'error': str(e)}, exc_info=True)
            QMessageBox.critical(self, "错误", f"文件加载失败: {str(e)}")
            self.status_label.setText("文件加载失败。")

    def _finalize_chapter_content(self, chapter_data, content_buffer):
        if chapter_data:
            chapter_data['content'] = '\n'.join(content_buffer).strip()
            chapter_data['word_count'] = len(chapter_data['content'])
        return []

    def _add_chapter_to_volume(self, volume_data, chapter_data):
        if volume_data and chapter_data:
            volume_data['chapters'].append(chapter_data)

    def _try_match_line(self, line, patterns):
        for i, pattern in enumerate(patterns):
            match = re.match(pattern, line)
            if match:
                return match, i < 2
        return None, False

    def parse_chapters(self, content):
        parsed_volumes = []
        lines = content.split('\n')
        current_volume_data = None
        current_chapter_data = None
        content_buffer = []
        for line_raw in lines:
            line = line_raw.strip()
            if not line:
                content_buffer.append('')
                continue
            match, is_volume_pattern = self._try_match_line(line, self.CHAPTER_PARSE_PATTERNS)
            if match:
                self._finalize_chapter_content(current_chapter_data, content_buffer)
                self._add_chapter_to_volume(current_volume_data, current_chapter_data)
                content_buffer = []
                if is_volume_pattern:
                    if current_volume_data and current_volume_data['chapters']:
                         parsed_volumes.append(current_volume_data)
                    elif current_volume_data and not current_volume_data['chapters'] and current_volume_data.get('title') != '正文':
                        parsed_volumes.append(current_volume_data)
                    current_volume_data = {'title': line, 'chapters': [], 'content': '', 'word_count': 0}
                    current_chapter_data = None
                else:
                    if not current_volume_data:
                        current_volume_data = {'title': '正文', 'chapters': [], 'content': '', 'word_count': 0}
                    current_chapter_data = {'title': line, 'content': '', 'word_count': 0}
            else:
                content_buffer.append(line_raw)
        self._finalize_chapter_content(current_chapter_data, content_buffer)
        self._add_chapter_to_volume(current_volume_data, current_chapter_data)
        if current_volume_data:
            if not (len(parsed_volumes) > 0 and current_volume_data['title'] == '正文' and not current_volume_data['chapters']):
                 if current_volume_data['chapters'] or current_volume_data['title'] != '正文' or not parsed_volumes:
                    parsed_volumes.append(current_volume_data)
        if not parsed_volumes and content.strip():
            return [{'title': '全文',
                     'chapters': [{'title': '内容', 'content': content.strip(), 'word_count': len(content.strip())}],
                     'content': '', 'word_count': len(content.strip())}]
        return parsed_volumes

    def build_chapter_tree(self, chapters):
        self.chapter_tree.setUpdatesEnabled(False)
        try:
            self.chapter_tree.clear()
            root_title = self.book_data.get("title", "未命名书籍")
            root_item = QTreeWidgetItem(self.chapter_tree, [root_title, ""])
            total_chapters_count = 0
            total_words_count = 0
            for volume_data in chapters:
                volume_words_count = sum(c['word_count'] for c in volume_data['chapters'])
                volume_item_text = f"{len(volume_data['chapters'])}章, {volume_words_count}字"
                volume_item = QTreeWidgetItem(root_item, [volume_data['title'], volume_item_text])
                for chapter_data in volume_data['chapters']:
                    chapter_tree_item_widget = ChapterTreeItem(
                        title=chapter_data['title'],
                        content=chapter_data['content'],
                        word_count=chapter_data['word_count'],
                        parent=volume_item
                    )
                    total_chapters_count += 1
                total_words_count += volume_words_count
            root_item.setText(1, f"{len(chapters)}卷, {total_chapters_count}章, {total_words_count}字")
            root_item.setExpanded(True)
            for i in range(root_item.childCount()):
                root_item.child(i).setExpanded(True)
        finally:
            self.chapter_tree.setUpdatesEnabled(True)

    def show_content(self, item):
        self.refresh_content_display_area()

    def toggle_display_mode(self):
        if hasattr(self, 'toggle_chapter_view_btn'):
            self.toggle_chapter_view_btn.click()
        else:
            current_item = self.chapter_tree.currentItem()
            if current_item:
                self.refresh_content_display_area()

    def get_current_model_name(self):
        current_data = self.model_combo.currentData()
        if current_data:
            return current_data
        return self.model_combo.currentText().strip()

    def validate_config(self):
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

    def handle_error(self, error_msg): # This seems like a legacy/unused error handler
        self.logging_service.error("Legacy handle_error called.", details={'error_message': error_msg})
        # print(f"ERROR_LEGACY_WORKER: {error_msg}")
        self.status_label.setText(f"处理错误: {error_msg[:150]}...")

    def _write_analysis_to_file_recursive(self, file_handle, items_list, current_indent_level, analysis_data_map, is_markdown=False):
        # Uses items_list which could be REQUIREMENTS_STRUCTURE or custom_created_requirements portions
        for base_item_data in items_list: # Renamed item_data to base_item_data to avoid confusion
            item_id = base_item_data['id']

            # Get effective data using _find_any_req_data
            effective_req_data = self._find_any_req_data(item_id)
            if not effective_req_data:
                self.logging_service.warning(f"Export: Could not find effective data for req_id {item_id}. Skipping.")
                continue

            title_to_display = effective_req_data.get('title', item_id)
            effective_processing_type = effective_req_data.get('processing_type')

            # --- Filtering for Markdown Export ---
            if is_markdown:
                if effective_processing_type == 'non_participating':
                    # Recursively call for sub_items even if parent is skipped
                    if 'sub_items' in base_item_data and base_item_data['sub_items']:
                         self._write_analysis_to_file_recursive(file_handle, base_item_data['sub_items'], current_indent_level, analysis_data_map, is_markdown)
                    continue # Skip this non-participating item entirely for Markdown

                # For chapter_specific items like "章节大纲", their content is handled in a separate section in export_markdown.
                # So, here we only write their title and a note, but not their snippets.
                # Their sub-items (if any conceptually, though usually not for chapter_specific) should still be processed.
                if effective_processing_type == 'chapter_specific' or item_id == self.CHAPTER_OUTLINE_REQ_ID:
                    heading_prefix = "#" * (current_indent_level + 1)
                    file_handle.write(f"{heading_prefix} {title_to_display} (ID: {item_id})\n\n")
                    file_handle.write(f"类型: `{effective_processing_type}`\n\n")
                    file_handle.write(f"_此类型（如章节大纲）的内容按章节导出，请参见报告中各章节的具体部分或“章节大纲汇总”部分。_\n\n")
                    if 'sub_items' in base_item_data and base_item_data['sub_items']:
                        self._write_analysis_to_file_recursive(file_handle, base_item_data['sub_items'], current_indent_level + 1, analysis_data_map, is_markdown)
                    continue # Move to next item after handling title and note for chapter_specific

            # --- Generic Indentation and Heading for both TXT and Markdown (unless filtered by MD above) ---
            indent_str = "    " * current_indent_level
            heading_prefix = "#" * (current_indent_level + 1)

            if is_markdown: # Write title for items that passed MD filters (prompt_only, aggregate)
                file_handle.write(f"{heading_prefix} {title_to_display} (ID: {item_id})\n\n")
            else: # TXT export writes all titles
                file_handle.write(f"{indent_str}{title_to_display} (ID: {item_id}):\n")

            # --- Content based on effective_processing_type ---
            if effective_processing_type == 'prompt_only':
                description = effective_req_data.get('description', '[无描述信息]')
                if is_markdown:
                    file_handle.write(f"类型: `设定参考 (Prompt Only)`\n\n")
                    file_handle.write(f"描述:\n```\n{description}\n```\n\n")
                else: # TXT output
                    file_handle.write(f"{indent_str}  类型: 设定参考 (Prompt Only)\n")
                    file_handle.write(f"{indent_str}  描述:\n{indent_str}    {description.replace(chr(10), chr(10) + indent_str + '    ')}\n\n")
            
            elif effective_processing_type == 'aggregate':
                if is_markdown:
                    file_handle.write(f"类型: `聚合分析 (Aggregate)`\n\n")
                else: # TXT output
                    file_handle.write(f"{indent_str}  类型: 聚合分析 (Aggregate)\n")
                
                snippet_list = analysis_data_map.get(item_id, [])
                if snippet_list:
                    sorted_snippets = sorted(snippet_list, key=lambda x: x.get('chapter_order', float('inf')))
                    if is_markdown: # Consolidated Markdown output for aggregate
                        consolidated_md_snippets = []
                        for snippet_entry in sorted_snippets:
                            chapter_title = snippet_entry.get('chapter_title', '未知章节')
                            snippet_text = snippet_entry.get('snippet', '[无内容]').strip()
                            # Add an extra newline after snippet_text for better spacing between chapter entries
                            consolidated_md_snippets.append(f"### 章节: {chapter_title}\n\n{snippet_text}\n")
                        file_handle.write("\n".join(consolidated_md_snippets) + "\n") # One \n before potential --- or next item.
                    else: # TXT output (original chapter-by-chapter)
                        for snippet_entry in sorted_snippets:
                            chapter_title = snippet_entry.get('chapter_title', '未知章节')
                            snippet_text = snippet_entry.get('snippet', '[无内容]').strip()
                            # chapter_order = snippet_entry.get('chapter_order', 'N/A') # Not used in TXT here
                            notes_indent = indent_str + "    "
                            file_handle.write(f"{notes_indent}章节: {chapter_title}\n")
                            indented_snippet = notes_indent + "    " + f'\n{notes_indent + "    "}'.join(snippet_text.split('\n'))
                            file_handle.write(f"{indented_snippet}\n\n")
                else: # No snippets
                    no_notes_message = "_此分析项无聚合笔记。_\n\n" if is_markdown else f"{indent_str}    [此分析项无聚合笔记]\n\n"
                    file_handle.write(no_notes_message)

            elif effective_processing_type == 'chapter_specific': # Already handled for MD, this is for TXT
                if not is_markdown: # TXT output for chapter_specific
                    file_handle.write(f"{indent_str}  类型: 章节大纲 (Chapter Specific)\n")
                    file_handle.write(f"{indent_str}  (章节大纲内容按章节导出，请参见后续的“章节大纲汇总”部分)\n\n")
            
            elif effective_processing_type == 'non_participating': # Already handled for MD, this is for TXT
                 if not is_markdown: # TXT output for non_participating
                    original_type = effective_req_data.get('original_processing_type', '未知')
                    file_handle.write(f"{indent_str}  类型: 未参与提取 (原类型: {original_type})\n\n")

            else: # Unknown or other types
                if is_markdown: # Should not be hit if MD filters are comprehensive
                    file_handle.write(f"_此分析项类型 ({effective_processing_type}) 未知或无特定导出格式。_\n\n")
                else: # TXT output
                    file_handle.write(f"{indent_str}    [此分析项类型 ({effective_processing_type}) 未知或无特定导出格式]\n\n")

            # Recursive call for sub-items (using base_item_data for structure)
            if 'sub_items' in base_item_data and base_item_data['sub_items']:
                self._write_analysis_to_file_recursive(file_handle, base_item_data['sub_items'], current_indent_level + 1, analysis_data_map, is_markdown)

            # Add a separator after a top-level item and its children have been processed in Markdown
            # The initial call from export_markdown uses current_indent_level = 1 for top-level items under "## 分析项层级结构与设定参考"
            if is_markdown and current_indent_level == 1:
                # Check if it's not a chapter_specific type that was just a title note, or a skipped non-participating.
                # This separator should ideally come after items that had actual content or were 'prompt_only'.
                if effective_processing_type != 'non_participating' and \
                   not (effective_processing_type == 'chapter_specific' or item_id == self.CHAPTER_OUTLINE_REQ_ID) : # Avoid adding --- for items that only print a note
                    file_handle.write("\n---\n\n")


    def export_results(self):
        if not self.default_export_path:
            reply = QMessageBox.question(self, '导出路径',
                                       '未设置默认导出路径，是否现在设置？',
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.set_export_path()
            else:
                return
        if not self.default_export_path:
            return
        try:
            safe_book_title = re.sub(r'[\/*?:"<>|]', '_', self.book_data.get("title", "Untitled_Novel"))
            if not safe_book_title.strip(): safe_book_title = "Untitled_Novel"
            book_dir = os.path.join(
                self.default_export_path, f"{safe_book_title}_结构分析报告")
            os.makedirs(book_dir, exist_ok=True)
            self.export_txt(book_dir)
            self.export_markdown(book_dir)
            self.export_json(book_dir)
            QMessageBox.information(self, "导出完成", f"结果已保存到:\n{book_dir}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出结果时发生错误: {str(e)}")

    def export_txt(self, output_directory_path):
        filename = os.path.join(output_directory_path, f"{self.book_data.get('title', 'Untitled')}_完整报告.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"小说: {self.book_data.get('title', 'N/A')} - 结构化分析报告\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Token消耗 (累计): 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}\n")
            f.write(f"{'=' * 70}\n\n")
            
            f.write("--- 分析项层级结构与设定参考 ---\n\n")
            self._write_analysis_to_file_recursive(f, REQUIREMENTS_STRUCTURE, 0, self.analysis_data, is_markdown=False)
            
            # Add section for chapter-specific analysis (e.g., 章节大纲)
            f.write(f"\n\n{'=' * 70}\n")
            f.write("--- 章节大纲汇总 ---\n\n")
            if self.chapter_tree.topLevelItemCount() > 0:
                book_root_item = self.chapter_tree.topLevelItem(0)
                for i in range(book_root_item.childCount()): # Volumes
                    vol_item = book_root_item.child(i)
                    f.write(f"卷: {vol_item.text(0)}\n")
                    for j in range(vol_item.childCount()): # Chapters
                        chap_item = vol_item.child(j)
                        if isinstance(chap_item, ChapterTreeItem):
                            f.write(f"  章节: {chap_item.original_title}\n")
                            snippets = self.analysis_data.get(self.CHAPTER_OUTLINE_REQ_ID, [])
                            found_for_chapter = False
                            for snippet_entry in snippets:
                                if snippet_entry.get('chapter_title') == chap_item.original_title:
                                    content = snippet_entry.get('snippet', '[无内容]').strip()
                                    indented_content = "    " + content.replace("\n", "\n    ")
                                    f.write(f"{indented_content}\n\n")
                                    found_for_chapter = True
                            if not found_for_chapter:
                                f.write(f"    [本章节无大纲信息]\n\n")
                    f.write("\n")

    def export_markdown(self, output_directory_path):
        filename = os.path.join(output_directory_path, f"{self.book_data.get('title', 'Untitled')}_完整报告.md")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# 小说: {self.book_data.get('title', 'N/A')} - 结构化分析报告\n\n")
            f.write(f"**生成时间:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Token消耗 (累计):** 输入 {self.total_tokens[0]} | 输出 {self.total_tokens[1]}\n\n---\n\n")
            
            f.write(f"## 分析项层级结构与设定参考\n\n")
            self._write_analysis_to_file_recursive(f, REQUIREMENTS_STRUCTURE, 1, self.analysis_data, is_markdown=True) # Start heading level at 2 (##)
            
            # Add section for chapter-specific analysis (e.g., 章节大纲)
            f.write(f"\n\n---\n## 章节大纲汇总\n\n")
            if self.chapter_tree.topLevelItemCount() > 0:
                book_root_item = self.chapter_tree.topLevelItem(0)
                for i in range(book_root_item.childCount()): # Volumes
                    vol_item = book_root_item.child(i)
                    f.write(f"### 卷: {vol_item.text(0)}\n\n") # H3 for Volume
                    for j in range(vol_item.childCount()): # Chapters
                        chap_item = vol_item.child(j)
                        if isinstance(chap_item, ChapterTreeItem):
                            f.write(f"#### 章节: {chap_item.original_title}\n\n") # H4 for Chapter
                            snippets = self.analysis_data.get(self.CHAPTER_OUTLINE_REQ_ID, [])
                            found_for_chapter = False
                            for snippet_entry in snippets:
                                if snippet_entry.get('chapter_title') == chap_item.original_title:
                                    content = snippet_entry.get('snippet', '[无内容]').strip()
                                    f.write(f"```text\n{content}\n```\n\n")
                                    found_for_chapter = True
                            if not found_for_chapter:
                                f.write(f"_本章节无大纲信息。_\n\n")
                    f.write("\n")

    def export_json(self, output_directory_path):
        # Collect prompt_only definitions
        prompt_only_definitions = []
        def collect_prompt_only_recursive(items_list, collector):
            for item_data in items_list:
                if item_data.get('processing_type') == 'prompt_only':
                    override_data = self.custom_requirements_overrides.get(item_data['id'], {})
                    title = override_data.get('title', item_data.get('title', item_data['id']))
                    description = override_data.get('description', item_data.get('description', ''))
                    collector.append({'id': item_data['id'], 'title': title, 'description': description})
                if 'sub_items' in item_data and item_data['sub_items']:
                    collect_prompt_only_recursive(item_data['sub_items'], collector)
        collect_prompt_only_recursive(REQUIREMENTS_STRUCTURE, prompt_only_definitions)

        # Separate analysis_data into aggregate and chapter_specific
        aggregate_analysis_results = {}
        chapter_specific_analysis_data = {self.CHAPTER_OUTLINE_REQ_ID: {}}

        for req_id, snippets_list in self.analysis_data.items():
            req_data = self._find_any_req_data(req_id) # To check processing_type
            processing_type = req_data.get('processing_type') if req_data else 'aggregate' # Default to aggregate if somehow not found

            if req_id == self.CHAPTER_OUTLINE_REQ_ID or processing_type == 'chapter_specific':
                for snippet_entry in snippets_list:
                    chapter_title = snippet_entry.get('chapter_title', '未知章节')
                    # Ensure chapter_title key exists
                    if chapter_title not in chapter_specific_analysis_data[req_id]:
                        chapter_specific_analysis_data[req_id][chapter_title] = []
                    chapter_specific_analysis_data[req_id][chapter_title].append(snippet_entry.get('snippet', ''))
            elif processing_type == 'aggregate':
                aggregate_analysis_results[req_id] = snippets_list
            # prompt_only items do not have analysis data in self.analysis_data

        data_to_export = {
            "title": self.book_data.get("title", "N/A"),
            "export_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "token_usage": {"input": self.total_tokens[0], "output": self.total_tokens[1]},
            "prompt_only_definitions": prompt_only_definitions,
            "aggregate_analysis_results": aggregate_analysis_results,
            "chapter_specific_analysis_data": chapter_specific_analysis_data,
            "custom_requirements_overrides": self.custom_requirements_overrides,
            # Optionally export the processed REQUIREMENTS_STRUCTURE if needed for full context
            # "requirements_structure_snapshot": REQUIREMENTS_STRUCTURE, 
            "novel_content_summary": [] # Keep this as it might be useful for other things
        }
        
        # Populate novel_content_summary (no change here, just for completeness)
        if self.chapter_tree.topLevelItemCount() > 0:
            book_item = self.chapter_tree.topLevelItem(0)
            if book_item:
                for i in range(book_item.childCount()):
                    vol_item = book_item.child(i)
                    volume_data_entry = {"title": vol_item.text(0), "chapters": []}
                    for j in range(vol_item.childCount()):
                        chap_item = vol_item.child(j)
                        if isinstance(chap_item, ChapterTreeItem):
                            volume_data_entry["chapters"].append({
                                "title": chap_item.original_title,
                                "original_length": chap_item.word_count,
                            })
                    if volume_data_entry["chapters"]:
                        data_to_export["novel_content_summary"].append(volume_data_entry)
        
        filename = os.path.join(output_directory_path, f"{self.book_data.get('title', 'Untitled')}_分析数据.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_export, f, ensure_ascii=False, indent=2)

    def set_export_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择默认导出目录")
        if path:
            self.default_export_path = path
            self.status_label.setText(f"导出目录设置为: {path}")

    def add_custom_model(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("添加自定义模型")
        dialog.setModal(True)
        layout = QFormLayout(dialog)
        name_input = QLineEdit(dialog)
        name_input.setPlaceholderText("例如: my-custom-model (唯一标识)")
        layout.addRow("模型ID (唯一):", name_input)
        display_name_input = QLineEdit(dialog)
        display_name_input.setPlaceholderText("例如: 我的自定义模型")
        layout.addRow("显示名称:", display_name_input)
        url_input = QLineEdit(dialog)
        url_input.setPlaceholderText("https://api.example.com/v1/chat/completions")
        layout.addRow("API地址:", url_input)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec_() == QDialog.Accepted:
            model_id = name_input.text().strip()
            display_name = display_name_input.text().strip()
            api_url = url_input.text().strip()
            if model_id and display_name and api_url:
                if model_id in self.model_configs:
                    QMessageBox.warning(self, "错误", f"模型ID '{model_id}' 已存在。")
                    return
                self.model_configs[model_id] = {"url": api_url, "display_name": display_name}
                self.model_combo.addItem(display_name, model_id)
                new_model_index = self.model_combo.findData(model_id)
                if new_model_index != -1:
                    self.model_combo.setCurrentIndex(new_model_index)
                self.api_url_input.setText(api_url)
                self.logging_service.info("Custom model added.", details={'model_id': model_id, 'display_name': display_name, 'url': api_url})
                QMessageBox.information(self, "成功", f"已添加自定义模型: {display_name}")
            else:
                self.logging_service.error("Failed to add custom model due to missing fields.", details={'model_id': model_id, 'display_name': display_name, 'url': api_url})
                QMessageBox.warning(self, "错误", "请填写模型ID、显示名称和API地址。")

    def save_config(self, silent=False):
        if silent and not self.book_data.get("file_path"):
            return
        config_to_save = self._get_current_config_for_saving()
        try:
            with open("config.json", 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, ensure_ascii=False, indent=2)
            self.logging_service.info("Configuration saved.", details={'path': 'config.json', 'silent': silent})
            if not silent:
                QMessageBox.information(self, "成功", "配置已保存")
            elif hasattr(self, 'status_label'):
                self.status_label.setText("配置已自动保存")
        except Exception as e:
            self.logging_service.error("Failed to save config.", details={'path': 'config.json', 'error': str(e)}, exc_info=True)
            if not silent:
                QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")
            else:
                print(f"Error during silent save_config: {e}")

    def get_chapter_states(self):
        chapter_states_list = []
        if self.chapter_tree.topLevelItemCount() == 0:
            return chapter_states_list
        book_item = self.chapter_tree.topLevelItem(0)
        if not book_item: return chapter_states_list
        for i in range(book_item.childCount()):
            vol_item = book_item.child(i)
            for j in range(vol_item.childCount()):
                chap_item = vol_item.child(j)
                if isinstance(chap_item, ChapterTreeItem):
                    chapter_states_list.append({
                        "path": [vol_item.text(0), chap_item.original_title],
                        "content": chap_item.content,
                        "word_count": chap_item.word_count
                    })
        return chapter_states_list

    def _apply_loaded_config_to_ui(self, config_data):
        if "custom_models" in config_data:
            for model_key, model_info in config_data["custom_models"].items():
                if model_key not in self.initial_model_keys:
                    self.model_configs[model_key] = model_info
                    self.model_combo.addItem(model_info.get("display_name", model_key), model_key)
        model_to_select = config_data.get("model")
        if model_to_select:
            idx = self.model_combo.findData(model_to_select)
            if idx == -1:
                idx = self.model_combo.findText(model_to_select)
            if idx != -1:
                self.model_combo.setCurrentIndex(idx)
        self.api_url_input.setText(config_data.get("api_url", ""))
        self.api_key_input.setText(config_data.get("api_key", ""))

    def load_config(self, silent=False):
        config_path = "config.json"
        if not os.path.exists(config_path):
            self.logging_service.info("Config file not found.", details={'path': config_path})
            if not silent:
                QMessageBox.information(self, "提示", "未找到配置文件。")
            return
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            self.default_export_path = config_data.get("export_path", "")
            self.custom_requirements_overrides = config_data.get("custom_requirements_overrides", {})
            self.custom_created_requirements = config_data.get("custom_created_requirements", {})
            self._apply_loaded_config_to_ui(config_data)
            reloaded_ok = False
            if "book_data" in config_data and config_data["book_data"].get("file_path"):
                self.book_data = config_data["book_data"]
                self.analysis_data = self.book_data.get("analysis", {})
                reloaded_ok = self.reload_novel()
            if ("custom_requirements_overrides" in config_data or reloaded_ok) and self._is_ui_ready :
                 self.populate_requirements_tree()
                 if self.current_selected_requirement_id:
                     self.refresh_analysis_input_area_for_selected_req()
            status_msg = "配置已加载"
            if reloaded_ok and self.book_data.get('title'):
                status_msg += f", 上次打开: {self.book_data['title']}"
            if not silent:
                QMessageBox.information(self, "成功", status_msg)
            elif hasattr(self, 'status_label'):
                self.status_label.setText(status_msg)
            self.logging_service.info("Configuration loaded.", details={'path': config_path, 'reloaded_novel': reloaded_ok})
        except Exception as e:
            self.logging_service.error("Failed to load config.", details={'path': config_path, 'error': str(e)}, exc_info=True)
            if not silent:
                QMessageBox.critical(self, "错误", f"加载配置失败: {str(e)}")
            else:
                 print(f"Error during silent load_config: {e}")

    def reload_novel(self):
        file_path = self.book_data.get("file_path")
        encoding = self.book_data.get("encoding")
        if not file_path or not os.path.exists(file_path) or not encoding:
            self.book_data = {"title": "", "volumes": []}
            self.analysis_data = {}
            self.custom_requirements_overrides = {}
            self.populate_requirements_tree()
            self.chapter_tree.clear()
            return False
        if hasattr(self, 'status_label'):
            self.status_label.setText(
                f"重新加载: {self.book_data.get('title', '未知')}...")
        QApplication.processEvents()
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            chapters = self.parse_chapters(content)
            self.build_chapter_tree(chapters)
            if hasattr(self, 'status_label'):
                self.status_label.setText(
                    f"已重新加载: {self.book_data.get('title', '未知')} (编码: {encoding})")
            if self.current_selected_requirement_id:
                self.refresh_analysis_input_area_for_selected_req()
            return True
        except Exception as e:
            self.book_data = {"title": "", "volumes": []}
            self.analysis_data = {}
            self.custom_requirements_overrides = {}
            self.populate_requirements_tree()
            self.chapter_tree.clear()
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"重新加载失败: {str(e)}")
            return False

    def _find_chapter_item_by_path_titles(self, book_item_node, path_titles_list):
        if not book_item_node or not path_titles_list or len(path_titles_list) != 2:
            return None
        target_vol_title, target_chap_original_title = path_titles_list
        for i in range(book_item_node.childCount()):
            vol_item_node = book_item_node.child(i)
            if vol_item_node.text(0) == target_vol_title:
                for j in range(vol_item_node.childCount()):
                    chap_item_node = vol_item_node.child(j)
                    if isinstance(chap_item_node, ChapterTreeItem) and \
                       hasattr(chap_item_node, 'original_title') and \
                       chap_item_node.original_title == target_chap_original_title:
                        return chap_item_node
                return None
        return None

    def restore_chapter_states(self, saved_chapter_states):
        if self.chapter_tree.topLevelItemCount() == 0: return
        book_item_node = self.chapter_tree.topLevelItem(0)
        if not book_item_node: return
        self.chapter_tree.setUpdatesEnabled(False)
        try:
            for state_data in saved_chapter_states:
                path_titles = state_data.get("path")
                chapter_tree_item = self._find_chapter_item_by_path_titles(book_item_node, path_titles)
                if chapter_tree_item:
                    if "content" in state_data:
                        chapter_tree_item.content = state_data.get("content", chapter_tree_item.content)
                    if "word_count" in state_data:
                        chapter_tree_item.word_count = state_data.get("word_count", len(chapter_tree_item.content))
                    chapter_tree_item.setText(1, f"{chapter_tree_item.word_count}字")
        finally:
            self.chapter_tree.setUpdatesEnabled(True)

    def test_connection(self):
        if not self.validate_config():
            return
        try:
            api_config = {"url": self.api_url_input.text().strip(),
                          "key": self.api_key_input.text().strip(),
                          "model": self.get_current_model_name()}
            encoding_object = self.get_tiktoken_encoding(api_config['model'])
            if encoding_object is None:
                err_msg = f"无法为模型 '{api_config.get('model','未知')}' 初始化Token编码器。测试中止。"
                self.logging_service.error("API connection test: Encoding init failed.", details={'model': api_config.get('model'), 'error': err_msg})
                QMessageBox.critical(self, "编码器错误", err_msg)
                return

            self.logging_service.info("Attempting API connection test.", details={'model': api_config['model'], 'url': api_config['url']})
            processor = LLMProcessor(api_config, "", encoding_object)
            self.status_label.setText("正在测试连接...")
            QApplication.processEvents()

            test_text = "这是一个连接测试。"
            self.logging_service.api_request("Connection test: summarize call.", details={'model': api_config['model'], 'text_length': len(test_text)})
            summary, input_tokens, output_tokens = processor.summarize(test_text, max_retries=1)
            self.logging_service.api_response("Connection test: summarize response.", details={'model': api_config['model'], 'input_tokens': input_tokens, 'output_tokens': output_tokens, 'summary_length': len(summary) if summary else 0})

            if summary is not None:
                self.logging_service.info("API connection test successful.")
                QMessageBox.information(self, "连接成功", f"API连接测试成功！\n模型: {api_config['model']}\n返回: {summary[:100]}...")
                self.status_label.setText("连接测试成功")
            else:
                self.logging_service.error("API connection test failed: API returned None summary.", details={'model': api_config['model']})
                raise ValueError("API返回内容为空或无效 (None)")
        except Exception as e:
            self.logging_service.error("API connection test failed.", details={'model': api_config.get('model'), 'url': api_config.get('url'), 'error': str(e)}, exc_info=True)
            QMessageBox.critical(self, "连接失败", f"API连接测试失败: {str(e)}")
            self.status_label.setText("连接测试失败")

    def auto_save(self):
        self.save_config(silent=True)

    def auto_export_novel_data(self):
        if not self.book_data.get("title"):
            return
        book_title_safe = re.sub(r'[\/*?:"<>|]', "_", self.book_data["title"])
        if not book_title_safe.strip():
            book_title_safe = "Untitled_Novel"
        export_path = os.path.join(self.auto_export_base_dir, f"{book_title_safe}_结构分析报告")
        try:
            os.makedirs(export_path, exist_ok=True)
            self.export_markdown(export_path)
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"'{book_title_safe}' 分析报告已自动保存。")
        except Exception as e:
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"自动导出分析报告失败: {str(e)}")

    def save_analysis_as(self):
        if not self.analysis_data:
            QMessageBox.information(self, "提示", "没有分析数据可供保存。")
            return

        suggested_filename = f"{self.book_data.get('title', 'Untitled_Novel')}_完整分析报告"

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "另存分析报告",
            suggested_filename,
            "Markdown 文件 (*.md);;文本文件 (*.txt)"
        )

        if not file_path: # User cancelled
            return

        is_markdown = "Markdown" in selected_filter

        try:
            report_content = self._generate_full_analysis_report_string(is_markdown)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            QMessageBox.information(self, "成功", f"分析报告已保存到:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存分析报告时发生错误: {str(e)}")
            print(f"Error saving analysis report: {e}")

    def _generate_full_analysis_report_string(self, is_markdown=False):
        lines = []
        novel_title = self.book_data.get("title", "未命名小说")

        if is_markdown:
            lines.append(f"# 分析报告: {novel_title}\n\n")
            lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        else:
            lines.append(f"分析报告: {novel_title}\n")
            lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            lines.append("=" * 50 + "\n\n")

        def append_requirement_data(req_items_list, current_heading_level_or_indent_factor):
            for req_item_data in req_items_list:
                req_id = req_item_data['id']
                override = self.custom_requirements_overrides.get(req_id, {})
                req_title = override.get('title', req_item_data['title'])
                snippets_for_this_req = sorted(
                    self.analysis_data.get(req_id, []),
                    key=lambda x: x.get('chapter_order', float('inf'))
                )
                if is_markdown:
                    lines.append(f"{'#' * current_heading_level_or_indent_factor} {req_title}\n\n")
                else:
                    indent = "    " * (current_heading_level_or_indent_factor -1)
                    lines.append(f"{indent}{req_title}:\n\n")
                if snippets_for_this_req:
                    for snippet_data in snippets_for_this_req:
                        chap_title = snippet_data.get('chapter_title', '未知章节')
                        snippet_text = snippet_data.get('snippet', '[无内容]').strip()
                        chap_order_display = snippet_data.get('chapter_order', 'N/A')
                        if is_markdown:
                            lines.append(f"### 来自章节: {chap_title} (顺序: {chap_order_display})\n")
                            lines.append(f"```text\n{snippet_text}\n```\n\n")
                        else:
                            notes_indent = indent + "    "
                            lines.append(f"{notes_indent}--- 章节: {chap_title} (顺序: {chap_order_display}) ---\n")
                            indented_snippet = (notes_indent + "    ") + snippet_text.replace("\n", f"\n{notes_indent + '    '}")
                            lines.append(f"{indented_snippet}\n\n")
                else:
                    if is_markdown:
                        lines.append("_此分析项无笔记。_\n\n")
                    else:
                        lines.append(f"{indent + '    '}[此分析项无笔记。]\n\n")
                if 'sub_items' in req_item_data and req_item_data['sub_items']:
                    append_requirement_data(req_item_data['sub_items'], current_heading_level_or_indent_factor + 1)
        append_requirement_data(REQUIREMENTS_STRUCTURE, 2 if is_markdown else 1)
        return "".join(lines)

    def save_chapter_edits(self):
        pass

    def get_tiktoken_encoding(self, model_name_from_config: str):
        effective_encoding_key = None
        try:
            _model_key_for_tiktoken = model_name_from_config.split('/')[-1].lower()
            encoding_map = {
                'gpt-4': 'cl100k_base', 'gpt-3.5-turbo': 'cl100k_base',
                'deepseek-chat': 'cl100k_base', 'qwen': 'cl100k_base',
                'chatglm': 'cl100k_base'
            }
            try:
                effective_encoding_key = _model_key_for_tiktoken
                with self.tiktoken_cache_lock:
                    if effective_encoding_key in self.tiktoken_encoding_cache:
                        return self.tiktoken_encoding_cache[effective_encoding_key]
                    try:
                        encoding_obj = tiktoken.encoding_for_model(effective_encoding_key)
                        self.tiktoken_encoding_cache[effective_encoding_key] = encoding_obj
                        return encoding_obj
                    except KeyError: pass
                derived_encoding_key = None
                for prefix, base_encoding_name in encoding_map.items():
                    if _model_key_for_tiktoken.startswith(prefix):
                        derived_encoding_key = base_encoding_name
                        break
                if not derived_encoding_key: derived_encoding_key = 'cl100k_base'
                effective_encoding_key = derived_encoding_key
                with self.tiktoken_cache_lock:
                    if effective_encoding_key in self.tiktoken_encoding_cache:
                        return self.tiktoken_encoding_cache[effective_encoding_key]
                    encoding_obj = tiktoken.get_encoding(effective_encoding_key)
                    self.tiktoken_encoding_cache[effective_encoding_key] = encoding_obj
                    return encoding_obj
            except Exception as e:
                self.logging_service.error(f"Tiktoken encoding failed for '{model_name_from_config}'. Using fallback.", details={'error': str(e)}, exc_info=True)
                # print(f"ERROR: Failed to get/create tiktoken encoding for '{model_name_from_config}'. Error: {e}. Using fallback 'cl100k_base'.") # Replaced by logging
                with self.tiktoken_cache_lock:
                    if 'cl100k_base' in self.tiktoken_encoding_cache:
                        return self.tiktoken_encoding_cache['cl100k_base']
                    try:
                        encoding_obj = tiktoken.get_encoding('cl100k_base')
                        self.tiktoken_encoding_cache['cl100k_base'] = encoding_obj
                        return encoding_obj
                    except Exception as e_default:
                        self.logging_service.error(f"CRITICAL: Failed to get default tiktoken encoder 'cl100k_base'.", details={'error': str(e_default)}, exc_info=True)
                        # print(f"CRITICAL ERROR: Failed to get default tiktoken encoder 'cl100k_base': {e_default}") # Replaced by logging
                        return None
        except Exception as e_outer:
             self.logging_service.error(f"CRITICAL: Outer error in get_tiktoken_encoding for '{model_name_from_config}'.", details={'error': str(e_outer)}, exc_info=True)
             # print(f"CRITICAL ERROR in get_tiktoken_encoding for '{model_name_from_config}': {e_outer}") # Replaced by logging
             return None

    def closeEvent(self, event):
        try:
            self.save_config(silent=True) # Ensure config is saved on close
        except Exception as e:
            self.logging_service.error("Error saving config on close.", details={'error': str(e)}, exc_info=True)
            # print(f"Error saving config on close: {e}") # Replaced by logging
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("小说智能分析工具")
    app.setApplicationVersion("2.3")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
