"""
Custom dialog definitions for the Novel Analyzer application.

This module contains QDialog subclasses used for user interactions like
managing custom models or other specific settings.
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem,
    QPushButton, QMessageBox, QHeaderView, QTableView, QAbstractItemView,
    QComboBox, QCheckBox
)
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QColor
from PyQt5.QtCore import Qt
# MainWindow is passed as an argument to __init__, not imported directly for class definition.
# No other external project modules are directly needed by ManageModelsDialog itself.
# LogLevel will be passed as a class to LogViewerDialog __init__.


class ManageModelsDialog(QDialog):
    """
    A dialog window for managing user-added custom LLM models.

    This dialog displays a list of custom models and allows users to remove them.
    It interacts with the main window instance to access and modify the application's
    model configurations.
    """

    def __init__(self, main_window, logging_service, parent=None):
        """
        Initializes the ManageModelsDialog.

        Args:
            main_window: A reference to the main application window (MainWindow).
                         This is used to access and modify shared model configurations
                         and update UI elements on the main window.
            logging_service: Instance of the LoggingService.
            parent (QWidget, optional): The parent widget of this dialog.
                                        Defaults to None.
        """
        super().__init__(parent)
        self.main_window = main_window  # Instance of MainWindow from novel_analyzer
        self.logging_service = logging_service
        self.setWindowTitle("管理自定义模型")
        self.setMinimumSize(600, 400)

        # Main layout for the dialog
        layout = QVBoxLayout(self)

        # Informational label
        info_label = QLabel("以下是您添加的自定义模型。预定义模型无法在此处移除。")
        layout.addWidget(info_label)

        # Tree widget to display models
        self.models_list_widget = QTreeWidget()
        self.models_list_widget.setHeaderLabels(
            ["模型显示名称", "模型ID", "API 地址", "操作"])
        self.models_list_widget.header().setSectionResizeMode(
            0, QHeaderView.Stretch)  # Stretch display name column
        self.models_list_widget.header().setSectionResizeMode(
            2, QHeaderView.Stretch)  # Stretch API address column
        layout.addWidget(self.models_list_widget)

        # Close button
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.accept)  # QDialog.accept() closes the dialog
        layout.addWidget(self.close_button)

        self.setLayout(layout)
        self.populate_models_list()  # Initial population of the list

    def populate_models_list(self):
        """
        Populates the tree widget with the list of custom models.

        Clears the existing list and adds items for each custom model found
        in the main window's configuration. Predefined models are not shown here.
        """
        self.models_list_widget.clear()
        custom_models_found = False
        # Accessing main_window attributes like model_configs and initial_model_keys
        for model_key, config_data in self.main_window.model_configs.items():
            # Only show models that are not part of the initial (default) set
            if model_key not in self.main_window.initial_model_keys:
                custom_models_found = True
                display_name = config_data.get("display_name", model_key)
                url = config_data.get("url", "N/A")
                tree_item = QTreeWidgetItem(self.models_list_widget, [
                                            display_name, model_key, url])

                # Add a remove button for each custom model
                remove_button = QPushButton("移除")
                remove_button.setProperty("model_key_to_remove", model_key)
                remove_button.clicked.connect(self.handle_remove_model)
                self.models_list_widget.setItemWidget(
                    tree_item, 3, remove_button)  # Add button to the 4th column

        if not custom_models_found:
            # Display a message if no custom models exist
            item = QTreeWidgetItem(
                self.models_list_widget, ["没有自定义模型可管理。"])
            self.models_list_widget.setEnabled(False) # Disable list if empty
        else:
            self.models_list_widget.setEnabled(True)


    def handle_remove_model(self):
        """
        Handles the action of removing a custom model.

        Triggered when a "移除" (Remove) button is clicked for a model.
        It confirms the removal with the user and updates the main window's
        configuration and UI if confirmed.
        """
        button_clicked = self.sender()  # Get the button that was clicked
        if not button_clicked:
            return

        model_key_to_remove = button_clicked.property("model_key_to_remove")
        if not model_key_to_remove:
            return

        model_display_name = self.main_window.model_configs.get(
            model_key_to_remove, {}).get("display_name", model_key_to_remove)

        # Confirm removal with a message box
        reply = QMessageBox.question(self, "确认移除",
                                     f"确定要移除自定义模型 '{model_display_name}' ({model_key_to_remove}) 吗？此操作无法撤销。",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Remove from main configuration
            if model_key_to_remove in self.main_window.model_configs:
                del self.main_window.model_configs[model_key_to_remove]

            # Remove from the main window's model selection QComboBox
            combo = self.main_window.model_combo
            for i in range(combo.count()):
                if combo.itemData(i) == model_key_to_remove:
                    if combo.currentIndex() == i: # If removed model was selected
                        combo.setCurrentIndex(-1) # Clear selection
                        self.main_window.api_url_input.clear() # Clear API URL field
                    combo.removeItem(i)
                    break

            # Update status bar on main window
            if hasattr(self.main_window, 'status_label'):
                self.main_window.status_label.setText(
                    f"自定义模型 '{model_display_name}' 已移除。")

            self.logging_service.info("Custom model removed.", details={'model_id': model_key_to_remove, 'display_name': model_display_name})
            self.populate_models_list() # Refresh the list in this dialog


class LogViewerDialog(QDialog):
    def __init__(self, logging_service, log_level_class, parent=None):
        super().__init__(parent)
        self.logging_service = logging_service
        self.log_level_class = log_level_class # Store the passed LogLevel class
        self.setWindowTitle("系统日志查看器")
        self.setGeometry(150, 150, 1000, 700) # x, y, width, height

        layout = QVBoxLayout(self)

        # Filter controls
        filter_layout = QHBoxLayout()
        self.level_filter_combo = QComboBox()
        self.level_filter_combo.addItem("所有级别", "ALL_LEVELS") # Special value for all levels

        # Populate combo box from the LogLevel class attributes
        if self.log_level_class:
            for attr_name in dir(self.log_level_class):
                if not attr_name.startswith('_') and isinstance(getattr(self.log_level_class, attr_name), str):
                    level_display_name = getattr(self.log_level_class, attr_name)
                    self.level_filter_combo.addItem(level_display_name, level_display_name)

        self.level_filter_combo.currentIndexChanged.connect(self.populate_log_table)
        filter_layout.addWidget(self.level_filter_combo)

        self.auto_refresh_checkbox = QCheckBox("打开时自动刷新")
        self.auto_refresh_checkbox.setChecked(True)
        filter_layout.addWidget(self.auto_refresh_checkbox)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Log table
        self.log_table = QTableView()
        self.log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setWordWrap(True) # Enable word wrap for better readability

        self.model = QStandardItemModel(0, 4, self)
        self.model.setHorizontalHeaderLabels(["时间戳", "级别", "消息", "详情"])
        self.log_table.setModel(self.model)

        horizontal_header = self.log_table.horizontalHeader()
        horizontal_header.setSectionResizeMode(0, QHeaderView.ResizeToContents) # Timestamp
        horizontal_header.setSectionResizeMode(1, QHeaderView.ResizeToContents) # Level
        horizontal_header.setSectionResizeMode(2, QHeaderView.Stretch) # Message
        horizontal_header.setSectionResizeMode(3, QHeaderView.Stretch) # Details
        self.log_table.setColumnWidth(0, 170) # Timestamp width
        self.log_table.setColumnWidth(1, 100) # Level width
        # Message and Details will stretch, but details can have a good initial proportion
        self.log_table.setColumnWidth(3, 350)


        layout.addWidget(self.log_table)

        # Action buttons
        button_layout = QHBoxLayout()
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self.populate_log_table)
        button_layout.addWidget(self.refresh_button)

        self.clear_button = QPushButton("清除内存日志")
        self.clear_button.clicked.connect(self.clear_logs_action)
        button_layout.addWidget(self.clear_button)

        button_layout.addStretch()
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        self.populate_log_table()

    def populate_log_table(self):
        self.model.removeRows(0, self.model.rowCount())

        selected_filter_value = self.level_filter_combo.currentData()
        level_to_filter = None
        if selected_filter_value != "ALL_LEVELS":
            level_to_filter = selected_filter_value

        logs = self.logging_service.get_logs(level_filter=level_to_filter)

        for log_entry in logs:
            timestamp_str = log_entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            level_str = str(log_entry['level'])
            message_str = str(log_entry['message'])

            details_obj = log_entry.get('details', '')
            if isinstance(details_obj, dict):
                # Simple conversion for dict, can be improved for complex dicts
                details_parts = []
                if 'traceback' in details_obj: # Prioritize traceback if present
                    details_parts.append(f"TRACEBACK: {details_obj['traceback'][:300]}...") # Keep it manageable
                for k,v in details_obj.items():
                    if k != 'traceback':
                         details_parts.append(f"{k}: {str(v)[:100]}") # Truncate long values
                details_str = "; ".join(details_parts)
            else:
                details_str = str(details_obj)

            row_items = [
                QStandardItem(timestamp_str),
                QStandardItem(level_str),
                QStandardItem(message_str),
                QStandardItem(details_str)
            ]

            # Basic color coding
            color = None
            if self.log_level_class: # Check if LogLevel class was provided
                if level_str == getattr(self.log_level_class, "ERROR", ""): color = QColor(Qt.red)
                elif level_str == getattr(self.log_level_class, "API_REQUEST", ""): color = QColor(Qt.blue)
                elif level_str == getattr(self.log_level_class, "API_RESPONSE", ""): color = QColor("darkblue") # Using string for QColor
                elif level_str == getattr(self.log_level_class, "DATA_CHANGE", ""): color = QColor(Qt.darkGreen)
                elif level_str == getattr(self.log_level_class, "DEBUG", ""): color = QColor(Qt.gray)

            if color:
                for item in row_items:
                    item.setForeground(color)

            self.model.appendRow(row_items)

        if self.model.rowCount() > 0:
            self.log_table.resizeRowsToContents() # Adjust row heights after populating
            self.log_table.scrollToBottom()


    def clear_logs_action(self):
        self.logging_service.clear_logs()
        self.populate_log_table()

    def showEvent(self, event):
        super().showEvent(event)
        if self.auto_refresh_checkbox.isChecked():
            self.populate_log_table()
