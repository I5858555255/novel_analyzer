# dialogs.py
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QTreeWidget,
                             QTreeWidgetItem, QPushButton, QMessageBox,
                             QHeaderView, QLabel)
# QLabel might be needed if errors are shown directly in dialog, or for general use.

class ManageModelsDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window # To access main_window.model_configs, etc.
        self.setWindowTitle("管理自定义模型")
        self.setMinimumSize(600, 400) # Set a reasonable minimum size

        layout = QVBoxLayout(self)

        # Optional: Add a label for instruction
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
            # Optional: Display a message if no custom models exist
            item = QTreeWidgetItem(self.models_list_widget, ["没有自定义模型可管理。"])
            self.models_list_widget.setEnabled(False)


    def handle_remove_model(self):
        button_clicked = self.sender()
        if not button_clicked:
            return

        model_key_to_remove = button_clicked.property("model_key_to_remove")
        if not model_key_to_remove:
            return

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
                        self.main_window.status_label.setText(f"活动模型 '{model_display_name}' 已移除。")
                    combo.removeItem(i)
                    break

            self.main_window.status_label.setText(f"自定义模型 '{model_display_name}' 已移除。")
            self.populate_models_list() # Refresh the list in the dialog
