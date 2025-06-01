# custom_widgets.py
from PyQt5.QtWidgets import QTreeWidgetItem # Ensure this import is present

class ChapterTreeItem(QTreeWidgetItem):
    """自定义树节点存储章节数据"""
    def __init__(self, title, content, word_count, parent=None):
        super().__init__(parent, [title, f"{word_count}字"])
        self.original_title = title # Store original title without markers
        self.content = content
        self.summary = ""
        self.word_count = word_count
        self.is_summarized = False
        self.summary_timestamp = 0

    def update_display_text(self):
        """Updates the chapter title in the tree to reflect summary status."""
        # current_display_text = self.text(0) # Not needed for logic
        marker = "* "
        expected_text_if_summarized = marker + self.original_title
        expected_text_if_not_summarized = self.original_title

        if self.is_summarized:
            if self.text(0) != expected_text_if_summarized:
                self.setText(0, expected_text_if_summarized)
        else:
            if self.text(0) != expected_text_if_not_summarized:
                self.setText(0, expected_text_if_not_summarized)
