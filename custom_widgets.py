"""
Custom Qt Widget definitions for the Novel Analyzer application.

This module contains custom widget classes derived from PyQt5 widgets,
specifically tailored for the application's UI needs, such as specialized
tree items.
"""
from PyQt5.QtWidgets import QTreeWidgetItem


class ChapterTreeItem(QTreeWidgetItem):
    """
    A custom QTreeWidgetItem to represent a chapter in the novel's structure.

    This item stores the chapter's title, full content, word count, summary,
    and summarization status. It also handles updating its display text
    to reflect whether it has been summarized.
    """

    def __init__(self, title, content, word_count, parent=None):
        """
        Initializes a ChapterTreeItem.

        Args:
            title (str): The original title of the chapter.
            content (str): The full text content of the chapter.
            word_count (int): The word count of the chapter.
            parent (QTreeWidgetItem, optional): The parent item in the tree.
                                                 Defaults to None.
        """
        super().__init__(parent, [title, f"{word_count}字"])
        self.original_title = title
        self.content = content
        self.word_count = word_count
        # Obsolete attributes (summary, is_summarized, summary_timestamp) already removed.

    def update_display_text(self):
        """
        Ensures the display text is the original title.
        This might be redundant if title is only set at init and never changed by other logic,
        but it's a safeguard.
        """
        if self.text(0) != self.original_title:
            self.setText(0, self.original_title)
