import sys
from PyQt5.QtWidgets import QApplication
from gui.novel_analyzer import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("小说智能分析工具")
    app.setApplicationVersion("2.3")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
