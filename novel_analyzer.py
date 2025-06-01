# novel_analyzer.py (Main Entry Point)
import sys
from PyQt5.QtWidgets import QApplication
# Removed other imports as they are now in main_window.py or other specific modules

# Import MainWindow from its new location
from main_window import MainWindow

if __name__ == "__main__":
    # import sys # sys is already imported at the top
    
    app = QApplication(sys.argv)
    app.setApplicationName("小说智能分析工具")
    app.setApplicationVersion("2.0")
    
    window = MainWindow() # MainWindow is now imported
    # window.load_config() # This is confirmed to be called within MainWindow.__init__
    window.show()
    
    sys.exit(app.exec_())