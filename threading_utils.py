# threading_utils.py
import queue # For WorkerThread
import time  # For WorkerThread and SummarizationTask (potentially)
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, QThread

# Moved from novel_analyzer.py (LLMProcessor also moved, ensure it's imported if needed by tasks, though tasks receive api_config)
# from llm_processor import LLMProcessor # LLMProcessor is instantiated within the task's run method using api_config

class SummarizationSignals(QObject):
    update_signal = pyqtSignal(object, str) # identifier, summary_text
    progress_signal = pyqtSignal(int, int, int) # in_tokens, out_tokens, count
    error_signal = pyqtSignal(object, str)   # identifier, error_message
    finished_signal = pyqtSignal(object) # identifier

class SummarizationTask(QRunnable):
    def __init__(self, chapter_item_identifier, chapter_content, chapter_context, api_config, custom_prompt_text, main_window_ref):
        super().__init__()
        self.identifier = chapter_item_identifier
        self.content = chapter_content
        self.context = chapter_context
        self.api_config = api_config
        self.custom_prompt_for_processor = custom_prompt_text
        self.signals = SummarizationSignals()
        self.main_window = main_window_ref # Store reference to MainWindow

        # This import needs to be here if llm_processor.py is a separate file
        from llm_processor import LLMProcessor


    def run(self):
        # Check stop flag before doing significant work
        if self.main_window.stop_batch_requested:
            self.signals.error_signal.emit(self.identifier, "处理被用户中止")
            self.signals.finished_signal.emit(self.identifier) # Still signal finished
            return

        # LLMProcessor is instantiated here, specific to this task
        processor = LLMProcessor(self.api_config, self.custom_prompt_for_processor)
        summary_text = None # Ensure it's defined for the finally block
        try:
            # Another check before the actual API call
            if self.main_window.stop_batch_requested:
                self.signals.error_signal.emit(self.identifier, "处理被用户中止")
                self.signals.finished_signal.emit(self.identifier)
                return

            summary_text, in_tokens, out_tokens = processor.summarize(self.content, self.context)

            if self.main_window.stop_batch_requested: # Check immediately after potentially long call
                self.signals.error_signal.emit(self.identifier, "处理完成但已被用户中止")
                self.signals.finished_signal.emit(self.identifier)
                return

            if summary_text is not None:
                 self.signals.update_signal.emit(self.identifier, summary_text)
            # progress_signal is for overall batch, tokens are summed up in main window
            # However, individual task token usage can be emitted if needed,
            # for now, it's handled by handle_task_finished and update_batch_progress
            # Let's assume update_batch_progress is the one that receives token data.
            self.signals.progress_signal.emit(in_tokens, out_tokens, 1) # count = 1 chapter

        except Exception as e:
            self.signals.error_signal.emit(self.identifier, str(e))
        finally:
            self.signals.finished_signal.emit(self.identifier)


class WorkerThread(QThread): # Old worker, still used by summarize_selected
    update_signal = pyqtSignal(str, object)
    progress_signal = pyqtSignal(int, int, int) # This is the old progress signal
    error_signal = pyqtSignal(str)

    def __init__(self, work_queue, llm_processor_instance): # Modified to take an instance
        super().__init__()
        self.work_queue = work_queue
        self.llm_processor = llm_processor_instance # Expecting an already configured LLMProcessor
        self.running = True

    def run(self):
        while self.running and not self.work_queue.empty():
            try:
                task_data = self.work_queue.get_nowait() # Expects (item, context)
                item, context = task_data

                # LLMProcessor instance is now passed in __init__
                summary, in_tokens, out_tokens = self.llm_processor.summarize(
                    item.content, context
                )

                self.update_signal.emit("summary", (item, summary)) # Old signal format
                self.progress_signal.emit(in_tokens, out_tokens, 1) # Old signal format

                time.sleep(0.5)

            except queue.Empty:
                break
            except Exception as e:
                # Need to ensure item identifier is available for error reporting if we want to use new error_signal
                # For now, stick to old error_signal for WorkerThread
                self.error_signal.emit(f"处理错误: {str(e)}")
                break # Stop on first error for old worker thread

    def stop(self):
        self.running = False
