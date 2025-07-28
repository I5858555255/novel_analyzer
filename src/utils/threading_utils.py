"""
Threading utilities for the Novel Analyzer application.

This module provides classes for managing background tasks, specifically
for chapter summarization, using Qt's QThread and QRunnable mechanisms.
It includes signal definitions for inter-thread communication.
"""
from PyQt5.QtCore import QObject, QRunnable, QThread, pyqtSignal
import queue  # For queue.Empty in WorkerThread
import time
import threading

from core.llm_processor import LLMProcessor
# MainWindow is not directly imported, but passed as main_window_ref to SummarizationTask.


class SummarizationSignals(QObject):
    """
    Defines signals used by summarization tasks to communicate with the main thread.

    Signals:
        update_signal (object, str): Emitted when a chapter's summary is ready.
                                     Passes chapter identifier and summary text.
        progress_signal (int, int, int): Emitted to update progress.
                                         Passes input tokens, output tokens, and chapters done.
        error_signal (object, str): Emitted when an error occurs during summarization.
                                    Passes chapter identifier and error message.
        finished_signal (object): Emitted when a summarization task (for one chapter) is finished.
                                  Passes chapter identifier.
    """
    update_signal = pyqtSignal(object, str)
    progress_signal = pyqtSignal(int, int, int)  # in_tokens, out_tokens, chapters_done
    error_signal = pyqtSignal(object, str)       # identifier, error_message
    finished_signal = pyqtSignal(object)         # identifier


class SummarizationTask(QRunnable):
    """
    A QRunnable task for summarizing a single chapter in a background thread pool.

    This task interacts with an LLMProcessor to perform the summarization and
    emits signals to update the UI or report errors/progress. It's designed
    to be managed by a QThreadPool.
    """

    def __init__(self, chapter_item_identifier, chapter_context, api_config,
                 custom_prompt_text, main_window_ref, encoding_object, max_tokens_override=None):
        """
        Initializes the SummarizationTask.

        Args:
            chapter_item_identifier (any): A unique identifier for the chapter item,
                                           typically a tuple (volume_title, chapter_title).
            chapter_context (str): Contextual information (e.g., summary of previous chapter).
            api_config (dict): Configuration for the LLM API.
            custom_prompt_text (str): The custom prompt to be used for summarization.
            main_window_ref (MainWindow): A reference to the main application window.
                                          Used to access shared state and methods (e.g., stop requests).
            encoding_object (tiktoken.Encoding): The Tiktoken encoding object for token counting.
            max_tokens_override (int, optional): User-defined maximum tokens for the LLM response.
        """
        super().__init__()
        self.identifier = chapter_item_identifier
        self.context = chapter_context
        self.api_config = api_config
        self.custom_prompt_for_processor = custom_prompt_text
        self.signals = SummarizationSignals()
        self.main_window = main_window_ref  # Reference to MainWindow instance
        self.encoding_object = encoding_object
        self.max_tokens_override = max_tokens_override

    def run(self):
        """
        The main execution logic for the summarization task.

        This method is called when the QThreadPool executes this runnable.
        It retrieves chapter content, initializes an LLMProcessor, performs
        summarization, and emits appropriate signals.
        """
        # thread_id = threading.get_ident() # Useful for debugging multi-threading issues
        if self.main_window.stop_batch_requested:
            self.signals.error_signal.emit(self.identifier, "处理被用户中止")
            self.signals.finished_signal.emit(self.identifier)
            return

        current_chapter_content = self.main_window.get_content_for_task(
            self.identifier)
        if current_chapter_content is None: # Content might have been cleared or processed by another thread already
            self.signals.error_signal.emit(self.identifier, "内容未找到或已被处理")
            self.signals.finished_signal.emit(self.identifier)
            return

        processor = None
        try:
            processor = LLMProcessor(
                self.api_config, self.custom_prompt_for_processor, self.encoding_object,
                max_tokens_override=self.max_tokens_override)
        except Exception as e_proc_init:
            self.signals.error_signal.emit(
                self.identifier, f"LLMProcessor init error: {str(e_proc_init)}")
            if self.main_window: # Ensure main_window ref is valid
                self.main_window.clear_content_for_task(self.identifier)
            self.signals.finished_signal.emit(self.identifier)
            return

        summary_text = None
        in_tokens, out_tokens = 0, 0
        try:
            if self.main_window.stop_batch_requested: # Check again before potentially long API call
                self.signals.error_signal.emit(self.identifier, "处理被用户中止")
                self.signals.finished_signal.emit(self.identifier)
                return

            summary_text, in_tokens, out_tokens = processor.summarize(
                current_chapter_content, self.context)

            if self.main_window.stop_batch_requested: # Check if stopped during the API call
                self.signals.error_signal.emit(self.identifier, "处理完成但已被用户中止")
                if summary_text is not None: # Still provide summary if available
                    self.signals.update_signal.emit(
                        self.identifier, summary_text)
                self.signals.progress_signal.emit(in_tokens, out_tokens, 1)
                self.signals.finished_signal.emit(self.identifier)
                return

            if summary_text is not None:
                self.signals.update_signal.emit(self.identifier, summary_text)
            # Progress signal even if summary is None, to account for token usage / attempt
            self.signals.progress_signal.emit(in_tokens, out_tokens, 1)
        except Exception as e_summarize:
            self.signals.error_signal.emit(self.identifier, str(e_summarize))
            # Still emit progress to account for any tokens used before error
            self.signals.progress_signal.emit(in_tokens, out_tokens, 1)
        finally:
            # Clean up content from shared store once processing (success or fail) is done for this task
            if self.main_window:
                self.main_window.clear_content_for_task(self.identifier)
            self.signals.finished_signal.emit(self.identifier)


class WorkerThread(QThread):
    """
    A QThread for handling a queue of summarization tasks one by one (legacy).

    This thread processes chapter summarization tasks from a work queue.
    It's an older approach compared to SummarizationTask with QThreadPool.
    It takes one LLMProcessor instance, implying all tasks processed by this
    thread instance will use the same LLM configuration.

    Signals:
        update_signal (str, object): Emitted when a summary is ready.
                                     Passes "summary" and a tuple (item, summary_text).
        progress_signal (int, int, int): Emitted for progress updates.
                                         Passes input tokens, output tokens, and count.
        error_signal (str): Emitted on processing error. Passes error message.
    """
    update_signal = pyqtSignal(str, object)
    progress_signal = pyqtSignal(int, int, int)
    error_signal = pyqtSignal(str)

    def __init__(self, work_queue, llm_processor_instance):
        """
        Initializes the WorkerThread.

        Args:
            work_queue (queue.Queue): The queue from which to fetch chapter data.
                                      Each item is expected to be a tuple (ChapterTreeItem, context_str).
            llm_processor_instance (LLMProcessor): An instance of LLMProcessor to use for summarization.
        """
        super().__init__()
        self.work_queue = work_queue
        self.llm_processor = llm_processor_instance
        self.running = True  # Flag to control the thread's execution loop

    def run(self):
        """
        The main execution loop for the worker thread.

        Continuously fetches tasks from the work_queue and processes them
        until the queue is empty or the thread is stopped.
        """
        while self.running and not self.work_queue.empty():
            try:
                task_data = self.work_queue.get_nowait()  # Non-blocking get
                item, context = task_data  # item is a ChapterTreeItem
                summary, in_tokens, out_tokens = self.llm_processor.summarize(
                    item.content, context)
                if self.running: # Check if still running before emitting
                    self.update_signal.emit("summary", (item, summary))
                    self.progress_signal.emit(in_tokens, out_tokens, 1)
                time.sleep(0.5)  # Small delay between tasks
            except queue.Empty: # Should not happen due to work_queue.empty() check, but good for safety
                break
            except Exception as e:
                if self.running: # Check if still running before emitting error
                    self.error_signal.emit(f"处理错误: {str(e)}")
                break # Stop processing on error

    def stop(self):
        """Stops the worker thread's execution loop."""
        self.running = False
