# threading_utils.py
import queue
import time
import threading
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, QThread

from llm_processor import LLMProcessor

class SummarizationSignals(QObject):
    update_signal = pyqtSignal(object, str)
    progress_signal = pyqtSignal(int, int, int)
    error_signal = pyqtSignal(object, str)
    finished_signal = pyqtSignal(object)

class SummarizationTask(QRunnable):
    # chapter_content removed from constructor
    def __init__(self, chapter_item_identifier, chapter_context, api_config, custom_prompt_text, main_window_ref, encoding_object):
        super().__init__()
        self.identifier = chapter_item_identifier
        # self.content = chapter_content # REMOVED
        self.context = chapter_context
        self.api_config = api_config
        self.custom_prompt_for_processor = custom_prompt_text
        self.signals = SummarizationSignals()
        self.main_window = main_window_ref
        self.encoding_object = encoding_object

    def run(self):
        thread_id = threading.get_ident()
        print(f"DEBUG: SummarizationTask.run START for identifier: {self.identifier} - Thread ID: {thread_id}")

        if self.main_window.stop_batch_requested:
            print(f"DEBUG: SummarizationTask.run - STOP REQUESTED (early) for: {self.identifier} - Thread ID: {thread_id}")
            self.signals.error_signal.emit(self.identifier, "处理被用户中止")
            self.signals.finished_signal.emit(self.identifier)
            return

        # Fetch content on demand
        print(f"DEBUG: SummarizationTask.run - Fetching content for: {self.identifier} - Thread ID: {thread_id}")
        current_chapter_content = self.main_window.get_content_for_task(self.identifier)

        if current_chapter_content is None:
            print(f"DEBUG: SummarizationTask.run - No content found for {self.identifier} (or already processed). Skipping. - Thread ID: {thread_id}")
            self.signals.error_signal.emit(self.identifier, "内容未找到或已被处理")
            self.signals.finished_signal.emit(self.identifier)
            return

        # Original content length (now fetched content length) for reference
        original_content_length = len(current_chapter_content)
        # The temporary test content logic has been removed as per the plan for this subtask.

        print(f"DEBUG: SummarizationTask.run - About to instantiate LLMProcessor for: {self.identifier} - Thread ID: {thread_id}")
        processor = None
        try:
            processor = LLMProcessor(self.api_config, self.custom_prompt_for_processor, self.encoding_object)
            print(f"DEBUG: SummarizationTask.run - LLMProcessor INSTANTIATED for: {self.identifier} - Thread ID: {thread_id}")
        except Exception as e_proc_init:
            print(f"DEBUG: SummarizationTask.run - EXCEPTION during LLMProcessor init for {self.identifier}: {str(e_proc_init)} - Thread ID: {thread_id}")
            self.signals.error_signal.emit(self.identifier, f"LLMProcessor init error: {str(e_proc_init)}")
            if self.main_window: # Attempt to clear content even on init error
                self.main_window.clear_content_for_task(self.identifier)
            self.signals.finished_signal.emit(self.identifier)
            return

        summary_text = None
        in_tokens, out_tokens = 0, 0
        try:
            if self.main_window.stop_batch_requested:
                print(f"DEBUG: SummarizationTask.run - STOP REQUESTED (before summarize) for: {self.identifier} - Thread ID: {thread_id}")
                self.signals.error_signal.emit(self.identifier, "处理被用户中止")
                self.signals.finished_signal.emit(self.identifier) # Content will be cleared in finally
                return

            print(f"DEBUG: SummarizationTask.run - Calling processor.summarize for: {self.identifier}. Content length: {len(current_chapter_content)}. Thread ID: {thread_id}")

            prompt_sample_text_for_log = current_chapter_content
            if self.custom_prompt_for_processor:
                prompt_to_log = f"{str(self.custom_prompt_for_processor)[:100]}...\n{prompt_sample_text_for_log[:200]}..."
            else:
                prompt_to_log = f"{prompt_sample_text_for_log[:300]}..."
            print(f"DEBUG: SummarizationTask.run - Prompt sample for {self.identifier}: {prompt_to_log} - Thread ID: {thread_id}")

            summary_text, in_tokens, out_tokens = processor.summarize(current_chapter_content, self.context)
            print(f"DEBUG: SummarizationTask.run - processor.summarize RETURNED for: {self.identifier}. Summary obtained: {summary_text is not None}. Thread ID: {thread_id}")

            if self.main_window.stop_batch_requested:
                print(f"DEBUG: SummarizationTask.run - STOP REQUESTED (after summarize) for: {self.identifier} - Thread ID: {thread_id}")
                self.signals.error_signal.emit(self.identifier, "处理完成但已被用户中止")
                if summary_text is not None:
                    self.signals.update_signal.emit(self.identifier, summary_text)
                self.signals.progress_signal.emit(in_tokens, out_tokens, 1)
                self.signals.finished_signal.emit(self.identifier) # Content will be cleared in finally
                return

            if summary_text is not None:
                self.signals.update_signal.emit(self.identifier, summary_text)
            self.signals.progress_signal.emit(in_tokens, out_tokens, 1)

        except Exception as e_summarize:
            print(f"DEBUG: SummarizationTask.run - EXCEPTION during processor.summarize for {self.identifier}: {str(e_summarize)} - Thread ID: {thread_id}")
            self.signals.error_signal.emit(self.identifier, str(e_summarize))
            self.signals.progress_signal.emit(in_tokens, out_tokens, 1)
        finally:
            print(f"DEBUG: SummarizationTask.run - FINALLY block for: {self.identifier} - Thread ID: {thread_id}")
            if self.main_window: # Clear content from store once task is done (success or fail)
                self.main_window.clear_content_for_task(self.identifier)
                print(f"DEBUG: SummarizationTask.run - Cleared content for {self.identifier} from store. - Thread ID: {thread_id}")
            self.signals.finished_signal.emit(self.identifier)
        print(f"DEBUG: SummarizationTask.run FINISHED for identifier: {self.identifier} - Thread ID: {thread_id}")


class WorkerThread(QThread):
    update_signal = pyqtSignal(str, object)
    progress_signal = pyqtSignal(int, int, int)
    error_signal = pyqtSignal(str)

    def __init__(self, work_queue, llm_processor_instance):
        super().__init__()
        self.work_queue = work_queue
        self.llm_processor = llm_processor_instance
        self.running = True

    def run(self):
        while self.running and not self.work_queue.empty():
            try:
                task_data = self.work_queue.get_nowait()
                item, context = task_data

                summary, in_tokens, out_tokens = self.llm_processor.summarize(
                    item.content, context
                )

                self.update_signal.emit("summary", (item, summary))
                self.progress_signal.emit(in_tokens, out_tokens, 1)

                time.sleep(0.5)

            except queue.Empty:
                break
            except Exception as e:
                self.error_signal.emit(f"处理错误: {str(e)}")
                break

    def stop(self):
        self.running = False
