import time
from PyQt5.QtCore import QObject, QRunnable, pyqtSignal, QTimer
from llm_processor import LLMProcessor
from llm_requirements_data import REQUIREMENTS_STRUCTURE
from custom_widgets import ChapterTreeItem
from PyQt5.QtWidgets import QApplication


class BatchAnalysisProcessor(QObject):
    def __init__(self, main_window_ref):
        super().__init__()
        self.main_window = main_window_ref

        # Overall Analysis State
        self.total_chapters_to_process_overall = 0
        self.overall_chapters_fully_processed = 0 # Chapters that completed S1 (even if failed) + S2 (if S1 success)
        self.batch_start_time = 0

        # Volume Processing State
        self.volume_queue = [] # Queue of volume QTreeWidgetItems
        self.current_volume_item = None
        self.current_volume_title = None
        self.current_volume_total_chapters = 0
        self.current_volume_s1_tasks_completed = 0 # S1 tasks (success or fail) done for this volume
        self.current_volume_s2_sequences_completed = 0 # Chapters that completed their S2 sequence in this volume
        self.volume_chapter_outlines_map = {}  # {vol_title: [{'title': ch_title, 'outline': text, 'item': ChapterTreeItem}, ...]}
        self.completed_volume_outlines_list = [] # For Stage 4 input: [{'title': vol_title, 'outline': text}, ...]

        # Stage 1 (Chapter Outline) Batching State
        pool_max_threads = getattr(self.main_window.thread_pool, 'maxThreadCount', lambda: 4)()
        self.s1_task_concurrency_limit = max(1, pool_max_threads // 2 if pool_max_threads > 1 else 1)
        self.active_s1_tasks = 0
        self.s1_batch_chapter_items_pending = [] # Chapters from current volume waiting for S1 launch
        self.s1_current_batch_results = {} # {ch_title: {'outline_text':..., 'item':..., 'success':..., ...}}

        # Stage 2 (Aggregate Analysis) Sequential-per-Chapter State
        self.s1_chapters_currently_processing_s2 = [] # Queue of ChapterTreeItems from completed S1 batch, awaiting S2
        self.current_chapter_item_for_s2 = None
        self.current_s2_chapter_outline_text = None
        self.s2_aggregate_req_ids_queue = []
        self.s2_current_requirement_id = None
        self.s2_total_aggregate_reqs_for_chapter = 0
        self.s2_processed_aggregate_reqs_for_chapter = 0

        self.is_processing_s2_for_batch = False # True if an S2 sequence for a chapter is active

        # General LLM Task Tracking
        self.active_llm_tasks = 0 # Total active AutoFindTasks (S1, S2, S3, S4)

    def start_full_analysis(self):
        self.main_window.is_full_analysis_active = True
        self.main_window.stop_batch_requested = False
        self.main_window.logging_service.info("BatchAnalysisProcessor: Starting full novel analysis.")

        # Reset all relevant states
        self.total_chapters_to_process_overall = 0
        self.overall_chapters_fully_processed = 0
        self.volume_queue.clear()
        self.current_volume_item = None
        self.current_volume_title = None
        self.volume_chapter_outlines_map.clear()
        self.completed_volume_outlines_list.clear()

        self.s1_batch_chapter_items_pending.clear()
        self.s1_current_batch_results.clear()
        self.s1_chapters_currently_processing_s2.clear()
        self.active_s1_tasks = 0
        self.active_llm_tasks = 0
        self.is_processing_s2_for_batch = False

        book_item = self.main_window.chapter_tree.topLevelItem(0)
        if book_item:
            for i in range(book_item.childCount()):
                vol_item = book_item.child(i)
                self.volume_queue.append(vol_item)
                self.total_chapters_to_process_overall += vol_item.childCount()

        if not self.volume_queue:
            self.main_window.status_label.setText("没有卷可供分析。")
            self._finalize_full_analysis(stopped_by_user=False)
            return False

        self.main_window.progress_bar.setMaximum(self.total_chapters_to_process_overall)
        self.main_window.progress_bar.setValue(0)
        self.main_window.status_label.setText(f"开始全面分析... 共 {len(self.volume_queue)} 卷, {self.total_chapters_to_process_overall} 章节。")
        QApplication.processEvents()

        self.batch_start_time = time.time()
        self._process_next_volume()
        return True

    def _process_next_volume(self):
        if self.main_window.stop_batch_requested:
            if self.active_llm_tasks == 0 : self._finalize_full_analysis(stopped_by_user=True)
            return

        if not self.volume_queue:
            self.main_window.logging_service.info("All volumes' S1/S2 processing complete. Proceeding to Stage 4 (Global Outline).")
            self._start_stage4_global_outline_generation()
            return

        self.current_volume_item = self.volume_queue.pop(0)
        self.current_volume_title = self.current_volume_item.text(0)
        self.volume_chapter_outlines_map[self.current_volume_title] = []

        self.s1_batch_chapter_items_pending.clear()
        for j in range(self.current_volume_item.childCount()):
            chap_item = self.current_volume_item.child(j)
            if isinstance(chap_item, ChapterTreeItem):
                self.s1_batch_chapter_items_pending.append(chap_item)

        self.current_volume_total_chapters = len(self.s1_batch_chapter_items_pending)
        self.current_volume_s1_tasks_completed = 0
        self.current_volume_s2_sequences_completed = 0

        if not self.s1_batch_chapter_items_pending:
            self.main_window.logging_service.warning(f"Volume '{self.current_volume_title}' has no chapters. Skipping to next volume.")
            QTimer.singleShot(0, self._process_next_volume)
            return

        self.main_window.status_label.setText(f"处理卷: {self.current_volume_title} ({self.current_volume_total_chapters} 章节)")
        QApplication.processEvents()
        self._try_launch_next_s1_tasks()

    def _try_launch_next_s1_tasks(self):
        if self.main_window.stop_batch_requested :
            return
        
        if self.is_processing_s2_for_batch or self.active_s1_tasks > 0 :
            return

        if not self.s1_batch_chapter_items_pending:
            if self.current_volume_s1_tasks_completed == self.current_volume_total_chapters and \
               self.current_volume_s2_sequences_completed == self.current_volume_total_chapters and \
               self.active_llm_tasks == 0:
                self.main_window.logging_service.info(f"Volume '{self.current_volume_title}' S1/S2 done. Moving to Stage 3.")
                self._start_stage3_volume_outline_generation(self.current_volume_title)
            return

        self.s1_current_batch_results.clear()
        
        available_slots_in_pool = self.main_window.thread_pool.maxThreadCount() - self.active_llm_tasks
        num_to_launch_this_iteration = min(self.s1_task_concurrency_limit - self.active_s1_tasks,
                                           len(self.s1_batch_chapter_items_pending),
                                           available_slots_in_pool)

        if num_to_launch_this_iteration <= 0 and self.s1_batch_chapter_items_pending:
             if self.active_llm_tasks == 0 and not self.is_processing_s2_for_batch:
                  self.main_window.logging_service.warning(f"S1 Launch: Stalled for '{self.current_volume_title}'. Chapters pending, no active tasks. Attempting S1 batch completion handler.")
                  self._handle_s1_batch_completion()
             return

        launched_count_this_call = 0
        for _ in range(num_to_launch_this_iteration):
            if not self.s1_batch_chapter_items_pending: break

            chapter_item = self.s1_batch_chapter_items_pending.pop(0)
            self.s1_current_batch_results[chapter_item.original_title] = {
                'outline_text': None, 'item': chapter_item,
                'input_tokens': 0, 'output_tokens': 0, 'success': False, 'error_msg': None
            }
            self._start_stage1_for_chapter(chapter_item)
            launched_count_this_call +=1

        if launched_count_this_call > 0:
            self.main_window.status_label.setText(
                f"卷 '{self.current_volume_title}': 启动 {launched_count_this_call} 个S1任务。"
                f"本卷待S1: {len(self.s1_batch_chapter_items_pending)}, S1运行中: {self.active_s1_tasks}"
            )
            QApplication.processEvents()

    def _start_stage1_for_chapter(self, chapter_item: ChapterTreeItem):
        req_id_outline = self.main_window.CHAPTER_OUTLINE_REQ_ID
        req_data_outline = self.main_window._find_any_req_data(req_id_outline)

        # Initializing result structure early for this chapter
        if chapter_item.original_title not in self.s1_current_batch_results:
             self.s1_current_batch_results[chapter_item.original_title] = {
                'outline_text': None, 'item': chapter_item, 'input_tokens':0, 'output_tokens':0,
                'success': False, 'error_msg': "Pre-launch initialization"
             }

        if not req_data_outline or req_data_outline.get('processing_type') == 'non_participating':
            error_msg_detail = "设定未找到" if not req_data_outline else "未勾选参与提取"
            log_message = f"章节大纲设定 (ID: {req_id_outline}) 未参与 ({error_msg_detail}). 跳过S1: '{chapter_item.original_title}'."
            self.main_window.logging_service.info(log_message)
            self.s1_current_batch_results[chapter_item.original_title].update({'success': False, 'error_msg': log_message})
            self.active_s1_tasks +=1; self.active_llm_tasks +=1
            self.handle_s1_task_finished(req_id_outline, chapter_item.original_title, 0, 0)
            return

        if req_data_outline.get('processing_type') != 'chapter_specific':
            err_msg = f"章节大纲ID '{req_id_outline}' 类型不正确 (is {req_data_outline.get('processing_type')})."
            self.main_window.logging_service.error(err_msg + f" 跳过S1: '{chapter_item.original_title}'.")
            self.s1_current_batch_results[chapter_item.original_title].update({'success': False, 'error_msg': err_msg})
            self.active_s1_tasks +=1; self.active_llm_tasks +=1
            self.handle_s1_task_finished(req_id_outline, chapter_item.original_title, 0, 0)
            return

        req_title = req_data_outline.get('title', "章节大纲")
        req_description = req_data_outline.get('description', '生成章节大纲')
        api_config = self.main_window._get_current_config_for_saving()
        encoding_object = self.main_window.get_tiktoken_encoding(api_config['model'])
        if encoding_object is None:
            error_msg = f"编码器初始化失败 model {api_config['model']} (S1 for {chapter_item.original_title})"
            self.main_window.logging_service.error(error_msg)
            self.s1_current_batch_results[chapter_item.original_title].update({'success': False, 'error_msg': error_msg})
            self.active_s1_tasks +=1; self.active_llm_tasks +=1
            self.handle_s1_task_finished(req_id_outline, chapter_item.original_title, 0, 0)
            return

        prompt_only_info = self._get_prompt_only_requirements_info()
        chapter_global_idx = self.main_window.get_chapter_global_index(chapter_item)

        task = AutoFindTask(
            chapter_global_idx, req_id_outline, chapter_item.original_title, chapter_item.content,
            req_title, req_description, api_config, encoding_object, self.main_window,
            prompt_only_info, processing_stage='outline_generation'
        )
        task.signals.found_snippet.connect(self.handle_s1_task_snippet)
        task.signals.error.connect(self.handle_s1_task_error)
        task.signals.finished.connect(self.handle_s1_task_finished)

        self.main_window.thread_pool.start(task)
        self.active_s1_tasks += 1
        self.active_llm_tasks += 1
        self.main_window.stop_btn.setEnabled(True)

    def handle_s1_task_snippet(self, req_id, chapter_order_index, chapter_title, snippet_text):
        if chapter_title in self.s1_current_batch_results:
            self.s1_current_batch_results[chapter_title]['outline_text'] = snippet_text
            self.s1_current_batch_results[chapter_title]['success'] = True

            current_chapter_item = self.s1_current_batch_results[chapter_title]['item']
            if self.current_volume_title not in self.volume_chapter_outlines_map:
                 self.volume_chapter_outlines_map[self.current_volume_title] = []

            existing_entry = next((e for e in self.volume_chapter_outlines_map[self.current_volume_title] if e['title'] == chapter_title), None)
            if not existing_entry:
                 self.volume_chapter_outlines_map[self.current_volume_title].append({
                    'title': chapter_title, 'outline': snippet_text, 'item': current_chapter_item
                 })
            else:
                 existing_entry['outline'] = snippet_text; existing_entry['item'] = current_chapter_item
            self.main_window.logging_service.debug(f"S1 Snippet for '{chapter_title}' stored.")
        else:
            self.main_window.logging_service.error(f"S1 Snippet for '{chapter_title}', but not in s1_current_batch_results.")

    def handle_s1_task_error(self, req_id, chapter_title, error_message):
        if chapter_title in self.s1_current_batch_results:
            self.s1_current_batch_results[chapter_title]['success'] = False
            self.s1_current_batch_results[chapter_title]['error_msg'] = error_message
        else:
            self.main_window.logging_service.error(f"S1 Error signal for '{chapter_title}', but not in s1_current_batch_results. Error: {error_message}")
        self.main_window.logging_service.warning(f"S1 Task Error for chapter '{chapter_title}', req '{req_id}'. Error: {error_message}")

    def handle_s1_task_finished(self, req_id, chapter_title, input_tokens, output_tokens):
        self.active_s1_tasks -= 1
        self.active_llm_tasks -= 1
        self.current_volume_s1_tasks_completed += 1

        self.main_window.total_tokens[0] += input_tokens
        self.main_window.total_tokens[1] += output_tokens

        if chapter_title in self.s1_current_batch_results:
            self.s1_current_batch_results[chapter_title]['input_tokens'] = input_tokens
            self.s1_current_batch_results[chapter_title]['output_tokens'] = output_tokens

        self.main_window.logging_service.info(
            f"S1 Task finished for '{chapter_title}'. Active S1: {self.active_s1_tasks}, Total LLM: {self.active_llm_tasks}, Vol S1 Done: {self.current_volume_s1_tasks_completed}/{self.current_volume_total_chapters}"
        )
        self._update_progress_and_status()

        if self.main_window.stop_batch_requested:
            if self.active_llm_tasks == 0: self._finalize_full_analysis(stopped_by_user=True)
            return

        if self.active_s1_tasks == 0 and not self.is_processing_s2_for_batch :
            self._handle_s1_batch_completion()
        
        if self.active_s1_tasks < self.s1_task_concurrency_limit and self.s1_batch_chapter_items_pending and not self.is_processing_s2_for_batch:
             self._try_launch_next_s1_tasks()

    def _handle_s1_batch_completion(self):
        self.main_window.logging_service.info(f"S1 Batch completed for volume '{self.current_volume_title}'. {len(self.s1_current_batch_results)} S1 tasks processed.")

        self.s1_chapters_currently_processing_s2 = []
        for title, result_data in self.s1_current_batch_results.items():
            if result_data.get('success') and result_data.get('item'):
                self.s1_chapters_currently_processing_s2.append(result_data['item'])
            else:
                self.current_volume_s2_sequences_completed += 1
                self.overall_chapters_fully_processed +=1
                self.main_window.logging_service.info(f"Chapter '{title}' S1 failed/no item, skipping S2. Overall processed: {self.overall_chapters_fully_processed}")
        self._update_progress_and_status()

        if not self.s1_chapters_currently_processing_s2:
            self.main_window.logging_service.info(f"No successful S1 tasks in last batch for vol '{self.current_volume_title}' for S2.")
            self.s1_current_batch_results.clear()

            if not self.s1_batch_chapter_items_pending and self.current_volume_s1_tasks_completed == self.current_volume_total_chapters:
                if self.current_volume_s2_sequences_completed == self.current_volume_total_chapters and self.active_llm_tasks == 0:
                     self._start_stage3_volume_outline_generation(self.current_volume_title)
            else:
                 self._try_launch_next_s1_tasks()
            return

        self.is_processing_s2_for_batch = True
        self._process_next_chapter_for_s2()

    def _process_next_chapter_for_s2(self):
        if self.main_window.stop_batch_requested:
            if self.active_llm_tasks == 0: self._finalize_full_analysis(stopped_by_user=True)
            return

        if not self.s1_chapters_currently_processing_s2:
            self.is_processing_s2_for_batch = False
            self.s1_current_batch_results.clear()
            self.main_window.logging_service.info(f"S2 processing for S1 batch in vol '{self.current_volume_title}' complete.")

            if not self.s1_batch_chapter_items_pending and \
               self.current_volume_s1_tasks_completed == self.current_volume_total_chapters and \
               self.current_volume_s2_sequences_completed == self.current_volume_total_chapters:
                if self.active_llm_tasks == 0:
                    self._start_stage3_volume_outline_generation(self.current_volume_title)
            else:
                self._try_launch_next_s1_tasks()
            return

        self.current_chapter_item_for_s2 = self.s1_chapters_currently_processing_s2.pop(0)
        chapter_outline_data = None
        if self.current_volume_title in self.volume_chapter_outlines_map:
            found_data = next((cod for cod in self.volume_chapter_outlines_map[self.current_volume_title] if cod['title'] == self.current_chapter_item_for_s2.original_title), None)
            if found_data and found_data.get('outline'):
                chapter_outline_data = found_data

        if not chapter_outline_data:
            self.main_window.logging_service.error(f"S2: No S1 outline for chapter '{self.current_chapter_item_for_s2.original_title}'. Skipping S2.")
            self.current_volume_s2_sequences_completed += 1
            self.overall_chapters_fully_processed +=1
            self._update_progress_and_status()
            QTimer.singleShot(0, self._process_next_chapter_for_s2)
            return

        self.current_s2_chapter_outline_text = chapter_outline_data['outline']
        self.main_window.status_label.setText(
            f"卷 '{self.current_volume_title}': 章节 '{self.current_chapter_item_for_s2.original_title}' - Stage 2: 分析聚合项..."
        )
        QApplication.processEvents()
        self._start_stage2_for_chapter_internal()

    def _start_stage2_for_chapter_internal(self):
        self.s2_aggregate_req_ids_queue = self._get_req_ids_by_type('aggregate')
        self.s2_total_aggregate_reqs_for_chapter = len(self.s2_aggregate_req_ids_queue)
        self.s2_processed_aggregate_reqs_for_chapter = 0

        if not self.s2_aggregate_req_ids_queue:
            self.main_window.logging_service.info(f"No aggregate items for S2 of chapter '{self.current_chapter_item_for_s2.original_title}'.")
            self.current_volume_s2_sequences_completed += 1
            self.overall_chapters_fully_processed +=1
            self._update_progress_and_status()
            QTimer.singleShot(0, self._process_next_chapter_for_s2)
            return
        self._process_next_aggregate_requirement_for_s2()

    def _process_next_aggregate_requirement_for_s2(self):
        if self.main_window.stop_batch_requested: return
        if not self.s2_aggregate_req_ids_queue: return

        self.s2_current_requirement_id = self.s2_aggregate_req_ids_queue.pop(0)
        req_data_s2 = self.main_window._find_any_req_data(self.s2_current_requirement_id)
        if not req_data_s2:
             self.main_window.logging_service.error(f"S2: Agg ID '{self.s2_current_requirement_id}' not found for chap '{self.current_chapter_item_for_s2.original_title}'.")
             self.handle_s2_task_finished(self.s2_current_requirement_id, self.current_chapter_item_for_s2.original_title, 0,0)
             return

        req_title = req_data_s2.get('title', "未知聚合项")
        req_description = req_data_s2.get('description', '')
        self.main_window.status_label.setText(
            f"卷 '{self.current_volume_title}': 章 '{self.current_chapter_item_for_s2.original_title}' S2 - '{req_title}' ({self.s2_processed_aggregate_reqs_for_chapter + 1}/{self.s2_total_aggregate_reqs_for_chapter})"
        )
        QApplication.processEvents()

        api_config = self.main_window._get_current_config_for_saving()
        encoding_object = self.main_window.get_tiktoken_encoding(api_config['model'])
        if encoding_object is None:
            error_msg = f"编码器失败 model {api_config['model']} (S2 for {self.current_chapter_item_for_s2.original_title}, req {req_title})"
            self.main_window.logging_service.error(error_msg)
            self.handle_s2_task_finished(self.s2_current_requirement_id, self.current_chapter_item_for_s2.original_title, 0,0)
            return
            
        prompt_only_info = self._get_prompt_only_requirements_info()
        chapter_global_idx = self.main_window.get_chapter_global_index(self.current_chapter_item_for_s2)

        task = AutoFindTask(
            chapter_global_idx, self.s2_current_requirement_id, self.current_chapter_item_for_s2.original_title,
            None, req_title, req_description, api_config, encoding_object, self.main_window,
            prompt_only_info, processing_stage='detail_extraction',
            chapter_outline_text_input=self.current_s2_chapter_outline_text
        )
        task.signals.found_snippet.connect(self.handle_s2_task_snippet)
        task.signals.error.connect(self.handle_s2_task_error)
        task.signals.finished.connect(self.handle_s2_task_finished)

        self.main_window.thread_pool.start(task)
        self.active_llm_tasks += 1
        self.main_window.stop_btn.setEnabled(True)

    def handle_s2_task_snippet(self, req_id, chapter_order_index, chapter_title, snippet_text):
        self.main_window.logging_service.debug(f"S2 Snippet for '{chapter_title}', req '{req_id}' received.")
    def handle_s2_task_error(self, req_id, chapter_title, error_message):
        self.main_window.logging_service.warning(f"S2 Task Error for '{chapter_title}', req '{req_id}'. Error: {error_message}")

    def handle_s2_task_finished(self, req_id, chapter_title_from_task, input_tokens, output_tokens):
        self.active_llm_tasks -= 1
        self.s2_processed_aggregate_reqs_for_chapter += 1
        self.main_window.total_tokens[0] += input_tokens
        self.main_window.total_tokens[1] += output_tokens
        self._update_progress_and_status()

        if self.main_window.stop_batch_requested:
            if self.active_llm_tasks == 0: self._finalize_full_analysis(stopped_by_user=True)
            return

        if self.s2_processed_aggregate_reqs_for_chapter == self.s2_total_aggregate_reqs_for_chapter:
            self.main_window.logging_service.info(f"S2 sequence for chapter '{chapter_title_from_task}' completed.")
            self.current_volume_s2_sequences_completed += 1
            self.overall_chapters_fully_processed += 1
            self._update_progress_and_status()
            QTimer.singleShot(0, self._process_next_chapter_for_s2)
        else:
            if self.s2_aggregate_req_ids_queue:
                 QTimer.singleShot(50, self._process_next_aggregate_requirement_for_s2)
            elif self.active_llm_tasks == 0 : # Should be caught by above, but as fallback
                 self.main_window.logging_service.warning(f"S2 for '{chapter_title_from_task}': agg queue empty, but processed != total. Forcing S2 completion.")
                 if self.s2_processed_aggregate_reqs_for_chapter >= self.s2_total_aggregate_reqs_for_chapter:
                       self.current_volume_s2_sequences_completed += 1
                       self.overall_chapters_fully_processed += 1
                       self._update_progress_and_status()
                       QTimer.singleShot(0, self._process_next_chapter_for_s2)

    def _update_progress_and_status(self):
        if self.total_chapters_to_process_overall > 0:
            self.main_window.progress_bar.setValue(self.overall_chapters_fully_processed)
        status_parts = []
        if self.current_volume_title:
            status_parts.append(f"卷 '{self.current_volume_title}'")
            if self.is_processing_s2_for_batch and self.current_chapter_item_for_s2:
                 status_parts.append(f"章 '{self.current_chapter_item_for_s2.original_title}' (S2: {self.s2_processed_aggregate_reqs_for_chapter}/{self.s2_total_aggregate_reqs_for_chapter})")
            elif self.active_s1_tasks > 0:
                 status_parts.append(f"S1运行中 ({self.active_s1_tasks}任务)")
            status_parts.append(f"[本卷S1完成: {self.current_volume_s1_tasks_completed}/{self.current_volume_total_chapters}, S2完成: {self.current_volume_s2_sequences_completed}/{self.current_volume_total_chapters}]")
        if not status_parts and self.main_window.is_full_analysis_active : status_parts.append("准备中...")
        final_status_text = " | ".join(status_parts)
        if final_status_text: self.main_window.status_label.setText(final_status_text)

        if self.active_llm_tasks > 0 and self.batch_start_time > 0 and self.overall_chapters_fully_processed > 0:
            elapsed_time_total = max(1e-6, time.time() - self.batch_start_time)
            avg_time_per_chapter = elapsed_time_total / self.overall_chapters_fully_processed
            remaining_chapters = self.total_chapters_to_process_overall - self.overall_chapters_fully_processed
            if remaining_chapters > 0:
                eta_seconds = remaining_chapters * avg_time_per_chapter
                self.main_window.eta_label.setText(f"ETA (总): {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}")
            else: self.main_window.eta_label.setText("ETA (总): 完成")
            chapters_per_minute = self.overall_chapters_fully_processed / (elapsed_time_total / 60) if elapsed_time_total > 60 else 0 # Show only if some time passed
            self.main_window.metrics_label.setText(f"{chapters_per_minute:.2f} 章/分钟" if chapters_per_minute > 0 else "速率: 计算中...")
        elif self.active_llm_tasks == 0 and not self.main_window.is_full_analysis_active:
             self.main_window.eta_label.setText("ETA: N/A")
             self.main_window.metrics_label.setText("速率: N/A")
        self.main_window.token_label.setText(f"总消耗: 输入 {self.main_window.total_tokens[0]} | 输出 {self.main_window.total_tokens[1]}")
        QApplication.processEvents()

    def _start_stage3_volume_outline_generation(self, volume_title):
        self.main_window.status_label.setText(f"卷 '{volume_title}': Stage 3 - 生成卷大纲...")
        QApplication.processEvents()
        self.main_window.logging_service.info(f"Starting Stage 3: Volume Outline Generation for '{volume_title}'.")
        if self.main_window.stop_batch_requested:
            if self.active_llm_tasks == 0: self._finalize_full_analysis(stopped_by_user=True)
            return

        successful_chapter_outlines = []
        if volume_title in self.volume_chapter_outlines_map:
            for co_data in self.volume_chapter_outlines_map[volume_title]:
                 if co_data.get('outline'):
                      successful_chapter_outlines.append(f"章节：{co_data['title']}\n大纲：\n{co_data['outline']}")
        if not successful_chapter_outlines:
            self.main_window.logging_service.error(f"No successful S1 outlines for vol '{volume_title}'. Cannot make vol outline.")
            self._handle_stage3_finished('AG_3_2_2', volume_title, 0, 0)
            return

        all_chapter_outlines_text = "\n\n---\n\n".join(successful_chapter_outlines)
        requirement_id_vol_outline = 'AG_3_2_2'
        req_data_vol = self.main_window._find_any_req_data(requirement_id_vol_outline)
        if not req_data_vol or req_data_vol.get('processing_type') == 'non_participating':
            error_msg = f"卷大纲设定 '{requirement_id_vol_outline}' 未找到或未参与."
            self.main_window.logging_service.error(error_msg + f" (Volume: {volume_title})")
            self._handle_stage3_finished(requirement_id_vol_outline, volume_title, 0, 0)
            return

        req_title = req_data_vol.get('title', "卷大纲生成")
        req_description = req_data_vol.get('description', '请根据以下各章节大纲，为整个卷生成一个全面的大纲。')
        api_config = self.main_window._get_current_config_for_saving()
        encoding_object = self.main_window.get_tiktoken_encoding(api_config['model'])
        if encoding_object is None:
            error_msg = f"编码器初始化失败 for model {api_config['model']} (Stage 3 for {volume_title})"
            self.main_window.logging_service.error(error_msg)
            self._handle_stage3_finished(requirement_id_vol_outline, volume_title, 0, 0)
            return

        prompt_only_info = self._get_prompt_only_requirements_info()
        task = AutoFindTask(
            0, requirement_id_vol_outline, volume_title, all_chapter_outlines_text,
            req_title, req_description, api_config, encoding_object, self.main_window,
            prompt_only_info, processing_stage='volume_outline_generation'
        )
        task.signals.found_snippet.connect(self._handle_stage3_volume_outline_result)
        task.signals.finished.connect(self._handle_stage3_finished)
        self.main_window.thread_pool.start(task)
        self.active_llm_tasks += 1
        self.main_window.stop_btn.setEnabled(True)

    def _handle_stage3_volume_outline_result(self, req_id, volume_title_from_task, outline_text):
        self.main_window.logging_service.info(f"Volume outline snippet received for '{volume_title_from_task}'.")
        self.completed_volume_outlines_list.append({'title': volume_title_from_task, 'outline': outline_text})

    def _handle_stage3_finished(self, req_id, volume_title_from_task, input_tokens, output_tokens):
        self.active_llm_tasks -= 1
        self.main_window.total_tokens[0] += input_tokens
        self.main_window.total_tokens[1] += output_tokens
        self._update_progress_and_status()
        self.main_window.logging_service.info(f"Volume outline task finished for '{volume_title_from_task}'. Active LLM: {self.active_llm_tasks}")
        if self.main_window.stop_batch_requested:
            if self.active_llm_tasks == 0: self._finalize_full_analysis(stopped_by_user=True)
            return
        if self.active_llm_tasks == 0:
            QTimer.singleShot(0, self._process_next_volume)

    def _start_stage4_global_outline_generation(self):
        self.main_window.status_label.setText("Stage 4: 生成全局小说大纲...")
        QApplication.processEvents()
        self.main_window.logging_service.info("Starting Stage 4: Global Novel Outline Generation.")
        if self.main_window.stop_batch_requested:
            if self.active_llm_tasks == 0: self._finalize_full_analysis(stopped_by_user=True)
            return
        if not self.completed_volume_outlines_list:
            self.main_window.logging_service.error("No volume outlines available for Stage 4.")
            self._finalize_full_analysis(stopped_by_user=False)
            return

        all_volume_outlines_text = "\n\n---\n\n".join([f"卷：{vo['title']}\n卷大纲：\n{vo['outline']}" for vo in self.completed_volume_outlines_list])
        requirement_id_global_outline = 'AG_3_2_1'
        req_data_global = self.main_window._find_any_req_data(requirement_id_global_outline)
        if not req_data_global or req_data_global.get('processing_type') == 'non_participating':
            error_msg = f"全局大纲设定 '{requirement_id_global_outline}' 未找到或未参与."
            self.main_window.logging_service.error(error_msg)
            self._handle_stage4_finished(requirement_id_global_outline, "GlobalOutline",0,0)
            return

        req_title = req_data_global.get('title', "全局大纲生成")
        req_description = req_data_global.get('description', '生成全局大纲')
        api_config = self.main_window._get_current_config_for_saving()
        encoding_object = self.main_window.get_tiktoken_encoding(api_config['model'])
        if encoding_object is None:
            error_msg = f"编码器初始化失败 model {api_config['model']} (Stage 4 Global Outline)"
            self.main_window.logging_service.error(error_msg)
            self._handle_stage4_finished(requirement_id_global_outline, "GlobalOutline",0,0)
            return

        prompt_only_info = self._get_prompt_only_requirements_info()
        task = AutoFindTask(
            -1, requirement_id_global_outline, "Global Novel Outline", all_volume_outlines_text,
            req_title, req_description, api_config, encoding_object, self.main_window,
            prompt_only_info, processing_stage='global_outline_generation'
        )
        task.signals.found_snippet.connect(self._handle_stage4_global_outline_result)
        task.signals.finished.connect(self._handle_stage4_finished)
        self.main_window.thread_pool.start(task)
        self.active_llm_tasks += 1
        self.main_window.stop_btn.setEnabled(True)

    def _handle_stage4_global_outline_result(self, req_id, identifier, outline_text):
        self.main_window.logging_service.info(f"Global outline snippet received for '{identifier}'.")

    def _handle_stage4_finished(self, req_id, identifier, input_tokens, output_tokens):
        self.active_llm_tasks -= 1
        self.main_window.total_tokens[0] += input_tokens
        self.main_window.total_tokens[1] += output_tokens
        self._update_progress_and_status()
        self.main_window.logging_service.info(f"Global outline task finished ('{identifier}'). Active LLM: {self.active_llm_tasks}")
        if self.active_llm_tasks == 0:
             self._finalize_full_analysis(stopped_by_user=self.main_window.stop_batch_requested)

    def _finalize_full_analysis(self, stopped_by_user=False):
        self.main_window.is_full_analysis_active = False
        self.main_window.run_full_analysis_btn.setEnabled(True)
        self.main_window.stop_btn.setEnabled(False)
        final_status_message = ""
        current_chapter_title_display = "N/A"
        if stopped_by_user:
            if self.is_processing_s2_for_batch and self.current_chapter_item_for_s2:
                current_chapter_title_display = self.current_chapter_item_for_s2.original_title + " (S2)"
            elif self.active_s1_tasks > 0 :
                 if self.s1_current_batch_results:
                      current_chapter_title_display = next(iter(self.s1_current_batch_results.keys()), "Unknown S1 Chapter") + " (S1)"
                 elif self.s1_batch_chapter_items_pending :
                      current_chapter_title_display = self.s1_batch_chapter_items_pending[0].original_title + " (S1 pending)"
            final_status_message = (
                f"全面分析已由用户停止。已处理 {self.overall_chapters_fully_processed}/{self.total_chapters_to_process_overall} 章节。"
                f"大致停止于卷 '{self.current_volume_title}', 章节 '{current_chapter_title_display}'."
            )
            if hasattr(self.main_window, 'progress_bar'):
                self.main_window.progress_bar.setValue(self.overall_chapters_fully_processed)
        else:
            if hasattr(self.main_window, 'progress_bar'):
                if self.total_chapters_to_process_overall > 0 :
                     self.main_window.progress_bar.setValue(self.total_chapters_to_process_overall)
                else: self.main_window.progress_bar.setValue(0)
            final_status_message = f"全面分析完成。共处理 {self.overall_chapters_fully_processed} 章节。"
            if self.total_chapters_to_process_overall == 0 and self.overall_chapters_fully_processed == 0:
                 final_status_message = "全面分析完成。未找到可处理的章节。"
        self.main_window.status_label.setText(final_status_message)
        self._update_progress_and_status() # To clear ETA/Metrics

        self.s1_batch_chapter_items_pending.clear()
        self.s1_current_batch_results.clear()
        self.s1_chapters_currently_processing_s2.clear()
        self.active_s1_tasks = 0
        self.active_llm_tasks = 0
        self.is_processing_s2_for_batch = False
        self.current_chapter_item_for_s2 = None
        self.current_s2_chapter_outline_text = None
        self.main_window.stop_batch_requested = False

    def _get_req_ids_by_type(self, target_type):
        ids = []
        def find_ids_recursive(items_list, type_to_find, collector):
            for item_data in items_list:
                effective_req_data = self.main_window._find_any_req_data(item_data['id'])
                if effective_req_data and effective_req_data.get('processing_type') == type_to_find:
                    collector.append(item_data['id'])
                if 'sub_items' in item_data and item_data['sub_items']:
                    find_ids_recursive(item_data['sub_items'], type_to_find, collector)
        find_ids_recursive(REQUIREMENTS_STRUCTURE, target_type, ids)
        return ids

    def _get_prompt_only_requirements_info(self):
        prompt_only_info_list = []
        def collect_prompt_only_recursive(items_list, collector):
            for item_data in items_list:
                effective_req_data = self.main_window._find_any_req_data(item_data['id'])
                if effective_req_data and effective_req_data.get('processing_type') == 'prompt_only':
                    title = effective_req_data.get('title', item_data['id'])
                    description = effective_req_data.get('description', '')
                    collector.append({'title': title, 'description': description})
                if 'sub_items' in item_data and item_data['sub_items']:
                    collect_prompt_only_recursive(item_data['sub_items'], collector)
        collect_prompt_only_recursive(REQUIREMENTS_STRUCTURE, prompt_only_info_list)
        return prompt_only_info_list

    def request_stop(self):
        self.main_window.logging_service.info("BatchAnalysisProcessor: Stop request acknowledged by processor.")

class AutoFindTaskSignals(QObject):
    found_snippet = pyqtSignal(str, int, str, str)
    error = pyqtSignal(str, str, str)
    finished = pyqtSignal(str, str, int, int)

class AutoFindTask(QRunnable):
    def __init__(self, chapter_order_index, requirement_id, chapter_title, chapter_content, 
                 requirement_title, requirement_description, api_config_dict, encoding_object,
                 main_window_ref, prompt_only_requirements_info_list, 
                 processing_stage, chapter_outline_text_input=None):
        super().__init__()
        self.signals = AutoFindTaskSignals()
        self.chapter_order_index = chapter_order_index
        self.requirement_id = requirement_id
        self.chapter_title = chapter_title
        self.chapter_content = chapter_content
        self.requirement_title = requirement_title
        self.requirement_description = requirement_description
        self.api_config = api_config_dict
        self.encoding_object = encoding_object
        self.main_window_ref = main_window_ref
        self.prompt_only_requirements_info_list = prompt_only_requirements_info_list
        self.processing_stage = processing_stage
        self.chapter_outline_text_input = chapter_outline_text_input

    def run(self):
        input_tokens, output_tokens = 0, 0
        try:
            if self.main_window_ref.stop_batch_requested:
                self.signals.finished.emit(self.requirement_id, self.chapter_title, 0, 0)
                return

            prompt_parts = ["您是一位专业的文学分析助手。\n"]
            if self.prompt_only_requirements_info_list:
                prompt_parts.append("\n--- 以下是相关的背景设定和创作要求，请在分析时予以参考 ---\n")
                for po_info in self.prompt_only_requirements_info_list:
                    prompt_parts.append(f"--- 参考设定：{po_info['title']} ---\n")
                    prompt_parts.append(f"{po_info['description']}\n")
                    prompt_parts.append("--- 参考设定结束 ---\n\n")
                prompt_parts.append("--- 背景设定和创作要求结束 ---\n\n")

            if self.processing_stage == 'outline_generation':
                prompt_parts.append("请仔细阅读以下小说章节原始内容：\n")
                prompt_parts.append(f"--- 小说章节原始内容 ---\n{self.chapter_content}\n--- 小说章节原始内容结束 ---\n\n")
            elif self.processing_stage == 'detail_extraction':
                prompt_parts.append("请仔细阅读以下已经生成的章节大纲：\n")
                prompt_parts.append(f"--- 章节大纲内容 ---\n{self.chapter_outline_text_input}\n--- 章节大纲内容结束 ---\n\n")
            elif self.processing_stage == 'volume_outline_generation':
                prompt_parts.append("请仔细阅读以下各章节大纲的汇总内容：\n")
                prompt_parts.append(f"--- 章节大纲汇总内容 ---\n{self.chapter_content}\n--- 章节大纲汇总内容结束 ---\n\n")
            elif self.processing_stage == 'global_outline_generation':
                prompt_parts.append("请仔细阅读以下各卷大纲的汇总内容：\n")
                prompt_parts.append(f"--- 各卷大纲汇总内容 ---\n{self.chapter_content}\n--- 各卷大纲汇总内容结束 ---\n\n")

            prompt_parts.append(f"我正在分析与写作要求“{self.requirement_title}”（ID: {self.requirement_id}）相关的内容。\n")
            prompt_parts.append(f"要求描述：“{self.requirement_description}”\n\n")
            
            if self.processing_stage == 'outline_generation':
                prompt_parts.append("请基于上述小说章节原始内容，并综合考虑前面提供的所有背景设定和创作要求，针对以上“要求描述”，生成一份详细的章节大纲。\n")
            elif self.processing_stage == 'detail_extraction':
                prompt_parts.append("请基于上述章节大纲内容，并综合考虑前面提供的所有背景设定和创作要求，针对以上“要求描述”，从大纲中总结与“要求标题”相关的核心信息和要点。\n")
            elif self.processing_stage == 'volume_outline_generation':
                prompt_parts.append(f"请基于上述各章节大纲的汇总内容，并综合考虑前面提供的所有背景设定和创作要求，针对以上“要求描述”（即“{self.requirement_title}”），为当前卷（卷名为“{self.chapter_title}”）生成一份连贯的、结构化的卷大纲。\n")
                prompt_parts.append("卷大纲应综合体现各章节的核心情节和发展脉络，展示卷的整体故事结构。\n")
            elif self.processing_stage == 'global_outline_generation':
                prompt_parts.append(f"请基于上述各卷大纲的汇总内容，并综合考虑前面提供的所有背景设定和创作要求，针对以上“要求描述”（即“{self.requirement_title}”），生成一份连贯的、结构化的全局小说大纲。\n")
                prompt_parts.append("全局大纲应高度概括整个小说的核心故事线、主要转折点和最终结局，体现作品的整体构思。\n")

            prompt_parts.append("您的总结应简明扼要，准确反映所提供文本中与该要求相关的内容。请不要直接大段摘抄原文。\n")
            prompt_parts.append(f"如果所提供文本内容未提及与“{self.requirement_title}”直接或间接相关的信息，则不需在分析笔记中添加任何内容。\n")
            prompt_parts.append("请避免添加无关的评论或猜测，专注于对文本内信息的归纳总结。\n")
            prompt_parts.append("请直接输出核心分析结果，不要包含如“好的，这是您要求的分析：”或“总结如下：”等多余的对话性文字。\n")
            full_prompt = "".join(prompt_parts)

            processor = LLMProcessor(
                self.api_config,
                custom_prompt_text="",
                encoding_object=self.encoding_object
            )
            processor.custom_prompt_for_processor = full_prompt

            self.main_window_ref.logging_service.api_request(
                "LLM call from AutoFindTask.",
                details={'model': self.api_config.get('model'), 'prompt_length': len(full_prompt), 'chapter': self.chapter_title, 'requirement': self.requirement_title, 'stage': self.processing_stage}
            )

            snippet_text, input_tokens, output_tokens = processor.summarize(
                text="请根据以上提供的完整指令和文本内容进行分析。", context=""
            )

            snippet_text_lower = snippet_text.lower() if snippet_text else ""
            is_empty_or_whitespace = not snippet_text or snippet_text.isspace()
            negative_keywords = ["未在当前章节找到", "not found", "未找到相关内容", "没有直接相关的信息","没有间接相关的信息", "信息不足", "not directly related", "insufficient information", "无法找到相关信息", "no relevant information", "unable to find relevant information", "没有找到", "找不到", "未提及", "没有提及", "未发现", "没有发现"]
            contains_negative_keyword = any(keyword in snippet_text_lower for keyword in negative_keywords) if not is_empty_or_whitespace else False
            found_snippet_bool = not is_empty_or_whitespace and not contains_negative_keyword

            self.main_window_ref.logging_service.api_response(
                "LLM response for AutoFindTask.",
                details={ 'model': self.api_config.get('model'), 'input_tokens': input_tokens, 'output_tokens': output_tokens, 'found_snippet': found_snippet_bool, 'snippet_length': len(snippet_text) if snippet_text else 0, 'chapter': self.chapter_title, 'requirement': self.requirement_title, 'stage': self.processing_stage}
            )

            if self.main_window_ref.stop_batch_requested:
                self.signals.finished.emit(self.requirement_id, self.chapter_title, input_tokens, output_tokens)
                return

            if found_snippet_bool:
                self.signals.found_snippet.emit(self.requirement_id, self.chapter_order_index, self.chapter_title, snippet_text)
                self.main_window_ref.handle_auto_find_snippet(self.requirement_id, self.chapter_order_index, self.chapter_title, snippet_text)

            self.signals.finished.emit(self.requirement_id, self.chapter_title, input_tokens, output_tokens)

        except Exception as e:
            self.main_window_ref.logging_service.error(
                f"Error in AutoFindTask for req '{self.requirement_id}', chapter '{self.chapter_title}', stage '{self.processing_stage}'.",
                details={'error': str(e)}, exc_info=True
            )
            self.signals.error.emit(self.requirement_id, self.chapter_title, str(e))
            self.main_window_ref.handle_auto_find_error(self.requirement_id, self.chapter_title, str(e))
            self.signals.finished.emit(self.requirement_id, self.chapter_title, input_tokens, output_tokens)
