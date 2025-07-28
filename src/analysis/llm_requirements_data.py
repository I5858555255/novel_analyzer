# llm_requirements_data.py
#
# Old to New ID Mapping:
# {   '0_core_principles': 'AG_1',
#     '1_1_1_physical_magic_rules': 'AG_2_1_1',
#     '1_1_2_social_structure_rules': 'AG_2_1_2',
#     '1_1_3_power_system': 'AG_2_1_3',
#     '1_1_4_time_space_rules': 'AG_2_1_4',
#     '1_1_5_key_constants': 'AG_2_1_5',
#     '1_1_core_rule_setting': 'AG_2_1',
#     '1_2_1_chronicles': 'AG_2_2_1',
#     '1_2_2_myths_legends': 'AG_2_2_2',
#     '1_2_3_gazetteer': 'AG_2_2_3',
#     '1_2_history_background_setting': 'AG_2_2',
#     '1_3_1_core_character_files': 'AG_2_3_1',
#     '1_3_2_character_relationship_map': 'AG_2_3_2',
#     '1_3_3_faction_organization_files': 'AG_2_3_3',
#     '1_3_character_basic_setting': 'AG_2_3',
#     '1_worldview_core_setting': 'AG_2',
#     '2_1_ultimate_goal_core_conflict': 'AG_3_1',
#     '2_2_1_global_outline_level1': 'AG_3_2_1',
#     '2_2_2_volume_outline_level2': 'AG_3_2_2',
#     '2_2_3_chapter_directory_outline_level3': 'AG_3_2_3',
#     '2_2_hierarchical_outline': 'AG_3_2',
#     '2_3_foreshadowing_management_system': 'AG_3_3',
#     '2_plot_structure_outline': 'AG_3',
#     '3_1_strict_adherence_to_outline': 'PO_1_1',
#     '3_3_writing_assistance_consistency_check': 'PO_1_2',
#     '3_4_dynamic_update_maintenance': 'PO_1_3',
#     '3_writing_execution_process_management': 'PO_1',
#     '4_1_modular_design': 'PO_2_1',
#     '4_2_character_lifecycle_management': 'PO_2_2',
#     '4_3_rhythm_redundancy_control': 'PO_2_3',
#     '4_special_requirements_for_ultra_long_high_chapter_count': 'PO_2'}
#
REQUIREMENTS_STRUCTURE = [   {   'description': '系统性、结构化、文档化、动态维护',
        'id': 'AG_1',
        'processing_type': 'prompt_only',
        'title': '核心原则'},
    {   'description': '创作一部千万字级别、数千章节的网络小说并保持情节、因果、伏笔的高度一致性，是一项宏大的系统工程，需要极其严谨的规划和执行。这绝非仅靠“写作天赋”或“即兴发挥”能完成的任务。以下是从专业角度出发，为保证一致性所需的具体要求和建议：',
        'id': 'AG_2',
        'processing_type': 'aggregate',
        'sub_items': [   {   'description': '',
                             'id': 'AG_2_1',
                             'processing_type': 'aggregate',
                             'sub_items': [   {   'description': '能量来源、运行逻辑、限制（如代价、冷却、等级上限）、可突破的条件。必须清晰、量化（至少是相对量化）。避免“为情节需要”临时修改规则。',
                                                  'id': 'AG_2_1_1',
                                                  'processing_type': 'aggregate',
                                                  'title': '物理/魔法规则'},
                                              {   'description': '权力体系、经济体系（货币、资源）、法律、文化禁忌、主要种族/势力关系图谱（初始状态）。设定其运作逻辑和稳定性/变革点。',
                                                  'id': 'AG_2_1_2',
                                                  'processing_type': 'aggregate',
                                                  'title': '社会结构规则'},
                                              {   'description': '修炼等级、职业划分、技能获取/升级方式、瓶颈与突破条件、不同体系间的兼容/冲突关系。需有明确、可验证的成长路径。',
                                                  'id': 'AG_2_1_3',
                                                  'processing_type': 'aggregate',
                                                  'title': '力量体系'},
                                              {   'description': '纪年法、地图（世界/区域/城市）、距离与时间换算、特殊空间（秘境、传送）的规则与限制。',
                                                  'id': 'AG_2_1_4',
                                                  'processing_type': 'aggregate',
                                                  'title': '时间与空间规则'},
                                              {   'description': '如寿命极限、关键资源（如灵石、特殊金属）的稀有度与分布、某些不可更改的历史事件（背景板）。',
                                                  'id': 'AG_2_1_5',
                                                  'processing_type': 'aggregate',
                                                  'title': '关键“常量”'}],
                             'title': '核心规则设定（铁律）：'},
                         {   'description': '',
                             'id': 'AG_2_2',
                             'processing_type': 'aggregate',
                             'sub_items': [   {   'description': '撰写详细的世界历史年表，包括重大事件、战争、王朝更迭、关键人物生平、灾难、科技/魔法突破点等。时间线必须精确到年（甚至月/日）。',
                                                  'id': 'AG_2_2_1',
                                                  'processing_type': 'aggregate',
                                                  'title': '编年史'},
                                              {   'description': '设定世界起源、创世神话、主要宗教/信仰体系及其核心教义、流传的预言和禁忌。区分哪些是真相，哪些是误传。',
                                                  'id': 'AG_2_2_2',
                                                  'processing_type': 'aggregate',
                                                  'title': '神话与传说'},
                                              {   'description': '绘制详细且分层级的地图（世界、大陆、国家、区域、城市），标注地形、气候、资源分布、主要交通路线、势力范围、危险区域。记录重要地点（如宗门、遗迹、都城）的详细描述。',
                                                  'id': 'AG_2_2_3',
                                                  'processing_type': 'aggregate',
                                                  'title': '地理志'}],
                             'title': '历史与背景设定（土壤）：'},
                         {   'description': '',
                             'id': 'AG_2_3',
                             'processing_type': 'aggregate',
                             'sub_items': [   {   'description': '主角、重要配角、关键反派。档案包含：姓名、称号、外貌（可配图）、性格（核心特质、优缺点、动机、恐惧、欲望）、出身背景（详细）、核心能力/功法/装备（来源、限制）、人际关系网（初始）、人生目标/执念、口头禅/习惯性动作。',
                                                  'id': 'AG_2_3_1',
                                                  'processing_type': 'aggregate',
                                                  'title': '核心人物档案'},
                                              {   'description': '使用markdown格式, '
                                                                 '动态更新的关系图，清晰展示所有重要角色之间的亲缘、师徒、盟友、敌对、爱慕、恩仇等关系及其变化时间点。',
                                                  'id': 'AG_2_3_2',
                                                  'processing_type': 'aggregate',
                                                  'title': '人物关系图谱'},
                                              {   'description': '记录所有重要宗门、家族、国家、佣兵团等的名称、标志、宗旨/教义、组织结构（领导人、核心成员）、势力范围、经济来源、内部派系、对外关系（盟友、敌人、中立）、核心功法/技术/资源。',
                                                  'id': 'AG_2_3_3',
                                                  'processing_type': 'aggregate',
                                                  'title': '势力组织档案'}],
                             'title': '人物基础设定（种子）：'}],
        'title': '一、 世界观与核心设定：打造坚不可摧的基石 (必须前置且详尽)'},
    {   'description': '',
        'id': 'AG_3',
        'processing_type': 'aggregate',
        'sub_items': [   {   'description': '明确小说的终极主题和主角（或群像）的最终目标。\n'
                                            '定义贯穿全书的核心冲突（如正邪之战、种族存亡、理念之争）及其本质。\n'
                                            '确定故事的大结局方向（不一定是细节，但要知道终点在哪）。',
                             'id': 'AG_3_1',
                             'processing_type': 'aggregate',
                             'title': '终极目标与核心冲突（灯塔）：'},
                         {   'description': '',
                             'id': 'AG_3_2',
                             'processing_type': 'aggregate',
                             'sub_items': [   {   'description': '划分几大卷（Part/Book），每卷对应一个核心冲突阶段或主角成长阶段（如：崛起、蛰伏、争霸、救世）。明确每卷的起止事件、核心目标、主要冲突方、关键转折点、卷末状态。',
                                                  'id': 'AG_3_2_1',
                                                  'processing_type': 'aggregate',
                                                  'title': '全局大纲 (Level 1)'},
                                              {   'description': '将每卷划分为若干大章（Arc），每个大章聚焦一个相对独立的核心事件或副本（如：宗门大比、秘境探险、国战、解决一个重大危机）。明确：\n'
                                                                 '大章目标。\n'
                                                                 '主要参与角色。\n'
                                                                 '核心矛盾与冲突点。\n'
                                                                 '起承转合的关键节点（开端、发展、高潮、转折、结局）。\n'
                                                                 '与本卷及全局目标的关联。\n'
                                                                 '必须回收的伏笔和需要埋下的新伏笔。',
                                                  'id': 'AG_3_2_2',
                                                  'processing_type': 'aggregate',
                                                  'title': '卷大纲 (Level 2)'},
                                              {   'description':
                                                                 '核心事件/POV（视点角色）。\n'
                                                                 '本章目标（角色要做什么/达成什么/揭示什么）。\n'
                                                                 '场景（地点、时间）。\n'
                                                                 '出场角色。\n'
                                                                 '情节推进点（具体发生了什么）。\n'
                                                                 '情感/氛围基调。\n'
                                                                 '回收的伏笔（精确到之前哪章埋下的）。\n'
                                                                 '新埋下的伏笔（内容、预计回收的大致位置/卷数）。\n'
                                                                 '对角色成长/关系/世界状态的影响。\n'
                                                                 '与前后章节的因果逻辑（为什么发生在此刻？导致什么后续？）。',
                                                  'id': 'AG_3_2_3',
                                                  'processing_type': 'chapter_specific',
                                                  'title': '章节大纲 (Level 3)'}],
                             'title': '分层级大纲（骨架与脉络）：'},
                         {   'description': '建立专属“伏笔档案”： 使用数据库（如Airtable, Notion）或电子表格。\n'
                                            '每条伏笔记录：\n'
                                            '伏笔ID（唯一编号）。\n'
                                            '埋下章节（精确编号）。\n'
                                            '内容描述（具体细节）。\n'
                                            '类型（人物、物品、事件、能力、世界观秘密等）。\n'
                                            '预计回收的大致范围（卷/大章）。\n'
                                            '实际回收章节（写作后填写）。\n'
                                            '状态（未回收/已回收/废弃）。\n'
                                            '关联伏笔（该伏笔可能指向或依赖的其他伏笔）。\n'
                                            '备注（灵感来源、可能的发展方向）。\n'
                                            '定期审查： '
                                            '写作过程中和每卷结束时，系统检查所有未回收伏笔，评估其合理性，调整回收计划或决定废弃（需有合理剧情解释）。',
                             'id': 'AG_3_3',
                             'processing_type': 'aggregate',
                             'title': '伏笔管理系统（暗线网络）：'}],
        'title': '二、 情节架构与大纲：构建清晰且可扩展的蓝图'},
    {   'description': '',
        'id': 'PO_1',
        'processing_type': 'aggregate',
        'sub_items': [   {   'description': '写作时，以章节目录大纲（Level '
                                            '3）为直接指导，确保每一章完成其预设目标（情节推进、伏笔回收/埋设、角色发展）。\n'
                                            '允许在大框架内进行局部微调（如对话细节、小场景优化），但涉及关键情节、设定、人物关系、核心伏笔的改动，必须 '
                                            '回溯修改大纲和相关设定文档，并评估对前后文一致性的影响。',
                             'id': 'PO_1_1',
                             'processing_type': 'aggregate',
                             'title': '严格遵守大纲（但不失灵活）：'},
                         {   'description': '人物/地点/物品追踪器： '
                                            '写作时，随时记录本章出现的人物（状态变化？）、地点（新描述？）、物品（获得/消耗/变化？）、能力（使用/升级？）。\n'
                                            '“前情提要”生成： '
                                            '对于关键转折或长时间断更后的章节，可考虑自动或手动生成简短的、精准的“前情提要”，只包含与本章直接相关的、必要的前置信息，避免信息冗余或误导。\n'
                                            '人工交叉检查：\n'
                                            '自查： '
                                            '每写完一定量（如10章或1个大章），停下来通读，重点检查情节逻辑、设定应用、人物言行一致性、伏笔处理。\n'
                                            '专业审读/编辑： '
                                            '聘请或培养熟悉你整个设定和大纲的专业编辑/审读者，定期进行深度审阅，专门挑刺找不一致和逻辑漏洞。\n'
                                            '读者反馈筛选： '
                                            '重视读者对明显漏洞（如时间线错误、人物失忆、能力矛盾）的反馈，但需结合自己的设定和大纲判断是否真为错误，或是读者理解偏差或伏笔。',
                             'id': 'PO_1_2',
                             'processing_type': 'prompt_only',
                             'title': '写作辅助与一致性检查（质检员）：'},
                         {   'description': '设定不是一成不变： 允许在写作过程中产生更优的设定灵感，但任何设定变更必须走流程：\n'
                                            '评估变更的必要性和优越性。\n'
                                            '评估变更对已发布内容、未写大纲、现有伏笔的全面影响。\n'
                                            '修改所有受影响的设定文档、大纲、伏笔档案。\n'
                                            '如果已发布内容受影响，考虑是否需要发布修订声明或在小说的后续内容中通过角色对话/发现进行“合理化”解释（需巧妙）。\n'
                                            '大纲的进化： '
                                            '随着写作深入，可能发现更好的情节走向。此时应回溯调整全局和卷级大纲，并同步更新所有后续的章节目录大纲和伏笔计划。避免“写到哪算哪”导致后期无法自圆其说。',
                             'id': 'PO_1_3',
                             'processing_type': 'prompt_only',
                             'title': '动态更新与维护（持续迭代）：'}],
        'title': '三、 写作执行与过程管理：确保精准落地的流程'},
    {   'description': '',
        'id': 'PO_2',
        'processing_type': 'prompt_only',
        'sub_items': [   {   'description': '将故事视为由相对独立但紧密关联的“模块”（大章/副本/事件）组成。每个模块有清晰的开端、发展、高潮、结局，并服务于更大的主线。\n'
                                            '方便管理、写作，也方便读者追读（避免数千章毫无喘息）。\n'
                                            '模块间通过关键线索人物、核心目标物品、重大悬疑事件、共同敌人/理念进行强关联。',
                             'id': 'PO_2_1',
                             'processing_type': 'prompt_only',
                             'title': '模块化设计'},
                         {   'description': '建立角色“登场-活跃-退场/死亡”记录。避免角色“神隐”或“诈尸”无交代。\n'
                                            '对于长期不活跃的重要配角，设定其“离线”期间合理的去向和活动。\n'
                                            '重要角色的退场（尤其是死亡）必须符合其性格、动机、故事逻辑，并在大纲中有计划。',
                             'id': 'PO_2_2',
                             'processing_type': 'aggregate',
                             'title': '角色生命周期管理'},
                         {   'description': '详略得当： '
                                            '千万字极易注水。严格区分核心情节推进、必要铺垫、氛围渲染与可有可无的冗余描写。每一章都应有其明确目的。\n'
                                            '定期“瘦身”审视： '
                                            '在卷末或特定节点，审视已写内容，删除或精简对主线、核心人物、关键伏笔无贡献的支线或冗余情节。\n'
                                            '避免“无限升级”陷阱： '
                                            '力量体系、地图、敌人强度需有规划，避免后期战力崩坏或地图无限扩张导致失控。设定明确的阶段性天花板和突破条件。',
                             'id': 'PO_2_3',
                             'processing_type': 'prompt_only',
                             'title': '节奏与冗余控制'}],
        'title': '四、 针对超长篇与高章节数的特殊要求'}]
