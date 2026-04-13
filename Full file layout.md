biovoice-agents/                        ← 项目根目录（git 仓库）
│
├── pyproject.toml                      ← 新的统一包配置（已提供）
├── .env.example                        ← 已提供
├── requirements.txt                    ← 已提供
├── README.md
│
│ ┌─────────────────────────────────────────────────────────────────┐
│ │  NEW — BioVoice-Agents 新增代码                                  │
│ └─────────────────────────────────────────────────────────────────┘
├── biovoice/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                     ← agents_base.py
│   │   ├── registry.py                 ← agents_registry.py
│   │   ├── pubmed_agent.py             ← agents_pubmed.py
│   │   ├── pdb_agent.py                ← 从 agents_science.py 拆出
│   │   ├── uniprot_agent.py            ← 从 agents_science.py 拆出
│   │   ├── clinicaltrials_agent.py     ← 从 agents_science.py 拆出
│   │   ├── chembl_agent.py             ← 从 agents_science.py 拆出
│   │   ├── local_data_agent.py         ← 从 agents_local_europmc.py 拆出
│   │   └── europe_pmc_agent.py         ← 从 agents_local_europmc.py 拆出
│   ├── models/
│   │   ├── __init__.py
│   │   └── base.py                     ← models.py（含 OpenAIClient + OllamaClient + factory）
│   ├── core/
│   │   ├── __init__.py
│   │   ├── orchestrator.py             ← core_orchestrator.py
│   │   └── task.py                     ← 从 core_task_settings.py 拆出（Task + TaskStatus）
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                 ← 从 core_task_settings.py 拆出（BioVoiceSettings）
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py                     ← 从 cli_and_bots.py 拆出（CLI 部分）
│   ├── bots/
│   │   ├── __init__.py
│   │   └── gateway.py                  ← 从 cli_and_bots.py 拆出（FastAPI 部分）
│   └── ui/
│       ├── __init__.py
│       └── app.py                      ← ui_app.py
│
│ ┌─────────────────────────────────────────────────────────────────┐
│ │  INHERITED — FluBroad-Voice 原有代码，原路径不变                  │
│ │  biovoice/core/orchestrator.py 中的 import 直接找到这些模块       │
│ └─────────────────────────────────────────────────────────────────┘
├── app/
│   ├── __init__.py
│   ├── orchestrator.py                 ← FluBroad-Voice 原编排器（保留备用）
│   └── gradio_ui.py                    ← FluBroad-Voice 原 UI（保留备用）
│
├── core/                               ← FluBroad-Voice 核心库（直接 import 路径）
│   ├── __init__.py
│   ├── narrative/
│   │   ├── __init__.py
│   │   └── generator.py                ← generator.py（已修复版）
│   ├── rag/
│   │   ├── __init__.py
│   │   └── vector_store.py             ← vector_store.py（已修复版）
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── pubmed.py                   ← pubmed.py（已修复版，含 DOI/PMCID 提取）
│   │   ├── pmc_fulltext.py             ← pmc_fulltext.py（已修复版）
│   │   ├── enhanced_fetcher.py         ← enhanced_fetcher.py（已修复版）
│   │   ├── europe_pmc.py               ← europe_pmc.py（已修复版）
│   │   ├── pdf_processor.py            ← pdf_processor.py（已修复版）
│   │   └── biorxiv.py
│   ├── presentation/
│   │   ├── __init__.py
│   │   ├── ppt_generator.py            ← ppt_generator.py
│   │   ├── speech_synthesizer.py       ← speech_synthesizer.py
│   │   └── visualizer.py
│   └── utils/
│       ├── __init__.py
│       └── query_builder.py
│
├── domain/
│   └── virology/
│       ├── prompts/
│       │   ├── __init__.py
│       │   └── pmrc_templates.py
│       └── schemas/
│           ├── __init__.py
│           └── antibody_schema.py
│
├── scripts/
│   ├── crawl.py                        ← crawl.py（已修复版）
│   ├── fetch_all_flu_bnabs.py
│   ├── build_vector_db.py
│   ├── generate_from_json.py
│   ├── rag_qa.py                       ← rag_qa.py（已修复版）
│   └── search_demo.py
│
└── data/                               ← .gitignore 掉
    ├── flu_bnabs_all_articles.json
    ├── vector_db/
    ├── cache/
    └── checkpoints/