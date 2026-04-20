# Multi-Agent Research System (homework-lesson-9)

Extends hw8 by migrating to a protocol-based architecture:
- **MCP** (FastMCP) exposes tools as HTTP servers
- **ACP** (acp-sdk) exposes agents as HTTP servers
- The Supervisor remains a local LangGraph agent that delegates via protocols

## Architecture

```
User (REPL)
  │
  ▼
Supervisor Agent (local, LangGraph)
  │
  ├── plan(request)       ──► ACP ──► Planner Agent  ──► MCP ──► SearchMCP :8901
  │                                                              (web_search, knowledge_search)
  │
  ├── research(plan)      ──► ACP ──► Research Agent ──► MCP ──► SearchMCP :8901
  │                                                              (web_search, read_url, knowledge_search)
  │
  ├── critique(findings)  ──► ACP ──► Critic Agent   ──► MCP ──► SearchMCP :8901
  │       ├── APPROVE → save_report
  │       └── REVISE  → back to research (max 2 rounds)
  │
  └── save_report(...)    ──► MCP ──► ReportMCP :8902  (HITL gated)
```

## Project Structure

```
homework-lesson-9/
├── main.py              # REPL with HITL interrupt/resume loop
├── supervisor.py        # Supervisor + ACP/MCP delegation tools
├── acp_server.py        # ACP server with 3 agents (planner, researcher, critic)
├── mcp_servers/
│   ├── search_mcp.py    # SearchMCP :8901 — web_search, read_url, knowledge_search
│   └── report_mcp.py    # ReportMCP :8902 — save_report
├── agents/
│   ├── planner.py       # Planner prompt + ResearchPlan schema (reused by acp_server)
│   ├── research.py      # Researcher prompt (reused by acp_server)
│   └── critic.py        # Critic prompt + CritiqueResult schema (reused by acp_server)
├── mcp_utils.py         # mcp_tools_to_langchain helper
├── schemas.py           # Pydantic models: ResearchPlan, CritiqueResult
├── config.py            # Settings + prompts + ports (8901, 8902, 8903)
├── retriever.py         # Hybrid FAISS + BM25 + reranker (from hw5)
├── ingest.py            # PDF ingestion pipeline (from hw5)
├── requirements.txt
├── data/                # PDF documents for RAG
└── .env                 # API keys (do not commit)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your-openai-api-key
MODEL_NAME=gpt-4o-mini
```

### 3. Ingest documents

```bash
python ingest.py
```

Builds the FAISS vector index in `vector_db/`. Run once (or when adding new PDFs).

## Startup Order

Each component runs as a separate process. Open four terminals:

```bash
# Terminal 1 — SearchMCP (tools: web_search, read_url, knowledge_search)
python mcp_servers/search_mcp.py

# Terminal 2 — ReportMCP (tool: save_report)
python mcp_servers/report_mcp.py

# Terminal 3 — ACP server (agents: planner, researcher, critic)
python acp_server.py

# Terminal 4 — Supervisor REPL
python main.py
```

## Usage

```
You: Compare RAG approaches: naive, sentence-window, and parent-child
```

The system will:
1. `[Supervisor → ACP → Planner]` — build a ResearchPlan
2. `[Supervisor → ACP → Researcher]` — execute research via SearchMCP tools
3. `[Supervisor → ACP → Critic]` — verify findings via SearchMCP tools
4. Revise if needed (max 2 rounds)
5. `[Supervisor → MCP → save_report]` — ask for HITL approval

### HITL Approval

```
⏸️  ACTION REQUIRES APPROVAL
  Tool:  save_report
  File:  rag_comparison.md

--- Report Preview ---
...
👉 approve / edit / reject:
```

- `approve` — saves to `output/`
- `edit` — enter feedback; Supervisor revises and asks again
- `reject` — cancels

## Configuration

| Setting | Default | Description |
|---|---|---|
| `MODEL_NAME` | `gpt-4o-mini` | LLM for all agents |
| `SEARCH_MCP_PORT` | `8901` | SearchMCP HTTP port |
| `REPORT_MCP_PORT` | `8902` | ReportMCP HTTP port |
| `ACP_PORT` | `8903` | ACP server HTTP port |
| `MAX_SEARCH_RESULTS` | `5` | Web search results per query |
| `TOP_K_RETRIEVAL` | `10` | RAG candidates before reranking |
| `TOP_K_RERANK` | `3` | Final RAG results after reranking |
| `OUTPUT_DIR` | `output` | Where reports are saved |

## What changed from hw8

| hw8 | hw9 |
|---|---|
| Tools as Python functions in one process | Tools exposed via FastMCP HTTP servers |
| Sub-agents as `@tool` wrappers | Sub-agents accessible via ACP server |
| Everything in one process | Each server is a separate HTTP endpoint |
| Direct function calls | Discovery → Delegate → Collect via protocols |
