# Homework: Langfuse Observability

Connect Langfuse to your multi-agent system from the previous homework, configure tracing and online evaluation via LLM-as-a-Judge.

---

### What changes compared to previous homework

| Before | After |
|---|---|
| No observability — the system works as a black box | Every run is traced in Langfuse with a full call tree |
| DeepEval tests run locally and manually (hw10) | Langfuse automatically evaluates new traces via LLM-as-a-Judge |
| Prompts are hardcoded in the code | All agent system prompts are moved to Langfuse Prompt Management |

---

### Tasks

#### Task 0 — Langfuse credentials configured ✅
Keys added to `.env`:
```
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

#### Task 1 — Tracing integrated ✅
- `supervisor.py`: `get_langfuse_handler()` creates a `CallbackHandler` per run
- `main.py`: handler is passed via `config["callbacks"]` on every `supervisor.stream()` call
- Each run creates a trace in Langfuse with `session_id`, `user_id`, and `tags=["multi-agent", "research"]`

#### Task 2 — Session & User tracking ✅
- `SESSION_ID` is a single UUID shared across all runs in one process (groups traces into one session)
- `USER_ID = "researcher-user"` is set on every trace
- Check: Langfuse UI → Sessions tab and Users tab

#### Task 3 — Prompt Management ✅
- `setup_langfuse_prompts.py` pushed all 4 prompts to Langfuse with label `production`
- `config.py`: `get_prompt(name)` loads from Langfuse at runtime, falls back to hardcoded if unavailable
- `PLANNER_PROMPT`, `RESEARCHER_PROMPT`, `CRITIC_PROMPT`, `SUPERVISOR_PROMPT` are now callables
- All agents (`planner.py`, `research.py`, `critic.py`, `acp_server.py`, `supervisor.py`) call them as functions

#### Task 4 — LLM-as-a-Judge evaluators (manual step)
Set up at least 2 evaluators in Langfuse UI:

1. Go to **LLM-as-a-Judge → Evaluators → + Set up evaluator**
2. Suggested evaluators:
   - `answer-relevance` (numeric 0–1): Does the output answer the user's research question?
     Prompt: `Given the input: {{input}}\nAnd the output: {{output}}\nRate how relevant the answer is to the question on a scale from 0 to 1.`
   - `research-completeness` (boolean): Does the output cover all key aspects of the topic?
     Prompt: `Given the input: {{input}}\nAnd the output: {{output}}\nDoes the research output comprehensively cover the topic? Answer true or false.`
3. After 3-5 runs, wait 1-2 minutes and check **Tracing → Traces → Scores tab**

---

### How to run

```bash
# 1. Push prompts to Langfuse (one-time)
python setup_langfuse_prompts.py

# 2. Start MCP servers and ACP server (in separate terminals)
python mcp_servers/search_mcp.py
python mcp_servers/report_mcp.py
python acp_server.py

# 3. Run the main system
python main.py
```

---

### What to submit
- A `screenshots/` folder with 4 screenshots from the Langfuse UI:
  1. Trace tree (Tracing → Traces → open one trace)
  2. Session view (Sessions tab)
  3. Evaluator scores (Traces → Scores tab)
  4. Prompt Management (Prompts tab showing all 4 prompts)
