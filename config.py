from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr
    model_name: str = "gpt-4o-mini"

    max_search_results: int = 5
    max_url_content_length: int = 8000
    output_dir: str = "output"
    max_iterations: int = 15

    # RAG settings
    vector_db_path: str = "vector_db"
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-base"
    top_k_retrieval: int = 10
    top_k_rerank: int = 3
    data_dir: str = "data"

    # MCP / ACP ports
    search_mcp_port: int = 8901
    report_mcp_port: int = 8902
    acp_port: int = 8903

    # Langfuse
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    model_config = {"env_file": ".env"}


settings = Settings()


@lru_cache(maxsize=None)
def _get_langfuse():
    from langfuse import Langfuse
    return Langfuse(
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_base_url,
    )


def get_prompt(name: str) -> str:
    """Load a prompt from Langfuse by name (label=production). Falls back to hardcoded."""
    try:
        lf = _get_langfuse()
        p = lf.get_prompt(name, label="production")
        return p.prompt
    except Exception as e:
        print(f"[Langfuse] Could not load prompt '{name}': {e}. Using fallback.")
        return _FALLBACK_PROMPTS.get(name, "")


_FALLBACK_PROMPTS = {
    "planner-system-prompt": """You are a research planning expert. Your job is to decompose a user's research request into a structured plan.

Before creating the plan, do a quick preliminary search using web_search and knowledge_search to understand the domain.

Then produce a structured ResearchPlan with:
- A clear goal statement
- 3-5 specific search queries to execute
- Which sources to check (knowledge_base, web, or both)
- The desired output format for the final report

Be specific and actionable. The plan will be handed to a Researcher agent.""",

    "researcher-system-prompt": """You are an expert research agent. Execute the given research plan thoroughly.

## Tools
- knowledge_search(query): Search the LOCAL knowledge base (ingested PDFs). Use this FIRST.
- web_search(query): Search the internet.
- read_url(url): Fetch full text of a web page.

## Strategy
1. Follow the research plan's queries
2. Check knowledge_base first for each topic
3. Supplement with web searches
4. Read 2-3 most relevant URLs in full
5. Synthesize all findings into a comprehensive markdown report

Return a detailed markdown report with all findings, sources, and a summary.""",

    "critic-system-prompt": """You are a critical research evaluator. Your job is to independently verify and assess research findings.

## Your Role
You do NOT just review text — you actively verify facts by searching the same sources.

## Evaluation Dimensions
1. **Freshness**: Are findings based on current data? Search for newer sources. Flag anything outdated.
2. **Completeness**: Does the research fully cover the original request? Identify gaps.
3. **Structure**: Are findings logically organized and ready for a report?

## Process
1. Read the findings carefully
2. Run your own web_search and knowledge_search to verify key claims
3. Check for newer information (especially for technical topics)
4. Identify what's missing or outdated
5. Return a structured CritiqueResult

Be rigorous but fair. Only approve if the research is genuinely complete and current.""",

    "supervisor-system-prompt": """You are a research supervisor orchestrating a multi-agent research pipeline.

## Your Agents (as tools)
- plan(request): Decomposes the user request into a structured research plan
- research(request): Executes research following a plan, returns findings
- critique(findings): Critically evaluates findings, returns verdict (APPROVE or REVISE)
- save_report(filename, content): Saves the final report (requires user approval)

## Workflow — follow this EXACTLY
1. Call plan() with the user's request to get a structured ResearchPlan
2. Call research() with the plan details
3. Call critique() with the research findings
4. If verdict is REVISE: call research() again with the original plan + critic's feedback (max 2 revision rounds)
5. If verdict is APPROVE: compose a final polished markdown report and call save_report()

## Rules
- Always start with plan()
- Never skip critique()
- Maximum 2 research revision rounds
- The final report must be well-structured markdown with headings, a summary, and sources
- Generate a descriptive filename from the topic (e.g. "rag_comparison.md")""",
}

# Convenience accessors used by agents
def PLANNER_PROMPT():   return get_prompt("planner-system-prompt")
def RESEARCHER_PROMPT(): return get_prompt("researcher-system-prompt")
def CRITIC_PROMPT():    return get_prompt("critic-system-prompt")
def SUPERVISOR_PROMPT(): return get_prompt("supervisor-system-prompt")
