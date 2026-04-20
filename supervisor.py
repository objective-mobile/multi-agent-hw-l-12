"""
Supervisor agent — local orchestrator with Langfuse tracing (langfuse v4).
"""
import json
import uuid

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
import asyncio

from fastmcp import Client as MCPClient

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from config import settings, SUPERVISOR_PROMPT

REPORT_MCP_URL = f"http://localhost:{settings.report_mcp_port}/mcp"

_langfuse = Langfuse(
    secret_key=settings.langfuse_secret_key.get_secret_value(),
    public_key=settings.langfuse_public_key,
    host=settings.langfuse_base_url,
)


def get_langfuse_handler(session_id: str, user_id: str = "default-user", trace_name: str = "mas-run"):
    """
    Langfuse v4: CallbackHandler only accepts public_key.
    Session/user metadata is attached by starting a parent observation first.
    """
    trace_id = str(uuid.uuid4())

    # Create a parent trace with session/user metadata
    _langfuse.create_event(
        name=trace_name,
        metadata={
            "session_id": session_id,
            "user_id": user_id,
            "tags": ["multi-agent", "research"],
        },
    )

    handler = CallbackHandler(
        public_key=settings.langfuse_public_key,
    )
    return handler


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@tool
def plan(request: str) -> str:
    """Decompose the user research request into a structured ResearchPlan via Planner agent."""
    print(f"\n[Supervisor → Planner]")
    from agents.planner import run_planner
    result = run_planner(request)
    output = (
        f"Goal: {result.goal}\n"
        f"Search queries: {result.search_queries}\n"
        f"Sources to check: {result.sources_to_check}\n"
        f"Output format: {result.output_format}"
    )
    print(f"  {output[:120]}")
    return output


@tool
def research(request: str) -> str:
    """Execute research following a plan via Researcher agent. Returns detailed findings."""
    print(f"\n[Supervisor → Researcher]")
    from agents.research import run_researcher
    result = run_researcher(request)
    print(f"  Research complete ({len(result)} chars)")
    return result


@tool
def critique(findings: str) -> str:
    """Critically evaluate research findings via Critic agent. Returns APPROVE or REVISE."""
    print(f"\n[Supervisor → Critic]")
    from agents.critic import run_critic
    result = run_critic(findings)
    output = (
        f"Verdict: {result.verdict}\n"
        f"Strengths: {result.strengths}\n"
        f"Gaps: {result.gaps}\n"
        f"Revision requests: {result.revision_requests}"
    )
    print(f"  {output[:120]}")
    return output


@tool
def save_report(filename: str, content: str) -> str:
    """Save the final research report via ReportMCP. Requires user approval (HITL)."""
    print(f"\n[Supervisor → MCP → save_report]")

    decision = interrupt({
        "tool": "save_report",
        "filename": filename,
        "content_preview": content[:500] + ("..." if len(content) > 500 else ""),
        "full_content": content,
    })

    action = decision.get("action", "reject")

    if action == "approve":
        async def _save():
            async with MCPClient(REPORT_MCP_URL) as client:
                result = await client.call_tool("save_report", {"filename": filename, "content": content})
                return result[0].text if result else "Saved."
        result = _run_async(_save())
        print(f"  Approved! {result}")
        return result

    elif action == "edit":
        feedback = decision.get("feedback", "")
        return f"EDIT_REQUESTED: {feedback}"

    else:
        reason = decision.get("reason", "User rejected")
        print(f"  Rejected: {reason}")
        return f"Report saving was rejected by user: {reason}"


def build_supervisor():
    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key.get_secret_value())
    checkpointer = MemorySaver()
    return create_react_agent(
        model=llm,
        tools=[plan, research, critique, save_report],
        prompt=SUPERVISOR_PROMPT(),
        checkpointer=checkpointer,
    )
