"""
Supervisor agent — local orchestrator with Langfuse tracing.
"""
import asyncio
import json
import uuid

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from acp_sdk.client import Client as ACPClient
from acp_sdk.models import Message, MessagePart
from fastmcp import Client as MCPClient

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from config import settings, SUPERVISOR_PROMPT

ACP_BASE_URL = f"http://localhost:{settings.acp_port}"
REPORT_MCP_URL = f"http://localhost:{settings.report_mcp_port}/mcp"

_langfuse = Langfuse(
    secret_key=settings.langfuse_secret_key.get_secret_value(),
    public_key=settings.langfuse_public_key,
    host=settings.langfuse_base_url,
)


def get_langfuse_handler(session_id: str, user_id: str = "default-user", trace_name: str = "mas-run"):
    return CallbackHandler(
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_base_url,
        session_id=session_id,
        user_id=user_id,
        trace_name=trace_name,
        tags=["multi-agent", "research"],
    )


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


async def _acp_call(agent_name: str, message: str) -> str:
    async with ACPClient(base_url=ACP_BASE_URL) as client:
        run = await client.run_sync(
            agent=agent_name,
            input=[Message(role="user", parts=[MessagePart(content=message)])],
        )
        parts = run.output[-1].parts if run.output else []
        return parts[0].content if parts else ""


@tool
def plan(request: str) -> str:
    """Decompose the user research request into a structured ResearchPlan via ACP → Planner."""
    print(f"\n[Supervisor → ACP → Planner]")
    result = _run_async(_acp_call("planner", request))
    print(f"  {result[:120]}")
    return result


@tool
def research(request: str) -> str:
    """Execute research following a plan via ACP → Researcher. Returns detailed findings."""
    print(f"\n[Supervisor → ACP → Researcher]")
    result = _run_async(_acp_call("researcher", request))
    print(f"  Research complete ({len(result)} chars)")
    return result


@tool
def critique(findings: str) -> str:
    """Critically evaluate research findings via ACP → Critic. Returns APPROVE or REVISE."""
    print(f"\n[Supervisor → ACP → Critic]")
    result = _run_async(_acp_call("critic", findings))
    print(f"  {result[:120]}")
    return result


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
