"""
ACP server — port 8903
Hosts three agents: planner, researcher, critic.
Each agent connects to SearchMCP (port 8901) for tools.
"""
# Compatibility patch: uvicorn >=0.30 renamed LoopSetupType → LoopFactoryType
import uvicorn.config as _uvc
if not hasattr(_uvc, "LoopSetupType"):
    _uvc.LoopSetupType = getattr(_uvc, "LoopFactoryType", str)

import json
import re
import asyncio

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from fastmcp import Client as MCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import settings, PLANNER_PROMPT, RESEARCHER_PROMPT, CRITIC_PROMPT
from schemas import ResearchPlan, CritiqueResult
from mcp_utils import mcp_tools_to_langchain

SEARCH_MCP_URL = f"http://localhost:{settings.search_mcp_port}/mcp"

server = Server()


async def _get_search_tools():
    """Connect to SearchMCP and return LangChain tools."""
    client = MCPClient(SEARCH_MCP_URL)
    async with client:
        mcp_tools = await client.list_tools()
    return mcp_tools_to_langchain(mcp_tools, MCPClient(SEARCH_MCP_URL))


def _build_llm():
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key.get_secret_value(),
    )


@server.agent()
async def planner(input: list[Message], context: Context) -> RunYield:
    """Planner agent: decomposes a research request into a structured ResearchPlan."""
    tools = await _get_search_tools()
    llm = _build_llm().with_structured_output(ResearchPlan)
    agent = create_react_agent(model=_build_llm(), tools=tools, prompt=PLANNER_PROMPT())

    user_text = input[-1].parts[0].content if input else ""
    result = agent.invoke({"messages": [{"role": "user", "content": user_text}]})
    text = result["messages"][-1].content

    # Try to parse structured output
    plan_obj = None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            plan_obj = ResearchPlan(**json.loads(m.group(1)))
        except Exception:
            pass
    if plan_obj is None:
        try:
            plan_obj = ResearchPlan(**json.loads(text))
        except Exception:
            pass
    if plan_obj is None:
        plan_obj = ResearchPlan(
            goal=user_text,
            search_queries=[user_text],
            sources_to_check=["knowledge_base", "web"],
            output_format="structured markdown report",
        )

    output = (
        f"Goal: {plan_obj.goal}\n"
        f"Search queries: {json.dumps(plan_obj.search_queries)}\n"
        f"Sources to check: {json.dumps(plan_obj.sources_to_check)}\n"
        f"Output format: {plan_obj.output_format}"
    )
    yield Message(role="agent", parts=[MessagePart(content=output)])


@server.agent()
async def researcher(input: list[Message], context: Context) -> RunYield:
    """Researcher agent: executes a research plan and returns detailed findings."""
    tools = await _get_search_tools()
    agent = create_react_agent(model=_build_llm(), tools=tools, prompt=RESEARCHER_PROMPT())

    user_text = input[-1].parts[0].content if input else ""
    result = agent.invoke({"messages": [{"role": "user", "content": user_text}]})
    text = result["messages"][-1].content

    yield Message(role="agent", parts=[MessagePart(content=text)])


@server.agent()
async def critic(input: list[Message], context: Context) -> RunYield:
    """Critic agent: evaluates research findings and returns a CritiqueResult."""
    tools = await _get_search_tools()
    agent = create_react_agent(model=_build_llm(), tools=tools, prompt=CRITIC_PROMPT())

    user_text = input[-1].parts[0].content if input else ""
    result = agent.invoke({"messages": [{"role": "user", "content": user_text}]})
    text = result["messages"][-1].content

    # Parse structured CritiqueResult
    critique_obj = None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            critique_obj = CritiqueResult(**json.loads(m.group(1)))
        except Exception:
            pass
    if critique_obj is None:
        try:
            critique_obj = CritiqueResult(**json.loads(text))
        except Exception:
            pass
    if critique_obj is None:
        verdict = "APPROVE" if "APPROVE" in text.upper() else "REVISE"
        critique_obj = CritiqueResult(
            verdict=verdict,
            is_fresh="outdated" not in text.lower(),
            is_complete="missing" not in text.lower(),
            is_well_structured=True,
            strengths=["Research completed"],
            gaps=[],
            revision_requests=[],
        )

    output = (
        f"Verdict: {critique_obj.verdict}\n"
        f"Is fresh: {critique_obj.is_fresh}\n"
        f"Is complete: {critique_obj.is_complete}\n"
        f"Is well structured: {critique_obj.is_well_structured}\n"
        f"Strengths: {json.dumps(critique_obj.strengths)}\n"
        f"Gaps: {json.dumps(critique_obj.gaps)}\n"
        f"Revision requests: {json.dumps(critique_obj.revision_requests)}"
    )
    yield Message(role="agent", parts=[MessagePart(content=output)])


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=settings.acp_port)
