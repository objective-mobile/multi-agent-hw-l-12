import sys, os, time, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import settings, CRITIC_PROMPT
from tools import web_search, read_url, knowledge_search
from schemas import CritiqueResult


def _build_agent():
    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key.get_secret_value())
    return create_react_agent(model=llm, tools=[web_search, read_url, knowledge_search], prompt=CRITIC_PROMPT())


def _invoke_with_retry(agent, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return agent.invoke({"messages": messages})
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 60
                m = re.search(r"retry after (\d+)", err, re.IGNORECASE)
                if m:
                    wait = int(m.group(1)) + 2
                print(f"  Rate limit, waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def run_critic(findings: str) -> CritiqueResult:
    result = _invoke_with_retry(_build_agent(), [{"role": "user", "content": findings}])
    text = result["messages"][-1].content

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return CritiqueResult(**json.loads(m.group(1)))
        except Exception:
            pass
    try:
        return CritiqueResult(**json.loads(text))
    except Exception:
        pass

    verdict = "APPROVE" if "APPROVE" in text.upper() else "REVISE"
    return CritiqueResult(
        verdict=verdict,
        is_fresh="outdated" not in text.lower(),
        is_complete="missing" not in text.lower(),
        is_well_structured=True,
        strengths=["Research completed"],
        gaps=[],
        revision_requests=[],
    )
