import sys, os, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import settings, RESEARCHER_PROMPT
from tools import web_search, read_url, knowledge_search


def _build_agent():
    llm = ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key.get_secret_value())
    return create_react_agent(model=llm, tools=[web_search, read_url, knowledge_search], prompt=RESEARCHER_PROMPT())


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


def run_researcher(request: str) -> str:
    result = _invoke_with_retry(_build_agent(), [{"role": "user", "content": request}])
    last = result["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)
