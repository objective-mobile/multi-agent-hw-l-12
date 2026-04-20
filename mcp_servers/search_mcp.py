"""
SearchMCP server — port 8901
Tools: web_search, read_url, knowledge_search
Resources: resource://knowledge-base-stats
"""
import os
import sys
import pickle
from datetime import datetime

import trafilatura
import httpx
from ddgs import DDGS
from fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings

mcp = FastMCP("SearchMCP")


@mcp.tool()
def web_search(query: str) -> str:
    """Search the internet for a given query. Returns titles, URLs, and snippets."""
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)
        if not results:
            return "No results found."
        formatted = [
            f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}"
            for r in results
        ]
        return f"Found {len(results)} results:\n\n" + "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"


@mcp.tool()
def read_url(url: str) -> str:
    """Fetch and return the full text content of a web page."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Could not fetch content from {url}"
        text = trafilatura.extract(downloaded)
        if not text:
            return f"Could not extract text from {url}"
        content = text[: settings.max_url_content_length]
        return f"[{len(content)} chars] {content}"
    except httpx.TimeoutException:
        return f"Timeout while fetching {url}"
    except Exception as e:
        return f"Error reading {url}: {str(e)}"


@mcp.tool()
def knowledge_search(query: str) -> str:
    """Search the LOCAL knowledge base (ingested PDFs). Use this FIRST before web_search."""
    try:
        from retriever import hybrid_search, is_index_ready

        if not is_index_ready():
            return "Knowledge base index not found. Run `python ingest.py` first."

        results = hybrid_search(query)
        if not results:
            return "No relevant documents found in the knowledge base."

        parts = [f"Found {len(results)} relevant passages:\n"]
        for i, r in enumerate(results, 1):
            fname = os.path.basename(r["source"])
            parts.append(
                f"[{i}] (score={r['score']}) [{fname} | Page {r['page']}]\n{r['content']}"
            )
        return "\n\n".join(parts)
    except Exception as e:
        return f"Knowledge search error: {str(e)}"


@mcp.resource("resource://knowledge-base-stats")
def knowledge_base_stats() -> str:
    """Returns number of documents and last update date of the knowledge base."""
    chunks_path = os.path.join(settings.vector_db_path, "chunks.pkl")
    if not os.path.exists(chunks_path):
        return "Knowledge base not built yet. Run `python ingest.py`."
    mtime = os.path.getmtime(chunks_path)
    last_updated = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    sources = {c.metadata.get("source", "unknown") for c in chunks}
    return (
        f"Documents: {len(sources)}\n"
        f"Total chunks: {len(chunks)}\n"
        f"Last updated: {last_updated}"
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8901)
