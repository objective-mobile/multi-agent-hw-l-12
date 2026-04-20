"""
Shared tools: web_search, read_url, knowledge_search, save_report.
Reused from hw5 with save_report replacing write_report.
"""
import os
import sys

import trafilatura
import httpx
from ddgs import DDGS
from langchain_core.tools import tool

# Allow importing retriever from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from config import settings


@tool
def web_search(query: str) -> str:
    """Search the internet for a given query. Returns titles, URLs, and snippets."""
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(
                f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}"
            )
        return f"Found {len(results)} results:\n\n" + "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
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


@tool
def knowledge_search(query: str) -> str:
    """Search the LOCAL knowledge base (ingested PDFs). Use this FIRST before web_search."""
    try:
        from retriever import hybrid_search, is_index_ready

        if not is_index_ready():
            return (
                "Knowledge base index not found. "
                "Run `python ingest.py` first to build the index."
            )

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


@tool
def save_report(filename: str, content: str) -> str:
    """Save a Markdown research report to the output directory."""
    try:
        os.makedirs(settings.output_dir, exist_ok=True)
        if not filename.endswith(".md"):
            filename += ".md"
        path = os.path.join(settings.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Report saved to {path}"
    except Exception as e:
        return f"Error saving report: {str(e)}"
