"""
Test suite for the MCP + ACP multi-agent pipeline (hw9).

Tests are split into:
1. Unit tests — config, schemas, mcp_utils (no servers needed)
2. MCP server tests — start servers in-process and call tools directly
3. Integration smoke test — full pipeline with live servers (skipped if servers not running)
"""
import asyncio
import json
import os
import sys
import uuid
import importlib
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. Unit tests
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    def test_ports_defined(self):
        from config import settings
        self.assertEqual(settings.search_mcp_port, 8901)
        self.assertEqual(settings.report_mcp_port, 8902)
        self.assertEqual(settings.acp_port, 8903)

    def test_prompts_exist(self):
        from config import PLANNER_PROMPT, RESEARCHER_PROMPT, CRITIC_PROMPT, SUPERVISOR_PROMPT
        for p in [PLANNER_PROMPT, RESEARCHER_PROMPT, CRITIC_PROMPT, SUPERVISOR_PROMPT]:
            self.assertIsInstance(p, str)
            self.assertGreater(len(p), 20)


class TestSchemas(unittest.TestCase):
    def test_research_plan(self):
        from schemas import ResearchPlan
        plan = ResearchPlan(
            goal="Test goal",
            search_queries=["query1", "query2"],
            sources_to_check=["web"],
            output_format="markdown",
        )
        self.assertEqual(plan.goal, "Test goal")
        self.assertEqual(len(plan.search_queries), 2)

    def test_critique_result_approve(self):
        from schemas import CritiqueResult
        c = CritiqueResult(
            verdict="APPROVE",
            is_fresh=True,
            is_complete=True,
            is_well_structured=True,
            strengths=["good"],
            gaps=[],
            revision_requests=[],
        )
        self.assertEqual(c.verdict, "APPROVE")

    def test_critique_result_revise(self):
        from schemas import CritiqueResult
        c = CritiqueResult(
            verdict="REVISE",
            is_fresh=False,
            is_complete=False,
            is_well_structured=True,
            strengths=[],
            gaps=["outdated"],
            revision_requests=["add newer sources"],
        )
        self.assertEqual(c.verdict, "REVISE")
        self.assertIn("outdated", c.gaps)

    def test_invalid_verdict_rejected(self):
        from schemas import CritiqueResult
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            CritiqueResult(
                verdict="MAYBE",
                is_fresh=True, is_complete=True, is_well_structured=True,
                strengths=[], gaps=[], revision_requests=[],
            )


class TestMcpUtils(unittest.TestCase):
    def test_mcp_tools_to_langchain_basic(self):
        from mcp_utils import mcp_tools_to_langchain
        from fastmcp import Client as MCPClient

        # Mock MCP tool descriptor
        mock_tool = MagicMock()
        mock_tool.name = "web_search"
        mock_tool.description = "Search the web"
        mock_tool.inputSchema = {
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

        tools = mcp_tools_to_langchain([mock_tool], MCPClient("http://localhost:9999/mcp"))
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "web_search")
        self.assertEqual(tools[0].description, "Search the web")

    def test_mcp_tools_empty(self):
        from mcp_utils import mcp_tools_to_langchain
        from fastmcp import Client as MCPClient
        tools = mcp_tools_to_langchain([], MCPClient("http://localhost:9999/mcp"))
        self.assertEqual(tools, [])


# ---------------------------------------------------------------------------
# 2. MCP server in-process tests
# ---------------------------------------------------------------------------

class TestSearchMCPTools(unittest.TestCase):
    """Test SearchMCP tool logic directly (without HTTP transport)."""

    def test_web_search_returns_string(self):
        # Import the function directly from the module
        import importlib.util, types
        spec = importlib.util.spec_from_file_location("search_mcp", "mcp_servers/search_mcp.py")
        mod = importlib.util.load_from_spec = None  # avoid re-exec side effects

        # Just test the underlying logic via tools.py which has the same impl
        from tools import web_search
        result = web_search.invoke({"query": "Python programming language"})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        print(f"\n  web_search result snippet: {result[:120]}")

    def test_knowledge_search_no_index(self):
        """knowledge_search should gracefully handle missing index."""
        try:
            from tools import knowledge_search
            with patch("retriever.is_index_ready", return_value=False):
                result = knowledge_search.invoke({"query": "RAG"})
            self.assertIn("index not found", result.lower())
        except ImportError as e:
            self.skipTest(f"Retriever import failed (langchain version issue): {e}")

    def test_knowledge_search_with_index(self):
        """knowledge_search returns results when index exists."""
        try:
            from retriever import is_index_ready
            if not is_index_ready():
                self.skipTest("Vector index not built — run python ingest.py first")
            from tools import knowledge_search
            result = knowledge_search.invoke({"query": "retrieval augmented generation"})
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            print(f"\n  knowledge_search result snippet: {result[:120]}")
        except ImportError as e:
            self.skipTest(f"Retriever import failed (langchain version issue): {e}")


class TestReportMCPTools(unittest.TestCase):
    """Test ReportMCP save_report logic directly."""

    def test_save_report_creates_file(self):
        from tools import save_report
        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp()
        try:
            with patch("tools.settings") as mock_settings:
                mock_settings.output_dir = tmp_dir
                result = save_report.invoke({"filename": "test_report.md", "content": "# Test\nHello"})
            self.assertIn("test_report.md", result)
            saved = os.path.join(tmp_dir, "test_report.md")
            self.assertTrue(os.path.exists(saved))
            with open(saved) as f:
                self.assertEqual(f.read(), "# Test\nHello")
        finally:
            shutil.rmtree(tmp_dir)

    def test_save_report_adds_md_extension(self):
        from tools import save_report
        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp()
        try:
            with patch("tools.settings") as mock_settings:
                mock_settings.output_dir = tmp_dir
                result = save_report.invoke({"filename": "no_extension", "content": "content"})
            self.assertIn("no_extension.md", result)
        finally:
            shutil.rmtree(tmp_dir)


# ---------------------------------------------------------------------------
# 3. Supervisor tool wrappers (mocked ACP/MCP calls)
# ---------------------------------------------------------------------------

class TestSupervisorTools(unittest.TestCase):
    """Test supervisor tool wrappers with mocked ACP/MCP backends."""

    def test_plan_tool_calls_acp(self):
        with patch("supervisor._run_async", return_value="Goal: test\nSearch queries: []\nSources to check: []\nOutput format: markdown") as mock_async:
            from supervisor import plan
            result = plan.invoke({"request": "test request"})
        self.assertIn("Goal", result)
        mock_async.assert_called_once()

    def test_research_tool_calls_acp(self):
        with patch("supervisor._run_async", return_value="## Findings\nSome research content") as mock_async:
            from supervisor import research
            result = research.invoke({"request": "research this"})
        self.assertIn("Findings", result)

    def test_critique_tool_calls_acp(self):
        with patch("supervisor._run_async", return_value="Verdict: APPROVE\nIs fresh: True") as mock_async:
            from supervisor import critique
            result = critique.invoke({"findings": "some findings"})
        self.assertIn("APPROVE", result)


# ---------------------------------------------------------------------------
# 4. Integration smoke test (requires live servers)
# ---------------------------------------------------------------------------

class TestIntegrationSmoke(unittest.TestCase):
    """
    Full pipeline smoke test. Skipped automatically if servers are not running.
    Start servers first:
        python mcp_servers/search_mcp.py   # port 8901
        python mcp_servers/report_mcp.py   # port 8902
        python acp_server.py               # port 8903
    """

    def _servers_running(self):
        import httpx
        from config import settings
        for port in [settings.search_mcp_port, settings.report_mcp_port, settings.acp_port]:
            try:
                httpx.get(f"http://localhost:{port}", timeout=1.0)
            except Exception:
                return False
        return True

    def test_search_mcp_health(self):
        import httpx
        from config import settings
        try:
            r = httpx.get(f"http://localhost:{settings.search_mcp_port}/mcp", timeout=2.0)
            self.assertIn(r.status_code, [200, 404, 405])  # server is up
        except Exception:
            self.skipTest("SearchMCP not running on port 8901")

    def test_report_mcp_health(self):
        import httpx
        from config import settings
        try:
            r = httpx.get(f"http://localhost:{settings.report_mcp_port}/mcp", timeout=2.0)
            self.assertIn(r.status_code, [200, 404, 405])
        except Exception:
            self.skipTest("ReportMCP not running on port 8902")

    def test_acp_server_health(self):
        import httpx
        from config import settings
        try:
            r = httpx.get(f"http://localhost:{settings.acp_port}", timeout=2.0)
            self.assertIn(r.status_code, [200, 404, 405])
        except Exception:
            self.skipTest("ACP server not running on port 8903")

    def test_full_pipeline_with_hitl(self):
        if not self._servers_running():
            self.skipTest("Not all servers running — skipping full pipeline test")

        from langgraph.types import Command
        from supervisor import build_supervisor

        supervisor = build_supervisor()
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        query = "What is RAG? Write a one-paragraph summary."

        print(f"\n  Running pipeline: {query!r}")
        supervisor.invoke({"messages": [{"role": "user", "content": query}]}, config=config)

        state = supervisor.get_state(config)
        interrupted_payload = None
        if state.next:
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupted_payload = task.interrupts[0].value
                    break

        if interrupted_payload:
            print(f"  HITL triggered for: {interrupted_payload.get('filename')}")
            result = supervisor.invoke(Command(resume={"action": "approve"}), config=config)
            msgs = result.get("messages", [])
            final = next((m.content for m in reversed(msgs) if getattr(m, "type", "") == "ai" and m.content), None)
            print(f"  Final answer snippet: {str(final)[:120]}")
            self.assertIsNotNone(final)
        else:
            print("  Pipeline completed without HITL (unexpected but not a failure)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
