"""
Helper to convert MCP tools (from fastmcp.Client) into LangChain-compatible tools.
"""
from langchain_core.tools import StructuredTool


def mcp_tools_to_langchain(mcp_tools, client) -> list:
    """
    Convert a list of MCP tool descriptors into LangChain StructuredTools.
    Each tool call is dispatched via the fastmcp Client.
    """
    lc_tools = []
    for t in mcp_tools:
        tool_name = t.name
        tool_desc = t.description or tool_name

        # Build a pydantic model from the MCP input schema so LangChain can validate args
        from pydantic import create_model
        import json

        schema = t.inputSchema if hasattr(t, "inputSchema") else {}
        props = schema.get("properties", {})
        required = schema.get("required", [])

        fields = {}
        for prop_name, prop_info in props.items():
            annotation = str  # default to str for all fields
            if prop_info.get("type") == "integer":
                annotation = int
            elif prop_info.get("type") == "boolean":
                annotation = bool
            default = ... if prop_name in required else None
            fields[prop_name] = (annotation, default)

        ArgsModel = create_model(f"{tool_name}_args", **fields)

        def make_func(name, c):
            import asyncio

            def call_tool(**kwargs):
                async def _call():
                    async with c:
                        result = await c.call_tool(name, kwargs)
                        # result is a list of content objects
                        texts = []
                        for item in result:
                            if hasattr(item, "text"):
                                texts.append(item.text)
                            else:
                                texts.append(str(item))
                        return "\n".join(texts)

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            future = pool.submit(asyncio.run, _call())
                            return future.result()
                    else:
                        return loop.run_until_complete(_call())
                except Exception as e:
                    return f"MCP tool error: {str(e)}"

            return call_tool

        lc_tools.append(
            StructuredTool.from_function(
                func=make_func(tool_name, client),
                name=tool_name,
                description=tool_desc,
                args_schema=ArgsModel,
            )
        )
    return lc_tools
