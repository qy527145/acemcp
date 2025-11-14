"""MCP server for codebase indexing."""

import argparse
import asyncio

import uvicorn
from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from acemcp.config import get_config, init_config
from acemcp.logging_config import setup_logging
from acemcp.tools import search_context_tool, shutdown_index_manager
from acemcp.web import create_app

app = Server("acemcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools.

    Returns:
        List of available tools
    """
    return [
        Tool(
            name="search_context",
            description="Search for relevant code context based on a query within a specific project. This tool automatically performs incremental indexing before searching, ensuring results are always up-to-date. Returns formatted text snippets from the codebase that are semantically related to your query. IMPORTANT: Use forward slashes (/) as path separators in project_root_path, even on Windows.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_root_path": {
                        "type": "string",
                        "description": "Absolute path to the project root directory. Use forward slashes (/) as separators. Example: C:/Users/username/projects/myproject",
                    },
                    "query": {
                        "type": "string",
                        "description": "Natural language search query to find relevant code context. This tool performs semantic search and returns code snippets that match your query. Examples: 'logging configuration setup initialization logger' (finds logging setup code), 'user authentication login' (finds auth-related code), 'database connection pool' (finds DB connection code), 'error handling exception' (finds error handling patterns), 'API endpoint routes' (finds API route definitions). The tool returns formatted text snippets with file paths and line numbers showing where the relevant code is located.",
                    },
                },
                "required": ["project_root_path", "query"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> dict:
    """Handle tool calls.

    Args:
        name: Tool name
        arguments: Tool arguments

    Returns:
        Tool execution results
    """
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    if name == "search_context":
        return await search_context_tool(arguments)

    return {"type": "text", "text": f"Unknown tool: {name}"}


async def run_web_server(port: int) -> None:
    """Run the web management server.

    Args:
        port: Port to run the web server on
    """
    web_app = create_app()
    # Configure uvicorn to use loguru through InterceptHandler
    # This prevents uvicorn from polluting stdout (which breaks MCP stdio protocol)
    config_uvicorn = uvicorn.Config(
        web_app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,  # Disable access log to reduce noise
        log_config=None,   # Disable default logging config to use our interceptor
    )
    server = uvicorn.Server(config_uvicorn)
    await server.serve()


async def main(base_url: str | None = None, token: str | None = None, web_port: int | None = None) -> None:
    """Run the MCP server.

    Args:
        base_url: Override BASE_URL from command line
        token: Override TOKEN from command line
        web_port: Port for web management interface (None to disable)
    """
    web_task: asyncio.Task | None = None
    try:
        config = init_config(base_url=base_url, token=token)
        config.validate()
        logger.info("Starting acemcp MCP server...")
        logger.info(f"Configuration: index_storage_path={config.index_storage_path}, batch_size={config.batch_size}")
        logger.info(f"API: base_url={config.base_url}")

        if web_port:
            logger.info(f"Starting web management interface on port {web_port}")
            web_task = asyncio.create_task(run_web_server(web_port))

        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    except Exception:
        logger.exception("Server error")
        raise
    finally:
        if web_task:
            web_task.cancel()
        await shutdown_index_manager()


def run() -> None:
    """Entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="Acemcp MCP Server for codebase indexing")
    parser.add_argument("--base-url", type=str, help="Override BASE_URL configuration")
    parser.add_argument("--token", type=str, help="Override TOKEN configuration")
    parser.add_argument("--web-port", type=int, help="Enable web management interface on specified port (e.g., 8080)")

    args = parser.parse_args()

    # If web interface is enabled, initialize log broadcaster before setting up logging
    # This ensures the WebSocket handler is preserved
    if args.web_port:
        from acemcp.web.log_handler import get_log_broadcaster
        get_log_broadcaster()  # Initialize the broadcaster

    # Setup logging after log broadcaster is initialized
    # Intercept stdlib logging (uvicorn, fastapi) to prevent stdout pollution
    setup_logging(intercept_stdlib=True)

    asyncio.run(main(base_url=args.base_url, token=args.token, web_port=args.web_port))


if __name__ == "__main__":
    run()
