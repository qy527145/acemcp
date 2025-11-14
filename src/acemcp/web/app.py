"""FastAPI web application for MCP server management."""

import asyncio
import json
import toml
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from acemcp.config import get_config
from acemcp.web.log_handler import get_log_broadcaster

# Initialize log broadcaster at module level to ensure single instance
log_broadcaster = get_log_broadcaster()


class ConfigUpdate(BaseModel):
    """Configuration update model."""

    base_url: str | None = None
    token: str | None = None
    batch_size: int | None = None
    max_lines_per_blob: int | None = None
    text_extensions: list[str] | None = None
    exclude_patterns: list[str] | None = None


class ToolRequest(BaseModel):
    """Tool execution request model."""

    tool_name: str
    arguments: dict


def create_app() -> FastAPI:
    """Create FastAPI application.

    Returns:
        FastAPI application instance
    """
    app = FastAPI(title="Acemcp Management", description="MCP Server Management Interface", version="0.1.0")

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        """Serve the main management page."""
        html_file = Path(__file__).parent / "templates" / "index.html"
        if html_file.exists():
            return html_file.read_text(encoding="utf-8")
        return "<h1>Acemcp Management</h1><p>Template not found</p>"

    @app.get("/api/config")
    async def get_config_api() -> dict:
        """Get current configuration."""
        config = get_config()
        return {
            "index_storage_path": str(config.index_storage_path),
            "batch_size": config.batch_size,
            "max_lines_per_blob": config.max_lines_per_blob,
            "base_url": config.base_url,
            "token": "***" if config.token else "",
            "token_full": config.token,
            "text_extensions": list(config.text_extensions),
            "exclude_patterns": config.exclude_patterns,
        }

    @app.post("/api/config")
    async def update_config_api(config_update: ConfigUpdate) -> dict:
        """Update configuration.

        Args:
            config_update: Configuration updates

        Returns:
            Updated configuration
        """
        try:
            from acemcp.config import USER_CONFIG_FILE

            if not USER_CONFIG_FILE.exists():
                msg = "User configuration file not found"
                raise HTTPException(status_code=404, detail=msg)

            with USER_CONFIG_FILE.open("r", encoding="utf-8") as f:
                settings_data = toml.load(f)

            if config_update.base_url is not None:
                settings_data["BASE_URL"] = config_update.base_url
            if config_update.token is not None:
                settings_data["TOKEN"] = config_update.token
            if config_update.batch_size is not None:
                settings_data["BATCH_SIZE"] = config_update.batch_size
            if config_update.max_lines_per_blob is not None:
                settings_data["MAX_LINES_PER_BLOB"] = config_update.max_lines_per_blob
            if config_update.text_extensions is not None:
                settings_data["TEXT_EXTENSIONS"] = config_update.text_extensions
            if config_update.exclude_patterns is not None:
                settings_data["EXCLUDE_PATTERNS"] = config_update.exclude_patterns

            with USER_CONFIG_FILE.open("w", encoding="utf-8") as f:
                toml.dump(settings_data, f)

            config = get_config()
            config.reload()

            logger.info("Configuration updated and reloaded successfully")
            return {"status": "success", "message": "Configuration updated and applied successfully!"}

        except Exception as e:
            logger.exception("Failed to update configuration")
            msg = f"Failed to update configuration: {e!s}"
            raise HTTPException(status_code=500, detail=msg) from e

    @app.get("/api/status")
    async def get_status() -> dict:
        """Get server status."""
        config = get_config()
        projects_file = config.index_storage_path / "projects.json"
        project_count = 0
        if projects_file.exists():
            import json

            try:
                with projects_file.open("r", encoding="utf-8") as f:
                    projects = json.load(f)
                    project_count = len(projects)
            except Exception:
                logger.exception("Failed to load projects")

        return {"status": "running", "project_count": project_count, "storage_path": str(config.index_storage_path)}

    @app.get("/api/tools")
    async def list_tools() -> dict:
        """List available tools for debugging.

        Returns:
            Dictionary containing available tools and their descriptions
        """
        return {
            "tools": [
                {
                    "name": "search_context",
                    "description": "Search for code context in indexed projects",
                    "status": "stable",
                    "parameters": {
                        "project_root_path": "string (required) - Absolute path to project root",
                        "query": "string (required) - Search query",
                    },
                },
            ]
        }

    @app.post("/api/tools/execute")
    async def execute_tool(tool_request: ToolRequest) -> dict:
        """Execute a tool for debugging.

        Args:
            tool_request: Tool execution request

        Returns:
            Tool execution result
        """
        try:
            from acemcp.tools import search_context_tool

            tool_name = tool_request.tool_name
            arguments = tool_request.arguments

            logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")

            if tool_name == "search_context":
                result = await search_context_tool(arguments)
            else:
                return {"status": "error", "message": f"Unknown tool: {tool_name}"}

            logger.info(f"Tool {tool_name} executed successfully")
            return {"status": "success", "result": result}

        except Exception as e:
            logger.exception(f"Failed to execute tool {tool_request.tool_name}")
            return {"status": "error", "message": str(e)}

    @app.websocket("/ws/logs")
    async def websocket_logs(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time logs."""
        await websocket.accept()
        queue: asyncio.Queue = asyncio.Queue()
        log_broadcaster.add_client(queue)
        logger.debug("WebSocket client connected")

        try:
            while True:
                log_message = await queue.get()
                await websocket.send_text(log_message)
        except WebSocketDisconnect:
            logger.debug("WebSocket client disconnected normally")
        except Exception as e:
            logger.warning(f"WebSocket error: {e}")
        finally:
            log_broadcaster.remove_client(queue)

    @app.on_event("shutdown")
    async def shutdown_tools() -> None:
        """Release shared tool resources when the web app stops."""
        from acemcp.tools import shutdown_index_manager

        await shutdown_index_manager()

    return app
