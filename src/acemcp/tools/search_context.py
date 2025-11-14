"""Search context tool for MCP server."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from acemcp.config import get_config
from acemcp.index import IndexManager

_index_manager: IndexManager | None = None
_index_manager_lock: asyncio.Lock | None = None


async def _get_index_manager() -> IndexManager:
    """Create or return the shared IndexManager instance."""
    global _index_manager, _index_manager_lock

    if _index_manager is not None:
        return _index_manager

    if _index_manager_lock is None:
        _index_manager_lock = asyncio.Lock()

    async with _index_manager_lock:
        if _index_manager is None:
            config = get_config()
            _index_manager = IndexManager(
                config.index_storage_path,
                config.base_url,
                config.token,
                config.text_extensions,
                config.batch_size,
                config.max_lines_per_blob,
                config.exclude_patterns,
            )

    return _index_manager


async def shutdown_index_manager() -> None:
    """Close the shared IndexManager instance."""
    global _index_manager, _index_manager_lock

    if _index_manager_lock is None:
        _index_manager_lock = asyncio.Lock()

    async with _index_manager_lock:
        if _index_manager is not None:
            await _index_manager.close()
            _index_manager = None


async def search_context_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Search for code context based on query.

    Args:
        arguments: Tool arguments containing:
            - project_root_path: Absolute path to the project root directory
            - query: Search query string

    Returns:
        Dictionary containing search results
    """
    try:
        project_root_path = arguments.get("project_root_path")
        query = arguments.get("query")

        if not project_root_path:
            return {"type": "text", "text": "Error: project_root_path is required"}

        if not query:
            return {"type": "text", "text": "Error: query is required"}

        logger.info(f"Tool invoked: search_context for project {project_root_path} with query: {query}")

        index_manager = await _get_index_manager()
        result = await index_manager.search_context(project_root_path, query)

        return {"type": "text", "text": result}

    except Exception as e:
        logger.exception("Error in search_context_tool")
        return {"type": "text", "text": f"Error: {e!s}"}
