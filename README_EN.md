[简体中文](./README.md) | English

# Acemcp

MCP server for codebase indexing and semantic search.

<a href="https://glama.ai/mcp/servers/@qy527145/acemcp">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@qy527145/acemcp/badge" alt="Acemcp MCP server" />
</a>

## Installation

### Install as Tool (Recommended)

```bash
# Install to system
uv tool install acemcp

# Or run temporarily (no installation required)
uvx acemcp
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/qy527145/acemcp.git
cd acemcp

# Install dependencies
uv sync

# Run
uv run acemcp
```

## Configuration

The configuration file is automatically created at `~/.acemcp/settings.toml` on first run with default values.

Edit `~/.acemcp/settings.toml` to configure:
```toml
BATCH_SIZE = 10
MAX_LINES_PER_BLOB = 800
BASE_URL = "https://your-api-endpoint.com"
TOKEN = "your-bearer-token-here"
TEXT_EXTENSIONS = [".py", ".js", ".ts", ...]
EXCLUDE_PATTERNS = [".venv", "node_modules", ".git", "__pycache__", "*.pyc", ...]
```

**Configuration options:**
- `BATCH_SIZE`: Number of files to upload per batch (default: 10)
- `MAX_LINES_PER_BLOB`: Maximum lines per blob before splitting large files (default: 800)
- `BASE_URL`: API endpoint URL
- `TOKEN`: Authentication token
- `TEXT_EXTENSIONS`: List of file extensions to index
- `EXCLUDE_PATTERNS`: List of patterns to exclude from indexing (supports wildcards like `*.pyc`)

You can also configure via:
- **Command line arguments** (highest priority): `--base-url`, `--token`
- **Web management interface** (updates user config file)
- **Environment variables** with `ACEMCP_` prefix

## MCP Configuration

Add the following to your MCP client configuration (e.g., Claude Desktop):

### Basic Configuration

```json
{
  "mcpServers": {
    "acemcp": {
      "command": "uvx",
      "args": [
        "acemcp"
      ]
    }
  }
}
```


**Available command line arguments:**
- `--base-url`: Override BASE_URL configuration
- `--token`: Override TOKEN configuration
- `--web-port`: Enable web management interface on specified port (e.g., 8080)

### Configuration with Web Management Interface

To enable the web management interface, add the `--web-port` argument:

```json
{
  "mcpServers": {
    "acemcp": {
      "command": "uvx",
      "args": [
        "acemcp",
        "--web-port",
        "8888"
      ]
    }
  }
}
```

Then access the management interface at `http://localhost:8888`

**Web Management Features:**
- **Configuration Management**: View and edit server configuration (BASE_URL, TOKEN, BATCH_SIZE, MAX_LINES_PER_BLOB, TEXT_EXTENSIONS)
- **Real-time Logs**: Monitor server logs in real-time via WebSocket connection with intelligent reconnection
  - Exponential backoff reconnection strategy (1s → 1.5s → 2.25s ... max 30s)
  - Maximum 10 reconnection attempts to prevent infinite loops
  - Automatic reconnection on network failures
  - Reduced log noise (WebSocket connections logged at DEBUG level)
- **Tool Debugger**: Test and debug MCP tools directly from the web interface
  - Test `search_context` tool with project path and query
  - View formatted results and error messages

## Tools

### search_context

Search for relevant code context based on a query. This tool **automatically performs incremental indexing** before searching, ensuring results are always up-to-date. It performs **semantic search** across your codebase and returns formatted text snippets showing where relevant code is located.

**Key Features:**
- **Automatic Incremental Indexing**: Before each search, the tool automatically indexes only new or modified files, skipping unchanged files for efficiency
- **No Manual Indexing Required**: You don't need to manually index your project - just search and the tool handles indexing automatically
- **Always Up-to-Date**: Search results reflect the current state of your codebase
- **Multi-Encoding Support**: Automatically detects and handles multiple file encodings (UTF-8, GBK, GB2312, Latin-1)
- **.gitignore Integration**: Automatically respects `.gitignore` patterns when indexing projects

**Parameters:**
- `project_root_path` (string): Absolute path to the project root directory
  - **IMPORTANT**: Use forward slashes (`/`) as path separators, even on Windows
  - Windows example: `C:/Users/username/projects/myproject`
  - Linux/Mac example: `/home/username/projects/myproject`
- `query` (string): Natural language search query to find relevant code context
  - Use descriptive keywords related to what you're looking for
  - The tool performs semantic matching, not just keyword search
  - Returns code snippets with file paths and line numbers

**What it returns:**
- Formatted text snippets from files that match your query
- File paths and line numbers for each snippet
- Context around the relevant code sections
- Multiple results ranked by relevance

**Query Examples:**

1. **Finding configuration code:**
   ```json
   {
     "project_root_path": "C:/Users/username/projects/myproject",
     "query": "logging configuration setup initialization logger"
   }
   ```
   Returns: Code related to logging setup, logger initialization, and configuration

2. **Finding authentication logic:**
   ```json
   {
     "project_root_path": "C:/Users/username/projects/myproject",
     "query": "user authentication login password validation"
   }
   ```
   Returns: Authentication handlers, login functions, password validation code

3. **Finding database code:**
   ```json
   {
     "project_root_path": "C:/Users/username/projects/myproject",
     "query": "database connection pool initialization"
   }
   ```
   Returns: Database connection setup, pool configuration, initialization code

4. **Finding error handling:**
   ```json
   {
     "project_root_path": "C:/Users/username/projects/myproject",
     "query": "error handling exception try catch"
   }
   ```
   Returns: Error handling patterns, exception handlers, try-catch blocks

5. **Finding API endpoints:**
   ```json
   {
     "project_root_path": "C:/Users/username/projects/myproject",
     "query": "API endpoint routes HTTP handlers"
   }
   ```
   Returns: API route definitions, HTTP handlers, endpoint implementations

**Tips for better results:**
- Use multiple related keywords (e.g., "logging configuration setup" instead of just "logging")
- Include technical terms specific to what you're looking for
- Describe the functionality rather than exact variable names
- Try different phrasings if the first query doesn't return what you need

**Indexing Features:**
- **Incremental Indexing**: Only new or modified files are uploaded, unchanged files are skipped
- **Hash-based Deduplication**: Files are identified by SHA-256 hash of path + content
- **Automatic Retry**: Network requests are automatically retried up to 3 times with exponential backoff (1s, 2s, 4s)
- **Batch Resilience**: If a batch upload fails after retries, the tool continues with the next batch
- **File Splitting**: Large files are automatically split into multiple blobs (default: 800 lines per blob)
- **Exclude Patterns**: Automatically skips virtual environments, node_modules, .git, build artifacts, etc.
- **Multi-Encoding Support**: Automatically detects file encoding (UTF-8, GBK, GB2312, Latin-1) with fallback to UTF-8 with error handling
- **.gitignore Integration**: Automatically loads and respects `.gitignore` patterns from project root, combined with configured exclude patterns

**Search Features:**
- **Automatic Retry**: Search requests are automatically retried up to 3 times with exponential backoff (2s, 4s, 8s)
- **Graceful Degradation**: Returns a clear error message if the search fails after all retries
- **Timeout Handling**: Uses a 60-second timeout to handle long-running searches
- **Empty Result Handling**: Returns a helpful message if no relevant code is found

**Default Exclude Patterns:**
```
.venv, venv, .env, env, node_modules, .git, .svn, .hg, __pycache__,
.pytest_cache, .mypy_cache, .tox, .eggs, *.egg-info, dist, build,
.idea, .vscode, .DS_Store, *.pyc, *.pyo, *.pyd, .Python,
pip-log.txt, pip-delete-this-directory.txt, .coverage, htmlcov,
.gradle, target, bin, obj
```
Patterns support wildcards (`*`, `?`) and match against directory/file names or paths.

**Note:** If a `.gitignore` file exists in the project root, its patterns will be automatically loaded and combined with the configured exclude patterns. The `.gitignore` patterns follow Git's standard wildmatch syntax.

## Advanced Features

### Multi-Encoding File Support

Acemcp automatically detects and handles files with different character encodings, making it suitable for international projects:

- **Automatic Detection**: Tries multiple encodings in order: UTF-8 → GBK → GB2312 → Latin-1
- **Fallback Handling**: If all encodings fail, uses UTF-8 with error handling to prevent crashes
- **Logging**: Records which encoding was successfully used for each file (DEBUG level)
- **No Configuration Required**: Works out of the box for most common encodings

This is particularly useful for:
- Projects with mixed encoding files (e.g., UTF-8 source code + GBK documentation)
- Legacy codebases using non-UTF-8 encodings
- International teams with files in different languages

### .gitignore Integration

Acemcp automatically respects your project's `.gitignore` file:

- **Automatic Loading**: Reads `.gitignore` from project root if it exists
- **Standard Syntax**: Supports Git's standard wildmatch patterns
- **Combined Filtering**: Works alongside configured `EXCLUDE_PATTERNS`
- **Directory Handling**: Properly handles directory patterns with trailing slashes
- **No Configuration Required**: Just place a `.gitignore` in your project root

**Example `.gitignore` patterns:**
```gitignore
# Dependencies
node_modules/
vendor/

# Build outputs
dist/
build/
*.pyc

# IDE files
.vscode/
.idea/

# Environment files
.env
.env.local
```

All these patterns will be automatically respected during indexing, in addition to the default exclude patterns.

## Usage

1. Start the MCP server (automatically started by MCP client)
2. Use `search_context` to search for code context
   - The tool automatically indexes your project before searching
   - Incremental indexing ensures only new/modified files are uploaded
   - No manual indexing step required!
   - Files are automatically handled regardless of encoding
   - `.gitignore` patterns are automatically respected

## Data Storage

- **Configuration**: `~/.acemcp/settings.toml`
- **Indexed projects**: `~/.acemcp/data/projects.json` (fixed location)
- **Log files**: `~/.acemcp/log/acemcp.log` (with automatic rotation)
- Projects are identified by their absolute path (normalized with forward slashes)

## Logging

The application automatically logs to `~/.acemcp/log/acemcp.log` with the following features:

- **Console output**: INFO level and above (colored output)
- **File output**: DEBUG level and above (detailed format with module, function, and line number)
- **Automatic rotation**: Log files are rotated when they reach 5MB
- **Retention**: Maximum of 10 log files are kept
- **Compression**: Rotated log files are automatically compressed to `.zip` format
- **Thread-safe**: Logging is thread-safe for concurrent operations

**Log format:**
```
2025-11-06 13:51:25 | INFO     | acemcp.server:main:103 - Starting acemcp MCP server...
```

The log files are automatically created on first run and require no manual configuration.

## Web Management Interface

The web management interface provides:
- **Real-time server status** monitoring
- **Live log streaming** via WebSocket
- **Configuration management**: View and edit server configuration
- **Token validation**: One-click API Key validation
- **Project statistics**: Number of indexed projects
- **Tool debugger**: Test and debug MCP tools directly from the web interface

To enable the web interface, use the `--web-port` argument when starting the server.

**Features:**
- Real-time log display with auto-scroll
- Server status and metrics
- Configuration overview and editing
- Responsive design with Tailwind CSS
- No build step required (uses CDN resources)
- Intelligent WebSocket reconnection with exponential backoff

## Recent Updates

### Version 0.2.2

**New Features:**
- ✨ **Upload Exception Handling System**: Brand new intelligent exception handling mechanism
  - **Binary Search Localization**: Automatically uses binary search to precisely locate failed blobs when batch uploads fail
  - **Failed Blob Storage**: Records detailed information of failed blobs (file path, hash, error message, timestamp)
  - **Intelligent Recovery**: Recovers successful blobs from failed batches, maximizing upload success rate
  - **Search Optimization**: Automatically excludes failed blobs during search to ensure result reliability
- ✨ **Enhanced Exception Logging**: Detailed logging of each blob's file path and hash value
- ✨ **Web Console Integration**: New `/api/failed-blobs` API endpoint supporting web console display of exception information

**Technical Details:**
- Implemented `_binary_search_failed_blobs()` recursive binary search algorithm with O(log n) complexity
- Added `failed_blobs.json` storage file to record project failed blob information
- Search functionality automatically filters failed blobs with statistical information
- Complete exception handling workflow: Detection → Localization → Recording → Exclusion

### Version 0.2.1

**Improvements:**
- 🔧 Optimized the prompt description for `search_context` tool
- 🔧 Adjusted tool parameter descriptions

### Version 0.2.0

**Bug Fixes:**
- 🐛 Fixed an issue where acemcp failed to start when a `.env` file with encoding errors existed in the project

**Dependency Updates:**
- ⬆️ Upgraded third-party dependency package versions


### Version 0.1.9

**New Features:**
  - Automatically detects if the web port is in use; if so, it reuses the web panel.

**Improvements:**
- [fix Google Antigravity MCP "invalid trailing data" error](https://github.com/qy527145/acemcp/pull/12)

### Version 0.1.8

**New Features:**
- ✨ **Token Validation Feature**: Web management interface now includes API Key validation button
  - Added "Validate Key" button in configuration section to instantly verify token validity
  - Supports token validation in both view and edit modes
  - Provides clear validation result feedback (success/failure messages)
  - Helps users quickly diagnose API configuration issues

**Technical Details:**
- New `/api/validate-token` API endpoint
- Validates token validity by sending test requests to the API
- Comprehensive error handling: 401 Unauthorized, 403 Forbidden, timeout, connection errors, etc.
- Supports both English and Chinese interfaces

### Version 0.1.7

**Improvements:**
- 🔧 **API Request Optimization**: https://github.com/qy527145/acemcp/pull/6
- 🔧 **Proxy Environment Compatibility**: Added httpx[socks] extension dependency to fix errors in proxy environments

### Version 0.1.6

**Improvements:**
- 🔧 **API Request Optimization**: Improved request handling
- 🔧 **Proxy Environment Support**: Better compatibility with proxy configurations

### Version 0.1.5

**New Features:**
- ✨ **Logging System Optimization**: Redirect FastAPI/Uvicorn logs to loguru to prevent pollution of MCP stdio protocol
- ✨ **Tool Debugging Interface**: Web management interface now includes tool listing and debugging functionality

**Improvements:**
- 🔧 **Log Output Control**: Removed console log output, only output to file to avoid interfering with stdio protocol
- 🔧 **Standard Library Log Interception**: Use `InterceptHandler` to intercept all standard library logs
- 🔧 **Web API Enhancement**: New `/api/tools` endpoint to list available tools

**Technical Details:**
- Implemented `InterceptHandler` class to intercept standard library logging
- Configured uvicorn with `log_config=None` to disable default logging
- All logs unified to output to `~/.acemcp/log/acemcp.log`

### Version 0.1.4

**New Features:**
- ✨ **Multi-Encoding Support**: Automatic detection and handling of multiple file encodings (UTF-8, GBK, GB2312, Latin-1)
- ✨ **.gitignore Integration**: Automatic loading and respect of `.gitignore` patterns from project root
- ✨ **Improved Tool Response Format**: Changed from list-based to dictionary-based response format for better client compatibility

**Improvements:**
- 🔧 **WebSocket Optimization**: Intelligent reconnection with exponential backoff (1s → 30s max)
- 🔧 **Reduced Log Noise**: WebSocket connections now logged at DEBUG level instead of INFO
- 🔧 **Connection Stability**: Maximum 10 reconnection attempts to prevent infinite loops
- 🔧 **Better Error Handling**: Graceful fallback for files that can't be decoded with any encoding

**Bug Fixes:**
- 🐛 Fixed frequent WebSocket connection/disconnection cycles
- 🐛 Fixed encoding errors when reading files with non-UTF-8 encodings
- 🐛 Improved handling of .gitignore patterns with directory matching

