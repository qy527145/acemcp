[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/qy527145-acemcp-badge.png)](https://mseep.ai/app/qy527145-acemcp)

简体中文 | [English](./README_EN.md)

# Acemcp

代码库索引和语义搜索的 MCP 服务器。

<a href="https://glama.ai/mcp/servers/@qy527145/acemcp">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@qy527145/acemcp/badge" alt="Acemcp MCP server" />
</a>

## 安装

### 作为工具安装（推荐）

```bash
# 安装到系统
uv tool install acemcp

# 或临时运行（无需安装）
uvx acemcp
```

### 开发安装

```bash
# 克隆仓库
git clone https://github.com/qy527145/acemcp.git
cd acemcp

# 安装依赖
uv sync

# 运行
uv run acemcp
```

## 配置

配置文件会在首次运行时自动创建在 `~/.acemcp/settings.toml`，包含默认值。

编辑 `~/.acemcp/settings.toml` 进行配置：
```toml
BATCH_SIZE = 10
MAX_LINES_PER_BLOB = 800
BASE_URL = "https://your-api-endpoint.com"
TOKEN = "your-bearer-token-here"
TEXT_EXTENSIONS = [".py", ".js", ".ts", ...]
EXCLUDE_PATTERNS = [".venv", "node_modules", ".git", "__pycache__", "*.pyc", ...]
```

**配置选项：**
- `BATCH_SIZE`: 每批上传的文件数量（默认：10）
- `MAX_LINES_PER_BLOB`: 大文件分割前的最大行数（默认：800）
- `BASE_URL`: API 端点 URL
- `TOKEN`: 认证令牌
- `TEXT_EXTENSIONS`: 要索引的文件扩展名列表
- `EXCLUDE_PATTERNS`: 要排除的模式列表（支持通配符如 `*.pyc`）

您还可以通过以下方式配置：
- **命令行参数**（最高优先级）：`--base-url`、`--token`
- **Web 管理界面**（更新用户配置文件）
- **环境变量**（使用 `ACEMCP_` 前缀）

## MCP 配置

将以下内容添加到您的 MCP 客户端配置中（例如 Claude Desktop）：

### 基础配置

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


**可用的命令行参数：**
- `--base-url`: 覆盖 BASE_URL 配置
- `--token`: 覆盖 TOKEN 配置
- `--web-port`: 在指定端口启用 Web 管理界面（例如 8080）

### 启用 Web 管理界面的配置

要启用 Web 管理界面，添加 `--web-port` 参数：

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

然后访问管理界面：`http://localhost:8888`

**Web 管理功能：**
- **配置管理**：查看和编辑服务器配置（BASE_URL、TOKEN、BATCH_SIZE、MAX_LINES_PER_BLOB、TEXT_EXTENSIONS）
- **实时日志**：通过 WebSocket 连接实时监控服务器日志，具有智能重连功能
  - 指数退避重连策略（1秒 → 1.5秒 → 2.25秒 ... 最大 30秒）
  - 最多 10 次重连尝试，防止无限循环
  - 网络故障时自动重连
  - 减少日志噪音（WebSocket 连接记录在 DEBUG 级别）
- **工具调试器**：直接从 Web 界面测试和调试 MCP 工具
  - 测试 `search_context` 工具，输入项目路径和查询
  - 查看格式化的结果和错误消息

## 工具

### search_context

基于查询搜索相关的代码上下文。此工具在搜索前**自动执行增量索引**，确保结果始终是最新的。它在您的代码库中执行**语义搜索**，并返回格式化的文本片段，显示相关代码的位置。

**核心特性：**
- **自动增量索引**：每次搜索前，工具自动仅索引新文件或修改过的文件，跳过未更改的文件以提高效率
- **无需手动索引**：您无需手动索引项目 - 只需搜索，工具会自动处理索引
- **始终保持最新**：搜索结果反映代码库的当前状态
- **多编码支持**：自动检测和处理多种文件编码（UTF-8、GBK、GB2312、Latin-1）
- **.gitignore 集成**：索引项目时自动遵守 `.gitignore` 模式

**参数：**
- `project_root_path`（字符串）：项目根目录的绝对路径
  - **重要**：即使在 Windows 上也使用正斜杠（`/`）作为路径分隔符
  - Windows 示例：`C:/Users/username/projects/myproject`
  - Linux/Mac 示例：`/home/username/projects/myproject`
- `query`（字符串）：用于查找相关代码上下文的自然语言搜索查询
  - 使用与您要查找的内容相关的描述性关键词
  - 工具执行语义匹配，而不仅仅是关键词搜索
  - 返回带有文件路径和行号的代码片段

**返回内容：**
- 与您的查询匹配的文件中的格式化文本片段
- 每个片段的文件路径和行号
- 相关代码部分周围的上下文
- 按相关性排序的多个结果

**查询示例：**

1. **查找配置代码：**
   ```json
   {
     "project_root_path": "C:/Users/username/projects/myproject",
     "query": "日志配置 设置 初始化 logger"
   }
   ```
   返回：与日志设置、logger 初始化和配置相关的代码

2. **查找认证逻辑：**
   ```json
   {
     "project_root_path": "C:/Users/username/projects/myproject",
     "query": "用户认证 登录 密码验证"
   }
   ```
   返回：认证处理器、登录函数、密码验证代码

3. **查找数据库代码：**
   ```json
   {
     "project_root_path": "C:/Users/username/projects/myproject",
     "query": "数据库连接池 初始化"
   }
   ```
   返回：数据库连接设置、连接池配置、初始化代码

4. **查找错误处理：**
   ```json
   {
     "project_root_path": "C:/Users/username/projects/myproject",
     "query": "错误处理 异常 try catch"
   }
   ```
   返回：错误处理模式、异常处理器、try-catch 块

5. **查找 API 端点：**
   ```json
   {
     "project_root_path": "C:/Users/username/projects/myproject",
     "query": "API 端点 路由 HTTP 处理器"
   }
   ```
   返回：API 路由定义、HTTP 处理器、端点实现

**获得更好结果的技巧：**
- 使用多个相关关键词（例如，"日志配置设置"而不仅仅是"日志"）
- 包含您要查找的特定技术术语
- 描述功能而不是确切的变量名
- 如果第一次查询没有返回您需要的内容，尝试不同的措辞

**索引特性：**
- **增量索引**：仅上传新文件或修改过的文件，跳过未更改的文件
- **基于哈希的去重**：通过路径 + 内容的 SHA-256 哈希识别文件
- **自动重试**：网络请求自动重试最多 3 次，采用指数退避（1秒、2秒、4秒）
- **批次弹性**：如果批次上传在重试后失败，工具会继续处理下一批次
- **文件分割**：大文件自动分割为多个块（默认：每块 800 行）
- **排除模式**：自动跳过虚拟环境、node_modules、.git、构建产物等
- **多编码支持**：自动检测文件编码（UTF-8、GBK、GB2312、Latin-1），并在失败时回退到 UTF-8 错误处理
- **.gitignore 集成**：自动从项目根目录加载并遵守 `.gitignore` 模式，与配置的排除模式结合使用

**搜索特性：**
- **自动重试**：搜索请求自动重试最多 3 次，采用指数退避（2秒、4秒、8秒）
- **优雅降级**：如果所有重试后搜索失败，返回清晰的错误消息
- **超时处理**：使用 60 秒超时来处理长时间运行的搜索
- **空结果处理**：如果未找到相关代码，返回有用的消息

**默认排除模式：**
```
.venv, venv, .env, env, node_modules, .git, .svn, .hg, __pycache__,
.pytest_cache, .mypy_cache, .tox, .eggs, *.egg-info, dist, build,
.idea, .vscode, .DS_Store, *.pyc, *.pyo, *.pyd, .Python,
pip-log.txt, pip-delete-this-directory.txt, .coverage, htmlcov,
.gradle, target, bin, obj
```
模式支持通配符（`*`、`?`），并匹配目录/文件名或路径。

**注意：** 如果项目根目录存在 `.gitignore` 文件，其模式将自动加载并与配置的排除模式结合使用。`.gitignore` 模式遵循 Git 的标准 wildmatch 语法。

## 高级特性

### 多编码文件支持

Acemcp 自动检测和处理不同字符编码的文件，适用于国际化项目：

- **自动检测**：按顺序尝试多种编码：UTF-8 → GBK → GB2312 → Latin-1
- **回退处理**：如果所有编码都失败，使用 UTF-8 错误处理以防止崩溃
- **日志记录**：记录每个文件成功使用的编码（DEBUG 级别）
- **无需配置**：开箱即用，支持大多数常见编码

这对以下情况特别有用：
- 混合编码文件的项目（例如，UTF-8 源代码 + GBK 文档）
- 使用非 UTF-8 编码的遗留代码库
- 具有不同语言文件的国际团队

### .gitignore 集成

Acemcp 自动遵守您项目的 `.gitignore` 文件：

- **自动加载**：如果存在，从项目根目录读取 `.gitignore`
- **标准语法**：支持 Git 的标准 wildmatch 模式
- **组合过滤**：与配置的 `EXCLUDE_PATTERNS` 一起工作
- **目录处理**：正确处理带有尾部斜杠的目录模式
- **无需配置**：只需在项目根目录放置 `.gitignore`

**`.gitignore` 模式示例：**
```gitignore
# 依赖
node_modules/
vendor/

# 构建输出
dist/
build/
*.pyc

# IDE 文件
.vscode/
.idea/

# 环境文件
.env
.env.local
```

所有这些模式在索引期间都会自动遵守，并与默认排除模式结合使用。

## 使用方法

1. 启动 MCP 服务器（由 MCP 客户端自动启动）
2. 使用 `search_context` 搜索代码上下文
   - 工具在搜索前自动索引您的项目
   - 增量索引确保仅上传新文件/修改过的文件
   - 无需手动索引步骤！
   - 无论编码如何，文件都会自动处理
   - 自动遵守 `.gitignore` 模式

## 数据存储

- **配置**：`~/.acemcp/settings.toml`
- **已索引项目**：`~/.acemcp/data/projects.json`（固定位置）
- **日志文件**：`~/.acemcp/log/acemcp.log`（自动轮转）
- 项目通过其绝对路径识别（使用正斜杠规范化）

## 日志记录

应用程序自动记录日志到 `~/.acemcp/log/acemcp.log`，具有以下特性：

- **控制台输出**：INFO 级别及以上（彩色输出）
- **文件输出**：DEBUG 级别及以上（详细格式，包含模块、函数和行号）
- **自动轮转**：日志文件达到 5MB 时自动轮转
- **保留策略**：最多保留 10 个日志文件
- **压缩**：轮转的日志文件自动压缩为 `.zip` 格式
- **线程安全**：日志记录对并发操作是线程安全的

**日志格式：**
```
2025-11-06 13:51:25 | INFO     | acemcp.server:main:103 - Starting acemcp MCP server...
```

日志文件在首次运行时自动创建，无需手动配置。

## Web 管理界面

Web 管理界面提供：
- **实时服务器状态**监控
- **实时日志流**通过 WebSocket
- **配置管理**：查看和编辑服务器配置
- **Token 验证**：一键检测 API Key 是否有效
- **项目统计**：已索引项目数量
- **工具调试器**：直接从 Web 界面测试和调试 MCP 工具

要启用 Web 界面，在启动服务器时使用 `--web-port` 参数。

**功能：**
- 带自动滚动的实时日志显示
- 服务器状态和指标
- 配置概览和编辑
- 使用 Tailwind CSS 的响应式设计
- 无需构建步骤（使用 CDN 资源）
- 具有指数退避的智能 WebSocket 重连

## 最近更新

### 版本 0.2.2

**新特性：**
- ✨ **上传异常处理系统**：全新的智能异常处理机制
  - **二分法定位**：当批量上传失败时，自动使用二分法精确定位异常的 blob
  - **异常 blob 存储**：记录异常 blob 的详细信息（文件路径、hash、错误信息、时间戳）
  - **智能恢复**：从失败批次中恢复成功的 blobs，最大化上传成功率
  - **搜索优化**：搜索时自动排除异常 blobs，确保搜索结果可靠性
- ✨ **增强的异常日志**：详细记录每个 blob 的文件路径和 hash 值
- ✨ **Web 控制台集成**：新增 `/api/failed-blobs` API 接口，支持 web 控制台显示异常信息

**技术细节：**
- 实现了 `_binary_search_failed_blobs()` 递归二分法算法，O(log n) 复杂度
- 新增 `failed_blobs.json` 存储文件，记录项目异常 blob 信息
- 搜索功能自动过滤异常 blobs，提供统计信息
- 完整的异常处理工作流：检测 → 定位 → 记录 → 排除

### 版本 0.2.1

**改进：**
- 🔧 优化了 `search_context` 工具的提示词描述
- 🔧 调整了工具参数的说明文字

### 版本 0.2.0

**错误修复：**
- 🐛 修复了项目中存在 `.env` 文件时，因编码错误导致 acemcp 启动失败的问题

**依赖更新：**
- ⬆️ 升级第三方依赖包版本

### 版本 0.1.9

**新特性：**
- 自动判断web-port端口是否占用，如果占用会复用web面板

**改进：**
- [修复Antigravity兼容性错误](https://github.com/qy527145/acemcp/pull/12)

### 版本 0.1.8

**新特性：**
- ✨ **Token 验证功能**：Web 管理界面新增 API Key 检测按钮
  - 在配置部分添加"检测 Key"按钮，可即时验证 token 是否有效
  - 支持在查看模式和编辑模式下验证 token
  - 提供清晰的验证结果反馈（成功/失败消息）
  - 帮助用户快速诊断 API 配置问题

**技术细节：**
- 新增 `/api/validate-token` API 端点
- 通过向 API 发送测试请求验证 token 有效性
- 完善的错误处理：401 未授权、403 禁止访问、超时、连接错误等
- 支持中英文界面

### 版本 0.1.7

**改进：**
- 🔧 **接口请求优化**：https://github.com/qy527145/acemcp/pull/6
- 🔧 **兼容代理环境**：添加httpx[socks]扩展依赖，解决代理环境下出错的bug

### 版本 0.1.5

**新特性：**
- ✨ **日志系统优化**：将 FastAPI/Uvicorn 日志重定向到 loguru，防止污染 MCP stdio 协议
- ✨ **工具调试接口**：Web 管理界面新增工具列表和调试功能

**改进：**
- 🔧 **日志输出控制**：移除控制台日志输出，仅输出到文件，避免干扰 stdio 协议
- 🔧 **标准库日志拦截**：使用 `InterceptHandler` 拦截所有标准库日志
- 🔧 **Web API 增强**：新增 `/api/tools` 端点列出可用工具

**技术细节：**
- 实现了 `InterceptHandler` 类来拦截标准库 logging
- 配置 uvicorn 使用 `log_config=None` 禁用默认日志
- 所有日志统一输出到 `~/.acemcp/log/acemcp.log`

### 版本 0.1.4

**新特性：**
- ✨ **多编码支持**：自动检测和处理多种文件编码（UTF-8、GBK、GB2312、Latin-1）
- ✨ **.gitignore 集成**：自动从项目根目录加载并遵守 `.gitignore` 模式
- ✨ **改进的工具响应格式**：从基于列表的格式改为基于字典的格式，以提高客户端兼容性

**改进：**
- 🔧 **WebSocket 优化**：具有指数退避的智能重连（1秒 → 最大 30秒）
- 🔧 **减少日志噪音**：WebSocket 连接现在记录在 DEBUG 级别而不是 INFO
- 🔧 **连接稳定性**：最多 10 次重连尝试，防止无限循环
- 🔧 **更好的错误处理**：对无法用任何编码解码的文件进行优雅回退

**错误修复：**
- 🐛 修复了频繁的 WebSocket 连接/断开循环
- 🐛 修复了读取非 UTF-8 编码文件时的编码错误
- 🐛 改进了对带有目录匹配的 .gitignore 模式的处理