# 上传异常处理和二分法定位功能实现

## 概述

本文档记录了为 acemcp 项目实现的上传异常处理和二分法定位功能。该功能解决了批量上传时可能出现的异常问题，通过二分法精确定位异常的 blobs，并在搜索时自动排除这些异常 blobs。

## 功能特性

### 1. 异常 Blob 存储机制

- **存储文件**: `failed_blobs.json`
- **数据结构**:
  ```json
  {
    "project_path": [
      {
        "blob_hash": "sha256_hash",
        "path": "file_path",
        "error": "error_message",
        "timestamp": "2025-12-18T10:30:00"
      }
    ]
  }
  ```

### 2. 二分法递归定位

当批量上传失败时，系统会自动使用二分法来精确定位导致异常的具体 blob：

- **递归分割**: 将失败的批次分成两半进行测试
- **精确定位**: 递归执行直到找到具体的异常 blob
- **智能恢复**: 成功的 blobs 会被正常上传，只有异常的 blobs 被记录

### 3. 增强的异常日志

- **详细信息**: 记录每个 blob 的文件路径和 hash 值
- **批次信息**: 显示失败批次中包含的所有 blobs
- **Web 控制台**: 通过 API 接口在 web 控制台显示异常信息

### 4. 搜索时自动排除异常 Blobs

- **自动过滤**: 搜索时自动排除已知的异常 blobs
- **统计信息**: 显示排除的异常 blobs 数量
- **保证质量**: 确保搜索结果的可靠性

## 实现细节

### 核心方法

#### 1. `_add_failed_blob()`
```python
def _add_failed_blob(self, project_path: str, blob_hash: str, blob_path: str, error: str) -> None:
    """添加失败的 blob 到存储中"""
```

#### 2. `_binary_search_failed_blobs()`
```python
async def _binary_search_failed_blobs(self, blobs: list[dict], project_path: str, error_msg: str) -> list[dict]:
    """使用二分法识别批次中的具体失败 blobs"""
```

#### 3. `_get_failed_blob_hashes()`
```python
def _get_failed_blob_hashes(self, project_path: str) -> set[str]:
    """获取项目的失败 blob hash 集合"""
```

### 上传流程增强

1. **正常上传**: 尝试批量上传
2. **异常检测**: 捕获上传异常
3. **详细日志**: 记录失败批次的所有 blob 信息
4. **二分法定位**: 自动启动二分法定位异常 blob
5. **智能恢复**: 上传成功识别的 blobs
6. **异常记录**: 将异常 blobs 记录到存储中

### 搜索流程增强

1. **自动索引**: 执行增量索引
2. **加载 Blobs**: 从项目存储中加载所有 blob names
3. **排除异常**: 自动排除已知的异常 blobs
4. **执行搜索**: 使用过滤后的 blobs 进行搜索
5. **结果返回**: 返回可靠的搜索结果

## Web API 接口

### 获取异常 Blobs 信息

```http
GET /api/failed-blobs
```

**响应格式**:
```json
{
  "failed_blobs": {
    "project_path": {
      "count": 5,
      "blobs": [
        {
          "blob_hash": "abc123...",
          "path": "src/file.py",
          "error": "Upload timeout",
          "timestamp": "2025-12-18T10:30:00"
        }
      ]
    }
  },
  "total_failed": 5,
  "projects_with_failures": 1
}
```

## 日志示例

### 上传异常日志
```
ERROR Batch 2 failed after retries: Connection timeout
ERROR Failed batch contained 10 blobs:
ERROR   - src/main.py (hash: abc12345...)
ERROR   - src/utils.py (hash: def67890...)
INFO  Starting binary search to identify failed blobs in batch 2...
INFO  Binary search: testing first half with 5 blobs...
WARN  Binary search: first half failed: Connection timeout
INFO  Binary search: testing second half with 5 blobs...
INFO  Binary search: second half succeeded with 5 blobs
ERROR Binary search identified failed blob: src/main.py (hash: abc12345...)
INFO  Binary search recovered 5 blobs from failed batch 2
```

### 搜索排除日志
```
INFO  Excluded 3 failed blobs from search (total available: 100, searching: 97)
INFO  Performing search with 97 blobs (excluded 3 failed blobs)...
```

## 配置说明

无需额外配置，功能会自动启用。异常 blobs 信息存储在：
```
{index_storage_path}/failed_blobs.json
```

## 优势

1. **自动化**: 无需人工干预，自动处理上传异常
2. **精确定位**: 二分法确保快速定位到具体的异常 blob
3. **智能恢复**: 最大化成功上传的 blobs 数量
4. **搜索可靠**: 自动排除异常 blobs，保证搜索质量
5. **详细日志**: 提供完整的异常信息用于调试
6. **Web 监控**: 通过 web 控制台监控异常状态

## 性能影响

- **二分法复杂度**: O(log n)，快速定位异常 blob
- **存储开销**: 最小化，仅存储异常 blob 的元数据
- **搜索性能**: 通过排除异常 blobs 可能略微提升搜索性能

## 兼容性

- **向后兼容**: 不影响现有功能
- **渐进式**: 异常处理功能渐进式启用
- **可选性**: 如果没有异常 blobs，功能不会产生额外开销

## 总结

该实现提供了完整的上传异常处理解决方案，通过二分法精确定位异常 blobs，并在搜索时自动排除这些异常数据，确保系统的稳定性和搜索结果的可靠性。所有功能都是自动化的，无需用户干预，同时提供了详细的日志和 web 监控界面。