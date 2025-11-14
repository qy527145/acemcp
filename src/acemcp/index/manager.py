"""Index manager for codebase indexing."""

import asyncio
import fnmatch
import hashlib
import json
import os
from pathlib import Path

import httpx
import pathspec
from loguru import logger


def read_file_with_encoding(file_path: Path) -> str:
    """Read file content with automatic encoding detection.

    Tries multiple encodings in order: utf-8, gbk, gb2312, latin-1.

    Args:
        file_path: Path to the file to read

    Returns:
        File content as string

    Raises:
        Exception: If file cannot be read with any supported encoding
    """
    encodings = ["utf-8", "gbk", "gb2312", "latin-1"]

    for encoding in encodings:
        try:
            with file_path.open("r", encoding=encoding) as f:
                content = f.read()
            logger.debug(f"Successfully read {file_path} with encoding: {encoding}")
            return content
        except (UnicodeDecodeError, LookupError):
            continue

    # If all encodings fail, try with errors='ignore'
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        logger.warning(f"Read {file_path} with utf-8 and errors='ignore' (some characters may be lost)")
        return content
    except Exception as e:
        logger.error(f"Failed to read {file_path} with any encoding: {e}")
        raise


def calculate_blob_name(path: str, content: str) -> str:
    """Calculate blob_name (blob id) using SHA-256 hash.

    Args:
        path: File path, e.g. "asdasd.md"
        content: File content, e.g. "什么是快乐星球？这是一个网络梗"

    Returns:
        64-character hexadecimal string (SHA-256 hash value)
    """
    hasher = hashlib.sha256()
    hasher.update(path.encode("utf-8"))
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


class IndexManager:
    """Manages codebase indexing and retrieval."""

    def __init__(self, storage_path: Path, base_url: str, token: str, text_extensions: set[str], batch_size: int, max_lines_per_blob: int = 800, exclude_patterns: list[str] | None = None) -> None:
        """Initialize index manager.

        Args:
            storage_path: Path to store index data
            base_url: Base URL for API requests
            token: Authorization token
            text_extensions: Set of text file extensions to index
            batch_size: Number of files to upload per batch
            max_lines_per_blob: Maximum lines per blob before splitting (default: 800)
            exclude_patterns: List of patterns to exclude from indexing (default: None)
        """
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.text_extensions = text_extensions
        self.batch_size = batch_size
        self.max_lines_per_blob = max_lines_per_blob
        self.exclude_patterns = exclude_patterns or []
        self.projects_file = storage_path / "projects.json"
        self._client: httpx.AsyncClient | None = None
        logger.info(f"IndexManager initialized with storage path: {storage_path}, batch_size: {batch_size}, max_lines_per_blob: {max_lines_per_blob}, exclude_patterns: {len(self.exclude_patterns)} patterns")

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx AsyncClient instance.

        Returns:
            httpx.AsyncClient instance
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
            logger.debug("Created new httpx.AsyncClient")
        return self._client

    async def close(self) -> None:
        """Close httpx client and release resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("Closed httpx.AsyncClient")

    def _normalize_path(self, path: str) -> str:
        """Normalize path to use forward slashes.

        Args:
            path: Path string

        Returns:
            Normalized path string
        """
        return str(Path(path).resolve()).replace("\\", "/")

    def _load_gitignore(self, root_path: Path) -> pathspec.PathSpec | None:
        """Load and parse .gitignore file from project root.

        Args:
            root_path: Root path of the project

        Returns:
            PathSpec object if .gitignore exists, None otherwise
        """
        gitignore_path = root_path / ".gitignore"
        if not gitignore_path.exists():
            logger.debug(f"No .gitignore found at {gitignore_path}")
            return None

        try:
            with gitignore_path.open("r", encoding="utf-8") as f:
                patterns = f.read().splitlines()
            spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
            logger.info(f"Loaded .gitignore with {len(patterns)} patterns from {gitignore_path}")
            return spec
        except Exception as e:
            logger.warning(f"Failed to load .gitignore from {gitignore_path}: {e}")
            return None

    async def _retry_request(self, func, max_retries: int = 3, retry_delay: float = 1.0, *args, **kwargs):
        """Retry an async function with exponential backoff.

        Args:
            func: Async function to retry
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Initial delay between retries in seconds (default: 1.0)
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result from func if successful

        Raises:
            Exception: Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {max_retries} attempts: {e}")
            except Exception as e:
                # For non-retryable errors, raise immediately
                logger.error(f"Non-retryable error: {e}")
                raise

        raise last_exception

    def _should_exclude(self, path: Path, root_path: Path, gitignore_spec: pathspec.PathSpec | None = None) -> bool:
        """Check if a path should be excluded based on exclude patterns and .gitignore.

        Args:
            path: Path to check
            root_path: Root path of the project
            gitignore_spec: PathSpec object from .gitignore (optional)

        Returns:
            True if path should be excluded, False otherwise
        """
        try:
            relative_path = path.relative_to(root_path)
            path_str = str(relative_path)
            path_parts = relative_path.parts

            # Check .gitignore patterns first
            if gitignore_spec is not None:
                # Use forward slashes for gitignore matching
                path_str_forward = path_str.replace("\\", "/")
                # Add trailing slash for directories
                if path.is_dir():
                    path_str_forward += "/"
                if gitignore_spec.match_file(path_str_forward):
                    logger.debug(f"Excluded by .gitignore: {path_str_forward}")
                    return True

            # Check exclude_patterns
            for pattern in self.exclude_patterns:
                # Check if pattern matches any part of the path
                for part in path_parts:
                    if fnmatch.fnmatch(part, pattern):
                        return True

                # Check if pattern matches the full relative path
                if fnmatch.fnmatch(path_str, pattern):
                    return True

                # Check if pattern matches with forward slashes
                path_str_forward = path_str.replace("\\", "/")
                if fnmatch.fnmatch(path_str_forward, pattern):
                    return True

            return False
        except ValueError:
            # Path is not relative to root_path
            return False

    def _load_projects(self) -> dict[str, list[str]]:
        """Load projects data from storage.

        Returns:
            Dictionary mapping normalized project_root_path to blob_names
        """
        if not self.projects_file.exists():
            return {}
        try:
            with self.projects_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("Failed to load projects data")
            return {}

    def _save_projects(self, projects: dict[str, list[str]]) -> None:
        """Save projects data to storage.

        Args:
            projects: Dictionary mapping normalized project_root_path to blob_names
        """
        try:
            with self.projects_file.open("w", encoding="utf-8") as f:
                json.dump(projects, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to save projects data")
            raise

    def _split_file_content(self, path: str, content: str) -> list[dict[str, str]]:
        """Split file content into multiple blobs if it exceeds max_lines_per_blob.

        Args:
            path: File path
            content: File content

        Returns:
            List of blobs (one or more if split)
        """
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        # If file is within limit, return as single blob
        if total_lines <= self.max_lines_per_blob:
            return [{"path": path, "content": content}]

        # Split into multiple blobs
        blobs = []
        num_chunks = (total_lines + self.max_lines_per_blob - 1) // self.max_lines_per_blob

        for chunk_idx in range(num_chunks):
            start_line = chunk_idx * self.max_lines_per_blob
            end_line = min(start_line + self.max_lines_per_blob, total_lines)
            chunk_lines = lines[start_line:end_line]
            chunk_content = "".join(chunk_lines)

            # Add chunk index to path to make it unique
            chunk_path = f"{path}#chunk{chunk_idx + 1}of{num_chunks}"
            blobs.append({"path": chunk_path, "content": chunk_content})

        logger.info(f"Split file {path} ({total_lines} lines) into {num_chunks} chunks")
        return blobs

    def _collect_files(self, project_root_path: str) -> list[dict[str, str]]:
        """Collect all text files from project directory.

        Args:
            project_root_path: Root path of the project

        Returns:
            List of blobs with path and content (large files may be split into multiple blobs)
        """
        blobs = []
        excluded_count = 0
        root_path = Path(project_root_path)

        if not root_path.exists():
            msg = f"Project root path does not exist: {project_root_path}"
            raise FileNotFoundError(msg)

        # Load .gitignore if exists
        gitignore_spec = self._load_gitignore(root_path)

        for dirpath, dirnames, filenames in os.walk(root_path):
            current_dir = Path(dirpath)

            # Filter out excluded directories to prevent os.walk from descending into them
            dirnames[:] = [
                d for d in dirnames
                if not self._should_exclude(current_dir / d, root_path, gitignore_spec)
            ]

            for filename in filenames:
                file_path = current_dir / filename

                # Check if file should be excluded
                if self._should_exclude(file_path, root_path, gitignore_spec):
                    excluded_count += 1
                    logger.debug(f"Excluded file: {file_path.relative_to(root_path)}")
                    continue

                if file_path.suffix.lower() not in self.text_extensions:
                    continue

                try:
                    relative_path = file_path.relative_to(root_path)
                    content = read_file_with_encoding(file_path)

                    # Split file if necessary
                    file_blobs = self._split_file_content(str(relative_path), content)
                    blobs.extend(file_blobs)

                    logger.debug(f"Collected file: {relative_path} ({len(file_blobs)} blob(s))")
                except Exception:
                    logger.warning(f"Failed to read file: {file_path}")
                    continue

        logger.info(f"Collected {len(blobs)} blobs from {project_root_path} (excluded {excluded_count} files/directories)")
        return blobs

    async def index_project(self, project_root_path: str) -> dict[str, str]:
        """Index a code project with incremental indexing support.

        Args:
            project_root_path: Absolute path to the project root directory

        Returns:
            Result dictionary with status and message
        """
        normalized_path = self._normalize_path(project_root_path)
        logger.info(f"Indexing project from {normalized_path}")

        try:
            blobs = self._collect_files(project_root_path)

            if not blobs:
                return {"status": "error", "message": "No text files found in project"}

            # Load existing projects to check for incremental indexing
            projects = self._load_projects()
            existing_blob_names = set(projects.get(normalized_path, []))

            # Calculate hash for all collected blobs
            blob_hash_map = {}  # hash -> blob
            for blob in blobs:
                blob_hash = calculate_blob_name(blob["path"], blob["content"])
                blob_hash_map[blob_hash] = blob

            # Separate blobs into existing and new
            all_blob_hashes = set(blob_hash_map.keys())
            existing_hashes = all_blob_hashes & existing_blob_names  # Intersection
            new_hashes = all_blob_hashes - existing_blob_names  # Difference

            # Blobs that need to be uploaded
            blobs_to_upload = [blob_hash_map[h] for h in new_hashes]

            logger.info(
                f"Incremental indexing: total={len(blobs)}, "
                f"existing={len(existing_hashes)}, "
                f"new={len(new_hashes)}, "
                f"to_upload={len(blobs_to_upload)}"
            )

            # Upload only new blobs
            uploaded_blob_names = []
            failed_batches = []

            if blobs_to_upload:
                total_batches = (len(blobs_to_upload) + self.batch_size - 1) // self.batch_size
                logger.info(f"Uploading {len(blobs_to_upload)} new blobs in {total_batches} batches (batch_size={self.batch_size})")

                client = self._get_client()
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(blobs_to_upload))
                    batch_blobs = blobs_to_upload[start_idx:end_idx]

                    logger.info(f"Uploading batch {batch_idx + 1}/{total_batches} ({len(batch_blobs)} blobs)")

                    try:
                        async def upload_batch():
                            payload = {"blobs": batch_blobs}
                            response = await client.post(
                                f"{self.base_url}/batch-upload",
                                headers={"Authorization": f"Bearer {self.token}"},
                                json=payload,
                            )
                            response.raise_for_status()
                            return response.json()

                        # Retry up to 3 times with exponential backoff
                        result = await self._retry_request(upload_batch, max_retries=3, retry_delay=1.0)

                        batch_blob_names = result.get("blob_names", [])
                        if not batch_blob_names:
                            logger.warning(f"Batch {batch_idx + 1} returned no blob names")
                            failed_batches.append(batch_idx + 1)
                            continue

                        uploaded_blob_names.extend(batch_blob_names)
                        logger.info(f"Batch {batch_idx + 1} uploaded successfully, got {len(batch_blob_names)} blob names")

                    except Exception as e:
                        logger.error(f"Batch {batch_idx + 1} failed after retries: {e}. Continuing with next batch...")
                        failed_batches.append(batch_idx + 1)
                        continue

                if not uploaded_blob_names and blobs_to_upload:
                    if failed_batches:
                        return {"status": "error", "message": f"All batches failed. Failed batches: {failed_batches}"}
                    return {"status": "error", "message": "No blob names returned from API"}
            else:
                logger.info("No new blobs to upload, all blobs already exist in index")

            # Merge existing and newly uploaded blob names
            all_blob_names = list(existing_hashes) + uploaded_blob_names

            # Update projects.json with the merged blob names
            projects = self._load_projects()
            projects[normalized_path] = all_blob_names
            self._save_projects(projects)

            # Build result message
            if blobs_to_upload:
                total_batches = (len(blobs_to_upload) + self.batch_size - 1) // self.batch_size
                success_batches = total_batches - len(failed_batches)
                message = (
                    f"Project indexed with {len(all_blob_names)} total blobs "
                    f"(existing: {len(existing_hashes)}, new: {len(uploaded_blob_names)}, "
                    f"batches: {success_batches}/{total_batches} successful)"
                )
            else:
                message = f"Project indexed with {len(all_blob_names)} total blobs (all existing, no upload needed)"

            if failed_batches:
                message += f". Failed batches: {failed_batches}"
                logger.warning(f"Project {normalized_path} indexed with some failures: {message}")
            else:
                logger.info(f"Project {normalized_path} indexed successfully: {message}")

            return {
                "status": "success" if not failed_batches else "partial_success",
                "message": message,
                "project_path": normalized_path,
                "failed_batches": failed_batches,
                "stats": {
                    "total_blobs": len(all_blob_names),
                    "existing_blobs": len(existing_hashes),
                    "new_blobs": len(uploaded_blob_names),
                    "skipped_blobs": len(existing_hashes),
                },
            }

        except Exception as e:
            logger.exception(f"Failed to index project {normalized_path}")
            return {"status": "error", "message": str(e)}

    async def search_context(self, project_root_path: str, query: str) -> str:
        """Search for code context based on query with automatic incremental indexing.

        This method automatically performs incremental indexing before searching,
        ensuring the search is always performed on the latest codebase.

        Args:
            project_root_path: Absolute path to the project root directory
            query: Search query string

        Returns:
            Formatted retrieval result
        """
        normalized_path = self._normalize_path(project_root_path)
        logger.info(f"Searching context in project {normalized_path} with query: {query}")

        try:
            # Step 1: Automatically perform incremental indexing
            logger.info(f"Auto-indexing project {normalized_path} before search...")
            index_result = await self.index_project(project_root_path)

            if index_result["status"] == "error":
                return f"Error: Failed to index project before search. {index_result['message']}"

            # Log indexing stats
            if "stats" in index_result:
                stats = index_result["stats"]
                logger.info(
                    f"Auto-indexing completed: total={stats['total_blobs']}, "
                    f"existing={stats['existing_blobs']}, new={stats['new_blobs']}"
                )

            # Step 2: Load indexed blob names
            projects = self._load_projects()
            blob_names = projects.get(normalized_path, [])

            if not blob_names:
                return f"Error: No blobs found for project {normalized_path} after indexing."

            # Step 3: Perform search
            logger.info(f"Performing search with {len(blob_names)} blobs...")
            payload = {
                "information_request": query,
                "blobs": {
                    "checkpoint_id": None,
                    "added_blobs": blob_names,
                    "deleted_blobs": [],
                },
                "dialog": [],
                "max_output_length": 0,
                "disable_codebase_retrieval": False,
                "enable_commit_retrieval": False,
            }

            client = self._get_client()
            async def search_request():
                response = await client.post(
                    f"{self.base_url}/agents/codebase-retrieval",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()

            # Retry up to 3 times with exponential backoff
            try:
                result = await self._retry_request(search_request, max_retries=3, retry_delay=2.0)
            except Exception as e:
                logger.error(f"Search request failed after retries: {e}")
                return f"Error: Search request failed after 3 retries. {e!s}"

            formatted_retrieval = result.get("formatted_retrieval", "")

            if not formatted_retrieval:
                logger.warning(f"Search returned empty result for project {normalized_path}")
                return "No relevant code context found for your query."

            logger.info(f"Search completed for project {normalized_path}")
            return formatted_retrieval

        except Exception as e:
            logger.exception(f"Failed to search context in project {normalized_path}")
            return f"Error: {e!s}"

