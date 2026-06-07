"""SQLite-backed episodic memory with hybrid keyword and vector retrieval."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import sqlite3
import struct
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from spomin.config import Settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_FTS_TERM_RE = re.compile(r"\w+", re.UNICODE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pack_embedding(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_embedding(value: bytes, dimensions: int) -> tuple[float, ...]:
    return struct.unpack(f"<{dimensions}f", value)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _chunk_text(text: str, target_tokens: int, max_tokens: int) -> list[str]:
    """Split text into paragraph-aware chunks using a cheap token estimate."""
    text = text.strip()
    if not text:
        return []
    if len(_TOKEN_RE.findall(text)) <= max_tokens:
        return [text]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(_TOKEN_RE.findall(paragraph)) <= max_tokens:
            pieces.append(paragraph)
            continue
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|\n+", paragraph)
            if part.strip()
        ]
        for sentence in sentences:
            tokens = _TOKEN_RE.findall(sentence)
            if len(tokens) <= max_tokens:
                pieces.append(sentence)
                continue
            words = sentence.split()
            # Words are a conservative approximation when one sentence is huge.
            for start in range(0, len(words), max_tokens):
                pieces.append(" ".join(words[start : start + max_tokens]))

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for piece in pieces:
        piece_tokens = len(_TOKEN_RE.findall(piece))
        if current and current_tokens + piece_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(piece)
        current_tokens += piece_tokens
        if current_tokens >= target_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks


class EpisodicMemory:
    """Persistent raw memory archive with asynchronous chunk embedding."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database_path = Path(settings.database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_client = AsyncOpenAI(
            api_key=settings.embedder_api_key,
            base_url=settings.embedder_base_url,
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._work_event = asyncio.Event()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except Exception:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")
        finally:
            connection.close()

    async def initialize(self) -> None:
        self._initialize_sync()
        self._stop_event.clear()
        self._worker_task = asyncio.create_task(
            self._embedding_worker(), name="spomin-embedding-worker"
        )
        self._work_event.set()

    def _initialize_sync(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    app TEXT,
                    model TEXT,
                    role TEXT,
                    project TEXT,
                    tier TEXT NOT NULL CHECK (tier IN ('archive', 'core')),
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    app TEXT,
                    model TEXT,
                    role TEXT,
                    project TEXT,
                    tier TEXT NOT NULL CHECK (tier IN ('archive', 'core')),
                    created_at TEXT NOT NULL,
                    embedding BLOB,
                    embedding_dimensions INTEGER,
                    embedding_revision INTEGER NOT NULL DEFAULT 0,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS chunk_messages (
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    text_fragment TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, message_id, position)
                );

                CREATE INDEX IF NOT EXISTS chunk_messages_message_idx
                    ON chunk_messages(message_id);
                CREATE INDEX IF NOT EXISTS chunks_created_at_idx
                    ON chunks(created_at DESC);
                CREATE INDEX IF NOT EXISTS chunks_conversation_idx
                    ON chunks(conversation_id);
                CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks(source);
                CREATE INDEX IF NOT EXISTS chunks_project_idx ON chunks(project);
                CREATE INDEX IF NOT EXISTS chunks_tier_idx ON chunks(tier);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    tokenize = 'unicode61'
                );

                CREATE TRIGGER IF NOT EXISTS chunks_fts_insert
                AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts (chunk_id, text)
                    VALUES (new.id, new.text);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_fts_update
                AFTER UPDATE OF text ON chunks BEGIN
                    DELETE FROM chunks_fts WHERE chunk_id = old.id;
                    INSERT INTO chunks_fts (chunk_id, text)
                    VALUES (new.id, new.text);
                END;

                CREATE TRIGGER IF NOT EXISTS chunks_fts_delete
                AFTER DELETE ON chunks BEGIN
                    DELETE FROM chunks_fts WHERE chunk_id = old.id;
                END;
                """
            )

    async def close(self) -> None:
        self._stop_event.set()
        self._work_event.set()
        if self._worker_task is not None:
            await self._worker_task
            self._worker_task = None
        await self.embedding_client.close()

    async def add_memory(
        self,
        text: str,
        source: str = "conversation",
        conversation_id: str | None = None,
        *,
        app: str | None = None,
        model: str | None = None,
        role: str | None = None,
        project: str | None = None,
        tier: str = "archive",
    ) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("Memory text cannot be empty.")
        if tier not in {"archive", "core"}:
            raise ValueError("tier must be 'archive' or 'core'.")
        if not source.strip():
            raise ValueError("source cannot be empty.")

        message_id = str(uuid.uuid4())
        created_at = _utc_now()
        chunk_ids = self._store_memory_sync(
            message_id,
            text,
            source.strip(),
            conversation_id,
            app,
            model,
            role,
            project,
            tier,
            created_at,
        )
        self._work_event.set()
        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunk_ids),
            "tier": tier,
            "created_at": created_at,
        }

    def _store_memory_sync(
        self,
        message_id: str,
        text: str,
        source: str,
        conversation_id: str | None,
        app: str | None,
        model: str | None,
        role: str | None,
        project: str | None,
        tier: str,
        created_at: str,
    ) -> list[str]:
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, text, source, app, model, role,
                    project, tier, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    text,
                    source,
                    app,
                    model,
                    role,
                    project,
                    tier,
                    created_at,
                ),
            )
            chunks = _chunk_text(
                text,
                target_tokens=self.settings.chunk_target_tokens,
                max_tokens=self.settings.chunk_max_tokens,
            )
            chunk_ids: list[str] = []
            if conversation_id and len(chunks) == 1:
                existing = connection.execute(
                    """
                    SELECT id, text, role
                    FROM chunks
                    WHERE conversation_id = ?
                      AND source = ?
                      AND app IS ?
                      AND model IS ?
                      AND project IS ?
                      AND tier = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (conversation_id, source, app, model, project, tier),
                ).fetchone()
                fragment = self._message_fragment(text, role)
                if existing is not None:
                    combined = f"{existing['text']}\n\n{fragment}"
                    existing_tokens = len(_TOKEN_RE.findall(existing["text"]))
                    combined_tokens = len(_TOKEN_RE.findall(combined))
                    if (
                        existing_tokens < self.settings.chunk_target_tokens
                        and combined_tokens <= self.settings.chunk_max_tokens
                    ):
                        position = connection.execute(
                            """
                            SELECT COALESCE(MAX(position), -1) + 1
                            FROM chunk_messages WHERE chunk_id = ?
                            """,
                            (existing["id"],),
                        ).fetchone()[0]
                        connection.execute(
                            """
                            INSERT INTO chunk_messages (
                                chunk_id, message_id, position, text_fragment
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (existing["id"], message_id, position, fragment),
                        )
                        combined_role = (
                            role if existing["role"] in (None, role) else None
                        )
                        connection.execute(
                            """
                            UPDATE chunks
                            SET text = ?, role = ?, created_at = ?,
                                embedding = NULL,
                                embedding_dimensions = NULL,
                                embedding_revision = embedding_revision + 1
                            WHERE id = ?
                            """,
                            (combined, combined_role, created_at, existing["id"]),
                        )
                        return [existing["id"]]

            for position, chunk_text in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                fragment = (
                    self._message_fragment(chunk_text, role)
                    if conversation_id
                    else chunk_text
                )
                connection.execute(
                    """
                    INSERT INTO chunks (
                        id, conversation_id, text, source, app,
                        model, role, project, tier, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        conversation_id,
                        fragment,
                        source,
                        app,
                        model,
                        role,
                        project,
                        tier,
                        created_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO chunk_messages (
                        chunk_id, message_id, position, text_fragment
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (chunk_id, message_id, position, fragment),
                )
            return chunk_ids

    @staticmethod
    def _message_fragment(text: str, role: str | None) -> str:
        return f"{role}: {text}" if role else text

    async def search(
        self,
        query: str,
        limit: int = 5,
        *,
        source: str | None = None,
        app: str | None = None,
        conversation_id: str | None = None,
        project: str | None = None,
        tier: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty.")
        limit = max(1, min(int(limit), self.settings.max_search_limit))
        filters = {
            "source": source,
            "app": app,
            "conversation_id": conversation_id,
            "project": project,
            "tier": tier,
            "since": since,
            "until": until,
        }
        if tier is not None and tier not in {"archive", "core"}:
            raise ValueError("tier must be 'archive' or 'core'.")

        candidate_limit = max(limit * 10, 50)
        keyword_rows = self._keyword_candidates_sync(query, candidate_limit, filters)

        query_embedding: list[float] | None = None
        try:
            response = await self.embedding_client.embeddings.create(
                model=self.settings.embedder_model,
                input=[query],
            )
            query_embedding = list(response.data[0].embedding)
        except Exception as exc:
            logger.warning("Vector query unavailable; using keyword search only: %s", exc)

        vector_rows: list[dict[str, Any]] = []
        if query_embedding is not None:
            embedded_rows = self._embedded_candidates_sync(filters)
            for row in embedded_rows:
                embedding = _unpack_embedding(
                    row["embedding"], row["embedding_dimensions"]
                )
                result = dict(row)
                result["vector_score"] = _cosine_similarity(
                    query_embedding, embedding
                )
                vector_rows.append(result)
            vector_rows.sort(key=lambda item: item["vector_score"], reverse=True)
            vector_rows = vector_rows[:candidate_limit]

        results = self._merge_rankings(keyword_rows, vector_rows, limit)
        if results:
            self._record_access_sync([result["id"] for result in results])
        return results

    def _filter_sql(
        self, filters: dict[str, str | None], alias: str = "c"
    ) -> tuple[str, list[str]]:
        clauses: list[str] = []
        values: list[str] = []
        for key in ("source", "app", "conversation_id", "tier"):
            if filters[key] is not None:
                clauses.append(f"{alias}.{key} = ?")
                values.append(str(filters[key]))
        if filters["project"] is not None:
            clauses.append(
                f"({alias}.project = ? OR "
                f"({alias}.project IS NULL AND {alias}.tier = 'core'))"
            )
            values.append(str(filters["project"]))
        if filters["since"] is not None:
            clauses.append(f"{alias}.created_at >= ?")
            values.append(str(filters["since"]))
        if filters["until"] is not None:
            clauses.append(f"{alias}.created_at <= ?")
            values.append(str(filters["until"]))
        return (" AND ".join(clauses), values)

    def _keyword_candidates_sync(
        self,
        query: str,
        limit: int,
        filters: dict[str, str | None],
    ) -> list[dict[str, Any]]:
        terms = _FTS_TERM_RE.findall(query)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        filter_sql, filter_values = self._filter_sql(filters)
        where = f" AND {filter_sql}" if filter_sql else ""
        sql = f"""
            SELECT c.*, bm25(chunks_fts) AS keyword_score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ? {where}
            ORDER BY keyword_score
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(
                sql, [fts_query, *filter_values, limit]
            ).fetchall()
        return [dict(row) for row in rows]

    def _embedded_candidates_sync(
        self, filters: dict[str, str | None]
    ) -> list[dict[str, Any]]:
        filter_sql, values = self._filter_sql(filters)
        where = f" AND {filter_sql}" if filter_sql else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT c.*
                FROM chunks c
                WHERE c.embedding IS NOT NULL {where}
                """,
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def _merge_rankings(
        self,
        keyword_rows: list[dict[str, Any]],
        vector_rows: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        combined: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}
        for rank, row in enumerate(keyword_rows, start=1):
            combined[row["id"]] = row
            scores[row["id"]] = scores.get(row["id"], 0.0) + (
                self.settings.keyword_weight / (60 + rank)
            )
        for rank, row in enumerate(vector_rows, start=1):
            combined.setdefault(row["id"], row)
            scores[row["id"]] = scores.get(row["id"], 0.0) + (
                self.settings.vector_weight / (60 + rank)
            )

        now = datetime.now(timezone.utc)
        for chunk_id, row in combined.items():
            created = datetime.fromisoformat(row["created_at"])
            age_days = max((now - created).total_seconds() / 86400, 0)
            recency = math.exp(-age_days / self.settings.recency_half_life_days)
            access = math.log1p(row["access_count"])
            scores[chunk_id] += self.settings.recency_weight * recency / 60
            scores[chunk_id] += self.settings.access_weight * access / 60
            if row["tier"] == "core":
                scores[chunk_id] += self.settings.core_memory_boost / 60

        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
        message_ids = self._message_ids_for_chunks_sync(ranked_ids)
        return [
            {
                "id": chunk_id,
                "message_id": message_ids[chunk_id][0],
                "message_ids": message_ids[chunk_id],
                "text": combined[chunk_id]["text"],
                "source": combined[chunk_id]["source"],
                "conversation_id": combined[chunk_id]["conversation_id"],
                "app": combined[chunk_id]["app"],
                "model": combined[chunk_id]["model"],
                "role": combined[chunk_id]["role"],
                "project": combined[chunk_id]["project"],
                "tier": combined[chunk_id]["tier"],
                "created_at": combined[chunk_id]["created_at"],
                "score": round(scores[chunk_id], 8),
            }
            for chunk_id in ranked_ids
        ]

    def _message_ids_for_chunks_sync(
        self, chunk_ids: list[str]
    ) -> dict[str, list[str]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT chunk_id, message_id
                FROM chunk_messages
                WHERE chunk_id IN ({placeholders})
                ORDER BY chunk_id, position
                """,
                chunk_ids,
            ).fetchall()
        result = {chunk_id: [] for chunk_id in chunk_ids}
        for row in rows:
            if row["message_id"] not in result[row["chunk_id"]]:
                result[row["chunk_id"]].append(row["message_id"])
        return result

    def _record_access_sync(self, chunk_ids: list[str]) -> None:
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE chunks
                SET access_count = access_count + 1, last_accessed_at = ?
                WHERE id IN ({placeholders})
                """,
                [_utc_now(), *chunk_ids],
            )

    async def recent(
        self,
        limit: int = 10,
        *,
        source: str | None = None,
        app: str | None = None,
        conversation_id: str | None = None,
        project: str | None = None,
        tier: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), self.settings.max_search_limit))
        filters = {
            "source": source,
            "app": app,
            "conversation_id": conversation_id,
            "project": project,
            "tier": tier,
            "since": None,
            "until": None,
        }
        return self._recent_sync(limit, filters)

    def _recent_sync(
        self, limit: int, filters: dict[str, str | None]
    ) -> list[dict[str, Any]]:
        filter_sql, values = self._filter_sql(filters, alias="m")
        where = f"WHERE {filter_sql}" if filter_sql else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT m.id, m.conversation_id, m.text, m.source, m.app,
                       m.model, m.role, m.project, m.tier, m.created_at
                FROM messages m
                {where}
                ORDER BY m.created_at DESC
                LIMIT ?
                """,
                [*values, limit],
            ).fetchall()
        return [dict(row) for row in rows]

    async def forget(self, memory_id: str) -> bool:
        return self._forget_sync(memory_id)

    def _forget_sync(self, memory_id: str) -> bool:
        with self._transaction() as connection:
            chunk_ids = [
                row["chunk_id"]
                for row in connection.execute(
                    "SELECT DISTINCT chunk_id FROM chunk_messages WHERE message_id = ?",
                    (memory_id,),
                )
            ]
            deleted = connection.execute(
                "DELETE FROM messages WHERE id = ?", (memory_id,)
            ).rowcount
            if deleted:
                for chunk_id in chunk_ids:
                    fragments = [
                        row["text_fragment"]
                        for row in connection.execute(
                            """
                            SELECT text_fragment FROM chunk_messages
                            WHERE chunk_id = ? ORDER BY position
                            """,
                            (chunk_id,),
                        )
                    ]
                    if fragments:
                        connection.execute(
                            """
                            UPDATE chunks
                            SET text = ?, embedding = NULL,
                                embedding_dimensions = NULL,
                                embedding_revision = embedding_revision + 1
                            WHERE id = ?
                            """,
                            ("\n\n".join(fragments), chunk_id),
                        )
                    else:
                        connection.execute(
                            "DELETE FROM chunks WHERE id = ?", (chunk_id,)
                        )
                self._work_event.set()
                return True

            return bool(
                connection.execute(
                    "DELETE FROM chunks WHERE id = ?", (memory_id,)
                ).rowcount
            )

    async def _embedding_worker(self) -> None:
        while not self._stop_event.is_set():
            self._work_event.clear()
            pending = self._pending_embeddings(self.settings.embedding_batch_size)
            if not pending:
                try:
                    await asyncio.wait_for(
                        self._work_event.wait(),
                        timeout=self.settings.embedding_poll_interval,
                    )
                except TimeoutError:
                    pass
                continue

            try:
                response = await self.embedding_client.embeddings.create(
                    model=self.settings.embedder_model,
                    input=[row["text"] for row in pending],
                )
                embeddings = [
                    item.embedding
                    for item in sorted(response.data, key=lambda item: item.index)
                ]
                if len(embeddings) != len(pending):
                    raise RuntimeError("Embedding server returned an unexpected batch size.")
                self._store_embeddings(pending, embeddings)
            except Exception as exc:
                logger.warning("Embedding batch failed: %s", exc)
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.settings.embedding_retry_interval,
                    )
                except TimeoutError:
                    pass

    def _pending_embeddings(self, limit: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, text, embedding_revision FROM chunks
                WHERE embedding IS NULL
                ORDER BY created_at
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _store_embeddings(
        self,
        rows: list[dict[str, Any]],
        embeddings: list[Sequence[float]],
    ) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                UPDATE chunks
                SET embedding = ?, embedding_dimensions = ?
                WHERE id = ? AND embedding_revision = ?
                """,
                [
                    (
                        _pack_embedding(embedding),
                        len(embedding),
                        row["id"],
                        row["embedding_revision"],
                    )
                    for row, embedding in zip(rows, embeddings, strict=True)
                ],
            )
