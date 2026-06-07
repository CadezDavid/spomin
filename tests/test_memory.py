import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from spomin.config import Settings
from spomin.memory import EpisodicMemory, _chunk_text


class FakeEmbeddings:
    async def create(self, *, model: str, input: list[str]):
        data = [
            SimpleNamespace(index=index, embedding=self._embed(text))
            for index, text in enumerate(input)
        ]
        return SimpleNamespace(data=data)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("python")),
            float(lowered.count("sqlite")),
            float(lowered.count("garden")),
            0.1,
        ]


class FakeEmbeddingClient:
    def __init__(self, embeddings=None) -> None:
        self.embeddings = embeddings or FakeEmbeddings()

    async def close(self) -> None:
        pass


class FlakyEmbeddings(FakeEmbeddings):
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, *, model: str, input: list[str]):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("embedding server unavailable")
        return await super().create(model=model, input=input)


def make_memory(
    database_path: Path, embeddings: FakeEmbeddings | None = None
) -> EpisodicMemory:
    settings = Settings(
        database_path=database_path,
        embedder_base_url="http://unused.invalid/v1",
        embedding_poll_interval=0.01,
        embedding_retry_interval=0.01,
    )
    memory = EpisodicMemory(settings)
    memory.embedding_client = FakeEmbeddingClient(embeddings)
    return memory


async def wait_until_embedded(database_path: Path, expected: int) -> None:
    for _ in range(100):
        with sqlite3.connect(database_path) as connection:
            ready = connection.execute(
                "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        if ready == expected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("embedding worker did not finish")


def test_chunk_text_respects_maximum() -> None:
    text = " ".join(f"word{index}." for index in range(120))
    chunks = _chunk_text(text, target_tokens=20, max_tokens=30)
    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 30 for chunk in chunks)


def test_store_search_recent_filter_and_forget(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "spomin.db"
        memory = make_memory(database_path)
        await memory.initialize()
        try:
            first = await memory.add_memory(
                "The backend uses Python and SQLite for durable storage.",
                source="conversation",
                conversation_id="chat-1",
                project="spomin",
                role="user",
            )
            second = await memory.add_memory(
                "The garden needs water every Friday.",
                source="note",
                project="home",
                tier="core",
            )
            third = await memory.add_memory(
                "The API is deployed in a container.",
                source="conversation",
                conversation_id="chat-1",
                project="spomin",
                role="assistant",
            )
            assert first["chunk_ids"] == third["chunk_ids"]
            await wait_until_embedded(database_path, expected=2)

            results = await memory.search("Python database", limit=2)
            assert results[0]["message_id"] == first["id"]
            assert results[0]["message_ids"] == [first["id"], third["id"]]
            assert results[0]["project"] == "spomin"

            filtered = await memory.search(
                "Friday", limit=5, project="home", tier="core"
            )
            assert [item["message_id"] for item in filtered] == [second["id"]]

            recent = await memory.recent(limit=5, conversation_id="chat-1")
            assert [item["id"] for item in recent] == [third["id"], first["id"]]

            assert await memory.forget(first["id"]) is True
            assert await memory.forget(first["id"]) is False
            remaining = await memory.search("container", limit=5, project="spomin")
            assert remaining[0]["message_ids"] == [third["id"]]
            assert "Python" not in remaining[0]["text"]
            assert await memory.forget(third["id"]) is True
            assert (
                await memory.search("Python SQLite", limit=5, project="spomin")
                == []
            )
        finally:
            await memory.close()

    asyncio.run(scenario())


def test_raw_write_is_searchable_before_embedding(tmp_path: Path) -> None:
    async def scenario() -> None:
        memory = make_memory(tmp_path / "spomin.db")
        await memory.initialize()
        memory._stop_event.set()
        memory._work_event.set()
        if memory._worker_task is not None:
            await memory._worker_task
            memory._worker_task = None
        try:
            stored = await memory.add_memory("Unique keyword heliotrope", source="note")
            results = await memory.search("heliotrope", limit=1)
            assert results[0]["message_id"] == stored["id"]
        finally:
            await memory.close()

    asyncio.run(scenario())


def test_failed_embeddings_retry_without_queue_state(tmp_path: Path) -> None:
    async def scenario() -> None:
        embeddings = FlakyEmbeddings()
        database_path = tmp_path / "spomin.db"
        memory = make_memory(database_path, embeddings)
        await memory.initialize()
        try:
            await memory.add_memory("Python retry behavior")
            await wait_until_embedded(database_path, expected=1)
            assert embeddings.calls >= 2
        finally:
            await memory.close()

    asyncio.run(scenario())


def test_previous_embedding_queue_schema_remains_compatible(tmp_path: Path) -> None:
    database_path = tmp_path / "spomin.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                app TEXT,
                model TEXT,
                role TEXT,
                project TEXT,
                tier TEXT NOT NULL,
                created_at TEXT NOT NULL,
                embedding BLOB,
                embedding_dimensions INTEGER,
                embedding_revision INTEGER NOT NULL DEFAULT 0,
                embedding_status TEXT NOT NULL DEFAULT 'pending',
                embedding_error TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed_at TEXT
            )
            """
        )

    async def scenario() -> None:
        memory = make_memory(database_path)
        await memory.initialize()
        try:
            await memory.add_memory("Compatible SQLite archive")
            await wait_until_embedded(database_path, expected=1)
            assert await memory.search("SQLite archive", limit=1)
        finally:
            await memory.close()

    asyncio.run(scenario())


def test_project_scope_includes_global_core_only(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "spomin.db"
        memory = make_memory(database_path)
        await memory.initialize()
        try:
            project_memory = await memory.add_memory(
                "Python convention for alpha",
                project="alpha",
            )
            global_core = await memory.add_memory(
                "Global Python preference",
                tier="core",
            )
            global_archive = await memory.add_memory(
                "Old global Python discussion",
            )
            other_project = await memory.add_memory(
                "Python convention for beta",
                project="beta",
            )
            await wait_until_embedded(database_path, expected=4)

            results = await memory.search(
                "Python convention preference",
                limit=10,
                project="alpha",
            )
            result_ids = {result["message_id"] for result in results}
            assert project_memory["id"] in result_ids
            assert global_core["id"] in result_ids
            assert global_archive["id"] not in result_ids
            assert other_project["id"] not in result_ids

            archive_results = await memory.search(
                "Python",
                limit=10,
                project="alpha",
                tier="archive",
            )
            assert [result["message_id"] for result in archive_results] == [
                project_memory["id"]
            ]

            recent = await memory.recent(limit=10, project="alpha")
            recent_ids = {result["id"] for result in recent}
            assert recent_ids == {project_memory["id"], global_core["id"]}
        finally:
            await memory.close()

    asyncio.run(scenario())
