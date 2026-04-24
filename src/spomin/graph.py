"""Graph memory layer wrapping graphiti-core."""

import json
from datetime import datetime, timezone
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.embedder.openai import OpenAIEmbedderConfig
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_CROSS_ENCODER
from pydantic import BaseModel

from spomin.config import Settings


def _patch_openai_client() -> None:
    """Patch graphiti-core to use chat/completions + JSON schema instead of responses.parse.

    llama-server supports /v1/chat/completions with response_format json_schema,
    but NOT the newer /v1/responses API that graphiti-core 0.28.2 expects.
    """

    async def patched_create_structured_completion(
        self,
        model: str,
        messages: list,
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ) -> Any:
        schema = response_model.model_json_schema()
        # Inject schema into the system prompt since llama-server
        # only supports json_object (not json_schema)
        schema_prompt = (
            "Respond with valid JSON matching this schema:\n"
            + json.dumps(schema, indent=2)
        )
        # Prepend schema instruction to the first (system) message
        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = (
                messages[0].get("content", "") + "\n\n" + schema_prompt
            )
        else:
            messages.insert(0, {"role": "system", "content": schema_prompt})

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature or 0,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response

    OpenAIClient._create_structured_completion = patched_create_structured_completion

    def patched_handle_structured_response(self, response: Any) -> tuple:
        content = response.choices[0].message.content or "{}"
        # Strip markdown code fences
        content = content.strip()
        if content.startswith("```json"):
            content = content.removeprefix("```json").strip()
        if content.startswith("```"):
            content = content.removeprefix("```").strip()
        if content.endswith("```"):
            content = content.removesuffix("```").strip()
        input_tokens = (
            getattr(response.usage, "prompt_tokens", 0)
            if hasattr(response, "usage") and response.usage
            else 0
        )
        output_tokens = (
            getattr(response.usage, "completion_tokens", 0)
            if hasattr(response, "usage") and response.usage
            else 0
        )

        parsed = json.loads(content)
        return parsed, input_tokens, output_tokens

    from graphiti_core.llm_client.openai_base_client import BaseOpenAIClient

    BaseOpenAIClient._handle_structured_response = patched_handle_structured_response


class GraphMemory:
    """Thin wrapper around graphiti-core Graphiti client."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # Patch graphiti-core to use chat/completions instead of responses API
        _patch_openai_client()

        # LLM client for entity/edge extraction (separate server)
        llm_config = LLMConfig(
            api_key=settings.extraction_llm_api_key,
            model=settings.extraction_llm_model,
            base_url=settings.extraction_llm_base_url,
        )
        llm_client = OpenAIClient(config=llm_config)

        # Embedder for semantic search
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=settings.embedder_api_key,
                embedding_model=settings.embedder_model,
                base_url=settings.embedder_base_url,
            )
        )

        # Reranker for result ranking
        cross_encoder = OpenAIRerankerClient(config=llm_config)

        # Graphiti instance
        self.graphiti = Graphiti(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

    async def initialize(self) -> None:
        """Build indices and constraints in Neo4j. Call once at startup."""
        await self.graphiti.build_indices_and_constraints(delete_existing=False)

    async def close(self) -> None:
        """Close the Neo4j connection."""
        await self.graphiti.close()

    async def add_episode(
        self, text: str, source: str = "conversation"
    ) -> dict:
        """Add a memory episode to the knowledge graph.

        Extracts entities and relationships from *text* using the LLM, then
        stores them as nodes and edges in Neo4j.

        Returns a dict with the episode uuid and counts of created nodes/edges.
        """
        results = await self.graphiti.add_episode(
            name=source,
            episode_body=text,
            source_description=source,
            reference_time=datetime.now(timezone.utc),
            source=EpisodeType.message,
            group_id=self.settings.user_id,
        )

        return {
            "episode_uuid": results.episode.uuid,
            "node_count": len(results.nodes),
            "edge_count": len(results.edges),
        }

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search the knowledge graph semantically.

        Returns a list of dicts, each describing a fact (edge) with source
        and target entity info.
        """
        search_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(
            update={"limit": int(limit)}
        )
        results = await self.graphiti.search_(
            query=query,
            config=search_config,
            group_ids=[self.settings.user_id],
        )

        # Build a lookup from node UUID to name for resolving edge endpoints
        node_names: dict[str, str] = {}
        for node in results.nodes:
            node_names[node.uuid] = node.name

        facts = []
        for edge in results.edges[:limit]:
            facts.append({
                "fact": edge.fact,
                "relation": edge.name,
                "source_uuid": edge.source_node_uuid,
                "target_uuid": edge.target_node_uuid,
                "source": node_names.get(edge.source_node_uuid, edge.source_node_uuid),
                "target": node_names.get(edge.target_node_uuid, edge.target_node_uuid),
            })

        return facts

    async def remove_episode(self, episode_uuid: str) -> None:
        """Delete an episode and its orphaned nodes/edges."""
        await self.graphiti.remove_episode(episode_uuid)

    async def list_episodes(self, limit: int = 20) -> list[dict]:
        """Return the most recent episodes."""
        episodes = await self.graphiti.retrieve_episodes(
            reference_time=datetime.now(timezone.utc),
            last_n=limit,
            group_ids=[self.settings.user_id],
        )

        return [
            {
                "uuid": ep.uuid,
                "name": ep.name,
                "content": (ep.content or "")[:200],
                "created_at": str(ep.created_at),
            }
            for ep in episodes
        ]
