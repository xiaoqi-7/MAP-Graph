from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from src.memory_baselines.base import BaseMemoryBackend, MemoryRecord
from src.utils.model_clients import EmbeddingClient, load_model_config


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)


class PerAgentIsolatedMemoryBackend(BaseMemoryBackend):
    """Per-agent isolated vector memory baseline.

    Each agent writes to and retrieves from its own Chroma collection. Other
    agents can influence the current agent only through ordinary conversation
    messages supplied by the workflow, not through direct memory retrieval.
    """

    method_name = "B2"

    def __init__(self, top_k: int | None = None):
        config = load_model_config()
        self.top_k = top_k or int(config.get("retrieval", {}).get("top_k", 5))
        self.embedder = EmbeddingClient()
        self._client = None
        self._collections: dict[str, Any] = {}
        self._records_by_agent: dict[str, dict[str, MemoryRecord]] = {}
        self._all_records: list[MemoryRecord] = []
        self._observed_source_keys: set[tuple[str, str]] = set()
        self._observed_tool_output_keys: set[tuple[str, str]] = set()
        self._seeded_task_id: str | None = None
        self._counter = 0

    def reset_task(self, task_id: str) -> None:
        super().reset_task(task_id)
        self._collections = {}
        self._records_by_agent = {}
        self._all_records = []
        self._observed_source_keys = set()
        self._observed_tool_output_keys = set()
        self._seeded_task_id = None
        self._counter = 0

        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "B2 isolated_vector requires Chroma. Install it with:\n"
                "  conda activate mapgraph\n"
                "  pip install chromadb\n"
            ) from exc

        self._client = chromadb.EphemeralClient()

    def observe_sources(self, agent_id: str, sources: list[dict], task: dict, metadata: dict) -> list[MemoryRecord]:
        self._ensure_task_seeded(task)
        written = []
        for source in sources:
            source_id = source.get("source_id")
            if not source_id:
                continue
            source_key = (agent_id, source_id)
            if source_key in self._observed_source_keys:
                continue
            self._observed_source_keys.add(source_key)

            record = self._make_record(
                owner_agent=agent_id,
                content=str(source.get("content", "")),
                visibility="isolated",
                source_id=None,
                derived_from=[],
                trust_label=None,
                metadata={**metadata, "origin": "source_document"},
            )
            self._add_record(agent_id, record)
            written.append(record)
        return written

    def observe_tool_outputs(
        self,
        agent_id: str,
        tool_outputs: list[dict],
        task: dict,
        metadata: dict,
    ) -> list[MemoryRecord]:
        self._ensure_task_seeded(task)
        written = []
        for tool_output in tool_outputs:
            output_id = tool_output.get("tool_output_id") or tool_output.get("output_id") or tool_output.get("id")
            if not output_id:
                continue
            output_key = (agent_id, str(output_id))
            if output_key in self._observed_tool_output_keys:
                continue
            self._observed_tool_output_keys.add(output_key)

            content = tool_output.get("content")
            if content is None:
                content = tool_output.get("output")
            if content is None:
                content = tool_output.get("result")
            if content is None:
                content = json.dumps(tool_output, ensure_ascii=False)

            record = self._make_record(
                owner_agent=agent_id,
                content=str(content),
                visibility="isolated",
                source_id=None,
                derived_from=[],
                trust_label=None,
                metadata={
                    **metadata,
                    "origin": "tool_output",
                    "tool_output_id": str(output_id),
                },
            )
            self._add_record(agent_id, record)
            written.append(record)
        return written

    def retrieve(self, agent_id: str, query: str, task: dict) -> list[MemoryRecord]:
        self._ensure_task_seeded(task)
        records = self._records_by_agent.get(agent_id, {})
        if not records:
            return []

        collection = self._collections.get(agent_id)
        if collection is None:
            raise RuntimeError(f"Isolated vector collection for agent {agent_id!r} has not been initialized.")

        query_embedding = self.embedder.embed(query)[0]
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(self.top_k, len(records)),
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        return [records[memory_id] for memory_id in ids if memory_id in records]

    def write(self, agent_id: str, content: str, metadata: dict) -> MemoryRecord | None:
        if not content or not content.strip():
            return None

        record = self._make_record(
            owner_agent=agent_id,
            content=content,
            visibility="isolated",
            source_id=None,
            derived_from=[],
            trust_label=None,
            metadata={**metadata, "origin": "agent_message"},
        )
        self._add_record(agent_id, record)
        return record

    def all_memories(self) -> list[MemoryRecord]:
        return list(self._all_records)

    def _ensure_task_seeded(self, task: dict) -> None:
        task_id = task.get("task_id", getattr(self, "task_id", "unknown_task"))
        if self._seeded_task_id == task_id:
            return

        for event in task.get("initial_memory_events", task.get("memory_events", [])):
            content = str(event.get("content", ""))
            if not content.strip():
                continue
            owner = str(event.get("owner_agent") or event.get("agent_id") or "unknown_owner")
            record = MemoryRecord(
                memory_id=_safe_id(str(event.get("memory_id") or f"seed_{task_id}_{owner}_{len(self._all_records)+1}")),
                content=content,
                owner_agent=owner,
                visibility="isolated",
                source_id=None,
                derived_from=[],
                trust_label=None,
                metadata={
                    "task_id": task_id,
                    "domain": task.get("domain"),
                    "origin": "seed_memory_event",
                    "event_id": event.get("event_id"),
                    "event_type": event.get("event_type"),
                    "source_id": event.get("source_id"),
                    "raw_visibility": event.get("visibility"),
                    "raw_permission_scope": event.get("permission_scope"),
                    "note": "B2 stores seeded memory events only in the owner agent's isolated collection.",
                },
            )
            self._add_record(owner, record)

        self._seeded_task_id = task_id

    def _collection_for_agent(self, agent_id: str):
        if self._client is None:
            raise RuntimeError("Chroma client has not been initialized. Call reset_task first.")

        if agent_id not in self._collections:
            task_id = getattr(self, "task_id", "task")
            collection_key = f"{task_id}:{agent_id}"
            collection_hash = hashlib.sha1(collection_key.encode("utf-8")).hexdigest()[:12]
            collection_name = _safe_id(f"isolated_vector_{collection_hash}_{agent_id}")
            self._collections[agent_id] = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[agent_id]

    def _make_record(
        self,
        owner_agent: str,
        content: str,
        visibility: str,
        source_id: str | None,
        derived_from: list[str],
        trust_label: str | None,
        metadata: dict[str, Any],
    ) -> MemoryRecord:
        self._counter += 1
        task_id = getattr(self, "task_id", "task")
        prefix = str(metadata.get("origin") or "memory")
        memory_id = _safe_id(f"{prefix}_{task_id}_{owner_agent}_{self._counter:04d}")
        return MemoryRecord(
            memory_id=memory_id,
            content=content,
            owner_agent=owner_agent,
            visibility=visibility,
            source_id=source_id,
            derived_from=derived_from,
            trust_label=trust_label,
            metadata=metadata,
        )

    def _add_record(self, agent_id: str, record: MemoryRecord) -> None:
        collection = self._collection_for_agent(agent_id)
        embedding = self.embedder.embed(record.content)[0]
        collection.add(
            ids=[record.memory_id],
            embeddings=[embedding],
            documents=[record.content],
            metadatas=[self._metadata_for_chroma(record)],
        )
        self._records_by_agent.setdefault(agent_id, {})[record.memory_id] = record
        self._all_records.append(record)

    def _metadata_for_chroma(self, record: MemoryRecord) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {
            "memory_id": record.memory_id,
            "owner_agent": record.owner_agent,
        }
        for key, value in record.metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = json.dumps(value, ensure_ascii=True)
        return metadata
