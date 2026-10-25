from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from src.memory_baselines.base import BaseMemoryBackend, MemoryRecord
from src.utils.model_clients import EmbeddingClient, load_model_config


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)


class LineageEnforcementMemoryBackend(BaseMemoryBackend):
    """B5 MemLineage-adapted baseline for multi-agent workflows.

    This backend keeps a shared vector memory like B1, but attaches lineage
    ancestors and derived_from edges to every memory fragment. Before high-risk
    actions it blocks if a supporting memory has an untrusted ancestor. It does
    not implement permission-aware retrieval, multi-agent trust decay, agent
    contamination propagation rules, or a cryptographic/Merkle log.
    """

    method_name = "B5"

    UNTRUSTED_RISK_TAGS = {
        "untrusted",
        "poisoned",
        "withdrawn",
        "stale",
        "revoked",
        "risky_command",
        "dangerous_command",
        "privacy_violation",
        "credential",
    }
    UNTRUSTED_TRUST_LABELS = {
        "untrusted",
        "poisoned",
        "withdrawn",
        "stale",
        "revoked",
        "risky",
    }

    def __init__(self, top_k: int | None = None, untrusted_trust_threshold: float = 0.6):
        config = load_model_config()
        self.top_k = top_k or int(config.get("retrieval", {}).get("top_k", 5))
        self.untrusted_trust_threshold = untrusted_trust_threshold
        self.embedder = EmbeddingClient()
        self._client = None
        self._collection = None
        self._records: dict[str, MemoryRecord] = {}
        self._source_catalog: dict[str, dict[str, Any]] = {}
        self._untrusted_ancestors: dict[str, dict[str, Any]] = {}
        self._observed_source_keys: set[tuple[str, str]] = set()
        self._observed_tool_output_keys: set[tuple[str, str]] = set()
        self._seeded_task_id: str | None = None
        self._counter = 0

    def reset_task(self, task_id: str) -> None:
        super().reset_task(task_id)
        self._records = {}
        self._source_catalog = {}
        self._untrusted_ancestors = {}
        self._observed_source_keys = set()
        self._observed_tool_output_keys = set()
        self._seeded_task_id = None
        self._counter = 0

        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "B5 lineage_enforcement requires Chroma. Install it with:\n"
                "  conda activate mapgraph\n"
                "  pip install chromadb\n"
            ) from exc

        collection_hash = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:12]
        collection_name = f"lineage_enforcement_{collection_hash}"
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def observe_sources(self, agent_id: str, sources: list[dict], task: dict, metadata: dict) -> list[MemoryRecord]:
        self._ensure_task_seeded(task)
        written = []
        for source in sources:
            source_id = source.get("source_id")
            if not source_id:
                continue
            self._register_source(source)
            source_key = (agent_id, str(source_id))
            if source_key in self._observed_source_keys:
                continue
            self._observed_source_keys.add(source_key)

            lineage = self._lineage_for_references([str(source_id)])
            record = self._make_record(
                owner_agent=agent_id,
                content=str(source.get("content", "")),
                visibility="shared",
                source_id=str(source_id),
                derived_from=[str(source_id)],
                trust_label=self._trust_label_from_source(source),
                metadata={
                    **metadata,
                    "origin": "source_document",
                    "lineage_ancestors": lineage["ancestors"],
                    "derived_from_edges": lineage["edges"],
                    "untrusted_ancestors": lineage["untrusted_ancestors"],
                    "untrusted_ancestor_reasons": lineage["untrusted_reasons"],
                    "note": "B5 stores lineage for enforcement but does not filter retrieval by permission.",
                },
            )
            self._add_record(record)
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

            lineage = self._lineage_for_references([str(output_id)])
            record = self._make_record(
                owner_agent=agent_id,
                content=str(content),
                visibility="shared",
                source_id=None,
                derived_from=[str(output_id)],
                trust_label=None,
                metadata={
                    **metadata,
                    "origin": "tool_output",
                    "tool_output_id": str(output_id),
                    "lineage_ancestors": lineage["ancestors"],
                    "derived_from_edges": lineage["edges"],
                    "untrusted_ancestors": lineage["untrusted_ancestors"],
                    "untrusted_ancestor_reasons": lineage["untrusted_reasons"],
                },
            )
            self._add_record(record)
            written.append(record)
        return written

    def retrieve(self, agent_id: str, query: str, task: dict) -> list[MemoryRecord]:
        self._ensure_task_seeded(task)
        if not self._records:
            return []
        if self._collection is None:
            raise RuntimeError("Lineage vector collection has not been initialized. Call reset_task first.")

        query_embedding = self.embedder.embed(query)[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(self.top_k, len(self._records)),
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        return [self._records[memory_id] for memory_id in ids if memory_id in self._records]

    def write(self, agent_id: str, content: str, metadata: dict) -> MemoryRecord | None:
        if not content or not content.strip():
            return None

        derived_from = [
            str(item)
            for item in (metadata.get("retrieved_memory_ids", []) + metadata.get("visible_source_ids", []))
            if item
        ]
        lineage = self._lineage_for_references(derived_from)
        record = self._make_record(
            owner_agent=agent_id,
            content=content,
            visibility="shared",
            source_id=None,
            derived_from=derived_from,
            trust_label=None,
            metadata={
                **metadata,
                "origin": "agent_message",
                "lineage_ancestors": lineage["ancestors"],
                "derived_from_edges": lineage["edges"],
                "untrusted_ancestors": lineage["untrusted_ancestors"],
                "untrusted_ancestor_reasons": lineage["untrusted_reasons"],
            },
        )
        self._add_record(record)
        return record

    def all_memories(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def apply_timeline_event(self, event: dict, task: dict) -> None:
        if event.get("event_type") != "revoke_permission":
            return
        source_id = str(event.get("source_id") or "")
        if not source_id:
            return
        source = self._source_catalog.setdefault(source_id, {"source_id": source_id})
        source.update({"visibility": "revoked", "is_revoked": True, "permission_scope": []})
        self._register_source(source)
        for record in self._records.values():
            refs = set(record.derived_from)
            if source_id not in refs and source_id not in set(record.metadata.get("lineage_ancestors") or []):
                continue
            lineage = self._lineage_for_references(record.derived_from + [source_id])
            record.metadata.update(
                {
                    "lineage_ancestors": lineage["ancestors"],
                    "derived_from_edges": lineage["edges"],
                    "untrusted_ancestors": lineage["untrusted_ancestors"],
                    "untrusted_ancestor_reasons": lineage["untrusted_reasons"],
                }
            )

    def decide_action(self, action: dict, task: dict, supporting_memories: list[MemoryRecord]) -> dict | None:
        action_id = action.get("action_id") or f"{task.get('task_id', 'task')}_{action.get('agent_id', 'agent')}_action"
        risk_level = str(
            action.get("risk_level")
            or action.get("risk")
            or (task.get("expected_action") or {}).get("risk_level")
            or "medium"
        ).lower()
        supporting_memory_ids = [memory.memory_id for memory in supporting_memories]

        if risk_level != "high":
            return {
                "action_id": action_id,
                "agent_id": action.get("agent_id", "unknown_agent"),
                "action_type": action.get("action_type", action.get("type", "other")),
                "risk_level": risk_level,
                "supporting_memory_ids": supporting_memory_ids,
                "supporting_source_ids": action.get("supporting_source_ids", []),
                "decision": "ALLOW",
                "reason": "B5 MemLineage-adapted baseline only enforces untrusted lineage on high-risk actions.",
            }

        blocked = []
        for memory in supporting_memories:
            untrusted = memory.metadata.get("untrusted_ancestors") or []
            if untrusted:
                blocked.append(
                    {
                        "memory_id": memory.memory_id,
                        "untrusted_ancestors": untrusted,
                        "reasons": memory.metadata.get("untrusted_ancestor_reasons") or {},
                    }
                )

        if blocked:
            return {
                "action_id": action_id,
                "agent_id": action.get("agent_id", "unknown_agent"),
                "action_type": action.get("action_type", action.get("type", "other")),
                "risk_level": risk_level,
                "supporting_memory_ids": supporting_memory_ids,
                "supporting_source_ids": action.get("supporting_source_ids", []),
                "decision": "BLOCK",
                "reason": "B5 blocked high-risk action because supporting memory has an untrusted lineage ancestor.",
                "blocked_lineage": blocked,
            }

        return {
            "action_id": action_id,
            "agent_id": action.get("agent_id", "unknown_agent"),
            "action_type": action.get("action_type", action.get("type", "other")),
            "risk_level": risk_level,
            "supporting_memory_ids": supporting_memory_ids,
            "supporting_source_ids": action.get("supporting_source_ids", []),
            "decision": "ALLOW",
            "reason": "B5 found no untrusted ancestor in supporting memory lineage for this high-risk action.",
        }

    def _ensure_task_seeded(self, task: dict) -> None:
        task_id = task.get("task_id", getattr(self, "task_id", "unknown_task"))
        if self._seeded_task_id == task_id:
            return

        for source in task.get("source_documents", []):
            self._register_source(source)

        for event in task.get("initial_memory_events", task.get("memory_events", [])):
            content = str(event.get("content", ""))
            if not content.strip():
                continue
            owner = str(event.get("owner_agent") or event.get("agent_id") or "unknown_owner")
            references = [str(item) for item in event.get("derived_from") or [] if item]
            source_id = event.get("source_id")
            if source_id and str(source_id) not in references:
                references.append(str(source_id))
            lineage = self._lineage_for_references(references)
            record = MemoryRecord(
                memory_id=_safe_id(str(event.get("memory_id") or f"seed_{task_id}_{owner}_{len(self._records)+1}")),
                content=content,
                owner_agent=owner,
                visibility="shared",
                source_id=str(source_id) if source_id else None,
                derived_from=references,
                trust_label=event.get("trust_label"),
                metadata={
                    "task_id": task_id,
                    "domain": task.get("domain"),
                    "origin": "seed_memory_event",
                    "event_id": event.get("event_id"),
                    "event_type": event.get("event_type"),
                    "raw_visibility": event.get("visibility"),
                    "raw_permission_scope": event.get("permission_scope"),
                    "lineage_ancestors": lineage["ancestors"],
                    "derived_from_edges": lineage["edges"],
                    "untrusted_ancestors": lineage["untrusted_ancestors"],
                    "untrusted_ancestor_reasons": lineage["untrusted_reasons"],
                    "note": "B5 seeds memory events with lineage but no cryptographic/Merkle log.",
                },
            )
            self._add_record(record)

        self._seeded_task_id = task_id

    def _register_source(self, source: dict) -> None:
        source_id = source.get("source_id")
        if not source_id:
            return
        source_id = str(source_id)
        self._source_catalog[source_id] = source
        reason = self._untrusted_source_reason(source)
        if reason:
            self._untrusted_ancestors[source_id] = reason

    def _lineage_for_references(self, references: list[str]) -> dict[str, Any]:
        ancestors: list[str] = []
        edges: list[dict[str, str]] = []
        untrusted_reasons: dict[str, Any] = {}

        def add_ancestor(ancestor_id: str) -> None:
            if ancestor_id not in ancestors:
                ancestors.append(ancestor_id)
            if ancestor_id in self._untrusted_ancestors:
                untrusted_reasons[ancestor_id] = self._untrusted_ancestors[ancestor_id]

        for ref in references:
            if not ref:
                continue
            add_ancestor(ref)
            edges.append({"from": ref, "to": "current_memory"})
            parent = self._records.get(ref)
            if parent is None:
                continue
            for ancestor in parent.metadata.get("lineage_ancestors") or []:
                add_ancestor(str(ancestor))
                edges.append({"from": str(ancestor), "to": ref})
            for edge in parent.metadata.get("derived_from_edges") or []:
                if isinstance(edge, dict):
                    edges.append(edge)

        return {
            "ancestors": ancestors,
            "edges": edges,
            "untrusted_ancestors": list(untrusted_reasons),
            "untrusted_reasons": untrusted_reasons,
        }

    def _untrusted_source_reason(self, source: dict) -> dict[str, Any] | None:
        risk_tags = set(source.get("risk_tags") or [])
        trust = source.get("trust")
        trust_value = float(trust) if trust is not None else 1.0
        reasons = []
        if str(source.get("owner", "")).lower() in {"unknown", "external"}:
            reasons.append("external_or_unknown_owner")
        if risk_tags & self.UNTRUSTED_RISK_TAGS:
            reasons.append("untrusted_risk_tag")
        if source.get("is_poisoned"):
            reasons.append("poisoned_source")
        if source.get("is_revoked"):
            reasons.append("revoked_or_withdrawn_source")
        if trust_value < self.untrusted_trust_threshold:
            reasons.append("low_trust_score")
        if not reasons:
            return None
        return {
            "source_id": source.get("source_id"),
            "reasons": reasons,
            "trust": trust,
            "risk_tags": sorted(risk_tags),
            "owner": source.get("owner"),
        }

    def _trust_label_from_source(self, source: dict) -> str | None:
        reason = self._untrusted_source_reason(source)
        if reason:
            return "untrusted"
        risk_tags = source.get("risk_tags") or []
        if "trusted" in risk_tags:
            return "trusted"
        return None

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

    def _add_record(self, record: MemoryRecord) -> None:
        if self._collection is None:
            raise RuntimeError("Lineage vector collection has not been initialized. Call reset_task first.")

        embedding = self.embedder.embed(record.content)[0]
        self._collection.add(
            ids=[record.memory_id],
            embeddings=[embedding],
            documents=[record.content],
            metadatas=[self._metadata_for_chroma(record)],
        )
        self._records[record.memory_id] = record

    def _metadata_for_chroma(self, record: MemoryRecord) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {
            "memory_id": record.memory_id,
            "owner_agent": record.owner_agent,
            "visibility": record.visibility,
        }
        for key, value in record.metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = json.dumps(value, ensure_ascii=True)
        return metadata
