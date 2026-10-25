from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.map_graph.graph_builder import MAPGraphBuilder
from src.map_graph.trust import MAPGraphTrustEngine
from src.memory_baselines.base import BaseMemoryBackend, MemoryRecord
from src.utils.model_clients import EmbeddingClient, load_model_config


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)


@dataclass(frozen=True)
class MAPGraphAblationConfig:
    """Component switches used by MAP-Graph ablation experiments."""

    name: str = "full"
    enable_graph_builder: bool = True
    enable_permission_filter: bool = True
    enable_trust_propagation: bool = True
    enable_trust_aware_reranking: bool = True
    enable_action_gate: bool = True
    enable_contamination_containment: bool = True
    enable_contextual_action_support: bool = True
    graph_detail: str = "full"

    @property
    def method_name(self) -> str:
        if self.name in {"", "full", "MAPGraph"}:
            return "MAPGraph"
        return f"MAPGraph_{_safe_id(self.name)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enable_graph_builder": self.enable_graph_builder,
            "enable_permission_filter": self.enable_permission_filter,
            "enable_trust_propagation": self.enable_trust_propagation,
            "enable_trust_aware_reranking": self.enable_trust_aware_reranking,
            "enable_action_gate": self.enable_action_gate,
            "enable_contamination_containment": self.enable_contamination_containment,
            "enable_contextual_action_support": self.enable_contextual_action_support,
            "graph_detail": self.graph_detail,
        }


ABLATION_PRESETS: dict[str, MAPGraphAblationConfig] = {
    "full": MAPGraphAblationConfig(name="full"),
    "wo_provenance_graph": MAPGraphAblationConfig(
        name="wo_provenance_graph",
        enable_graph_builder=False,
        enable_permission_filter=False,
        enable_trust_propagation=False,
        enable_trust_aware_reranking=False,
        enable_action_gate=False,
        enable_contamination_containment=False,
        enable_contextual_action_support=False,
        graph_detail="none",
    ),
    "wo_permission_filter": MAPGraphAblationConfig(name="wo_permission_filter", enable_permission_filter=False),
    "wo_trust_propagation": MAPGraphAblationConfig(
        name="wo_trust_propagation",
        enable_trust_propagation=False,
        enable_trust_aware_reranking=False,
    ),
    "wo_action_time_gate": MAPGraphAblationConfig(name="wo_action_time_gate", enable_action_gate=False),
    "wo_contamination_containment": MAPGraphAblationConfig(
        name="wo_contamination_containment",
        enable_contamination_containment=False,
    ),
    "compressed_graph_only": MAPGraphAblationConfig(
        name="compressed_graph_only",
        graph_detail="compressed",
    ),
    # Backward-compatible optional aliases. The six variants above are the ones
    # specified in the experiment document.
    "no_permission_filter": MAPGraphAblationConfig(name="wo_permission_filter", enable_permission_filter=False),
    "no_trust_propagation": MAPGraphAblationConfig(
        name="wo_trust_propagation",
        enable_trust_propagation=False,
        enable_trust_aware_reranking=False,
    ),
    "no_trust_reranking": MAPGraphAblationConfig(name="no_trust_reranking", enable_trust_aware_reranking=False),
    "no_action_gate": MAPGraphAblationConfig(name="wo_action_time_gate", enable_action_gate=False),
    "no_contamination_containment": MAPGraphAblationConfig(
        name="wo_contamination_containment",
        enable_contamination_containment=False,
    ),
    "retrieval_only": MAPGraphAblationConfig(
        name="wo_provenance_graph",
        enable_graph_builder=False,
        enable_permission_filter=False,
        enable_trust_propagation=False,
        enable_trust_aware_reranking=False,
        enable_action_gate=False,
        enable_contamination_containment=False,
        enable_contextual_action_support=False,
        graph_detail="none",
    ),
}


def ablation_config_from_name(name: str) -> MAPGraphAblationConfig:
    if name not in ABLATION_PRESETS:
        valid = ", ".join(sorted(ABLATION_PRESETS))
        raise ValueError(f"Unknown MAPGraph ablation variant {name!r}. Valid variants: {valid}")
    return ABLATION_PRESETS[name]


class MAPGraphMemoryBackend(BaseMemoryBackend):
    """Full MAP-Graph memory backend.

    Implements graph construction, permission-aware retrieval, path-based trust
    propagation, trust-aware reranking, contamination containment, and
    action-time gating according to the experiment document.
    """

    method_name = "MAPGraph"

    def __init__(self, top_k: int | None = None, ablation: MAPGraphAblationConfig | None = None):
        config = load_model_config()
        self.ablation = ablation or ABLATION_PRESETS["full"]
        self.method_name = self.ablation.method_name
        self.top_k = top_k or int(config.get("retrieval", {}).get("top_k", 5))
        self.embedder = EmbeddingClient()
        self.trust_engine = MAPGraphTrustEngine(
            enable_trust_propagation=self.ablation.enable_trust_propagation,
            enable_action_gate=self.ablation.enable_action_gate,
            enable_contamination_containment=self.ablation.enable_contamination_containment,
        )
        self.graph_builder = MAPGraphBuilder()
        self._client = None
        self._collection = None
        self._records: dict[str, MemoryRecord] = {}
        self._source_catalog: dict[str, dict[str, Any]] = {}
        self._task_agents: list[str] = []
        self._observed_source_keys: set[tuple[str, str]] = set()
        self._observed_tool_output_keys: set[tuple[str, str]] = set()
        self._seeded_task_id: str | None = None
        self._counter = 0
        self._retrieval_events: list[dict[str, Any]] = []
        self._gate_events: list[dict[str, Any]] = []
        self._latest_trust: dict[str, dict[str, Any]] = {}
        self._claim_counter = 0
        self._message_claim_ids: dict[str, list[str]] = {}
        self._last_retrieval: dict[str, Any] = {}

    def reset_task(self, task_id: str) -> None:
        super().reset_task(task_id)
        self._records = {}
        self._source_catalog = {}
        self._task_agents = []
        self._observed_source_keys = set()
        self._observed_tool_output_keys = set()
        self._seeded_task_id = None
        self._counter = 0
        self._retrieval_events = []
        self._gate_events = []
        self._latest_trust = {}
        self._claim_counter = 0
        self._message_claim_ids = {}
        self._last_retrieval = {}
        self.graph_builder.reset(task_id)

        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "MAPGraph requires Chroma. Install it with:\n"
                "  conda activate mapgraph\n"
                "  pip install chromadb\n"
            ) from exc

        collection_hash = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:12]
        collection_name = f"map_graph_{collection_hash}"
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
            source_id = str(source_id)
            self._register_source(source)
            if self._full_graph_enabled():
                self.graph_builder.add_observed_from(agent_id, source_id, **metadata)
            source_key = (agent_id, source_id)
            if source_key in self._observed_source_keys:
                continue
            self._observed_source_keys.add(source_key)

            record = self._make_record(
                owner_agent=agent_id,
                content=str(source.get("content", "")),
                visibility=str(source.get("visibility", "shared")),
                source_id=source_id,
                derived_from=[source_id],
                trust_label=self._trust_label_from_payload(source),
                metadata={
                    **metadata,
                    "origin": "source_document",
                    "permission_scope": self._permission_scope(source, task, default_owner=agent_id),
                    "source_visibility": source.get("visibility"),
                    "source_trust": source.get("trust"),
                    "source_risk_tags": source.get("risk_tags", []),
                    "source_owner": source.get("owner"),
                },
            )
            self._add_record(record, current_agent=agent_id, action=None)
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
            output_id = str(output_id)
            output_key = (agent_id, output_id)
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

            tool_id = str(tool_output.get("tool_name") or tool_output.get("tool") or "tool")
            if self._full_graph_enabled():
                self.graph_builder.add_tool(tool_id, tool_type=tool_id, risk_level=tool_output.get("risk_level"))
            record = self._make_record(
                owner_agent=agent_id,
                content=str(content),
                visibility=str(tool_output.get("visibility", "shared")),
                source_id=None,
                derived_from=[output_id],
                trust_label=self._trust_label_from_payload(tool_output),
                metadata={
                    **metadata,
                    "origin": "tool_output",
                    "tool_output_id": output_id,
                    "tool_id": tool_id,
                    "permission_scope": self._permission_scope(tool_output, task, default_owner=agent_id),
                },
            )
            self._add_record(record, current_agent=agent_id, action=None)
            written.append(record)
        return written

    def retrieve(self, agent_id: str, query: str, task: dict) -> list[MemoryRecord]:
        self._ensure_task_seeded(task)
        if not self._records:
            return []
        if self._collection is None:
            raise RuntimeError("MAPGraph vector collection has not been initialized. Call reset_task first.")

        action = task.get("expected_action") or {}
        query_embedding = self.embedder.embed(query)[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=len(self._records),
            include=["distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        candidates: list[tuple[float, float, float, MemoryRecord, dict[str, Any]]] = []
        filtered_out: list[dict[str, Any]] = []

        for memory_id, distance in zip(ids, distances):
            record = self._records.get(memory_id)
            if record is None:
                continue
            semantic_score = max(0.0, 1.0 - float(distance))
            trust_result = self.trust_engine.compute_memory_trust(
                record,
                agent_id,
                action,
                self._source_catalog,
                self._records,
                self._can_read,
                self._agent_reliability_map(),
            )
            record.metadata.update(trust_result.to_dict())
            record.metadata["last_query_agent"] = agent_id
            record.metadata["semantic_score"] = round(semantic_score, 6)
            if self.ablation.enable_permission_filter and not self._can_read(agent_id, record):
                filtered_out.append(
                    {
                        "memory_id": record.memory_id,
                        "reason": "permission_filter",
                        "visibility": record.visibility,
                        "permission_scope": record.metadata.get("permission_scope"),
                    }
                )
                continue
            final_score = semantic_score * trust_result.trust if self.ablation.enable_trust_aware_reranking else semantic_score
            candidates.append((final_score, semantic_score, trust_result.trust, record, trust_result.to_dict()))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [record for _, _, _, record, _ in candidates[: self.top_k]]
        for final_score, semantic_score, path_trust, record, trust_payload in candidates[: self.top_k]:
            record.metadata["map_graph_final_score"] = round(final_score, 6)
            record.metadata["semantic_score"] = round(semantic_score, 6)
            record.metadata["path_trust"] = round(path_trust, 6)
            record.metadata["trust_aware_reranking"] = {
                "semantic_score": round(semantic_score, 6),
                "path_trust": round(path_trust, 6),
                "final_score": round(final_score, 6),
            }
            if self._full_graph_enabled():
                self.graph_builder.add_memory_read(record.memory_id, agent_id, final_score=final_score, path_trust=path_trust)
            self._latest_trust[record.memory_id] = trust_payload

        self._retrieval_events.append(
            {
                "agent_id": agent_id,
                "query_hash": hashlib.sha1(query.encode("utf-8")).hexdigest(),
                "candidate_count": len(ids),
                "permission_filtered_count": len(filtered_out),
                "permission_filtered": filtered_out[:20],
                "selected": [
                    {
                        "memory_id": record.memory_id,
                        "final_score": record.metadata.get("map_graph_final_score"),
                        "path_trust": record.metadata.get("path_trust"),
                        "semantic_score": record.metadata.get("semantic_score"),
                    }
                    for record in selected
                ],
                "top_k": self.top_k,
                "ablation": self.ablation.to_dict(),
            }
        )
        self._last_retrieval = {
            "agent_id": agent_id,
            "candidate_memory_ids": [str(memory_id) for memory_id in ids],
            "permission_denied_memory_ids": [item["memory_id"] for item in filtered_out],
            "selected_memory_ids": [record.memory_id for record in selected],
        }
        return selected

    def write(self, agent_id: str, content: str, metadata: dict) -> MemoryRecord | None:
        if not content or not content.strip():
            return None
        derived_from = [
            str(item)
            for item in (metadata.get("retrieved_memory_ids", []) + metadata.get("visible_source_ids", []))
            if item
        ]
        permission_scope = self._derived_permission_scope(derived_from, agent_id)
        visibility = "shared" if set(permission_scope) >= set(self._task_agents) else "private"
        message_id = _safe_id(f"msg_{getattr(self, 'task_id', 'task')}_{agent_id}_{metadata.get('step', self._counter + 1)}")
        receivers = [agent for agent in permission_scope if agent != agent_id]
        if self._full_graph_enabled():
            self.graph_builder.add_message(message_id, agent_id, receivers, content, derived_from=derived_from, **metadata)
            record_derived_from = derived_from + [message_id]
        else:
            record_derived_from = derived_from
        record = self._make_record(
            owner_agent=agent_id,
            content=content,
            visibility=visibility,
            source_id=None,
            derived_from=record_derived_from,
            trust_label=None,
            metadata={
                **metadata,
                "origin": "agent_message",
                "message_id": message_id,
                "permission_scope": permission_scope,
            },
        )
        self._add_record(record, current_agent=agent_id, action=None)
        if self._full_graph_enabled():
            self._add_claims_for_message(
                agent_id=agent_id,
                message_id=message_id,
                content=content,
                derived_from=record.derived_from,
            )
        return record

    def filter_conversation(
        self,
        agent_id: str,
        conversation: list[dict[str, Any]],
        task: dict,
    ) -> list[dict[str, Any]]:
        if not self.ablation.enable_permission_filter:
            return list(conversation)
        return [
            item
            for item in conversation
            if "permission_scope" not in item
            or str(agent_id) in {str(agent) for agent in item.get("permission_scope") or []}
        ]

    def all_memories(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def last_retrieval_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_retrieval)

    def apply_timeline_event(self, event: dict, task: dict) -> None:
        if event.get("event_type") != "revoke_permission":
            return
        source_id = str(event.get("source_id") or "")
        if not source_id:
            return
        source = self._source_catalog.setdefault(source_id, {"source_id": source_id})
        source.update({"visibility": "revoked", "is_revoked": True, "permission_scope": []})
        if self._full_graph_enabled():
            resource_node = self.graph_builder.resource_node(source_id)
            if self.graph_builder.graph.has_node(resource_node):
                self.graph_builder.graph.nodes[resource_node].update(
                    {"visibility": "revoked", "is_revoked": True, "permission_scope": []}
                )
        for record in self._records.values():
            refs = set(record.derived_from)
            if source_id not in refs and source_id not in set(record.metadata.get("affected_ancestors") or []):
                continue
            result = self.trust_engine.compute_memory_trust(
                record,
                str(event.get("agent_id") or "unknown_agent"),
                None,
                self._source_catalog,
                self._records,
                self._can_read,
                self._agent_reliability_map(),
            )
            record.metadata.update(result.to_dict())
            record.metadata["affected"] = bool(result.affected_ancestors)
            if self._full_graph_enabled():
                self.graph_builder.add_invalidated_by(record.memory_id, source_id, event_id=event.get("event_id"))

    def decide_action(self, action: dict, task: dict, supporting_memories: list[MemoryRecord]) -> dict | None:
        self._ensure_task_seeded(task)
        action_type = action.get("action_type") or action.get("type") or (task.get("expected_action") or {}).get("type") or "other"
        risk_level = action.get("risk_level") or action.get("risk") or (task.get("expected_action") or {}).get("risk_level") or "medium"
        agent_id = str(action.get("agent_id") or "unknown_agent")
        retrieved_supporting_ids = [memory.memory_id for memory in supporting_memories]
        contextual_supporting = self._contextual_action_support(agent_id) if self.ablation.enable_contextual_action_support else []
        combined_supporting = list(supporting_memories)
        seen_supporting = set(retrieved_supporting_ids)
        for memory in contextual_supporting:
            if memory.memory_id in seen_supporting:
                continue
            combined_supporting.append(memory)
            seen_supporting.add(memory.memory_id)

        trust_results = {}
        for memory in combined_supporting:
            trust_results[memory.memory_id] = self.trust_engine.compute_memory_trust(
                memory,
                agent_id,
                {"action_type": action_type, "risk_level": risk_level},
                self._source_catalog,
                self._records,
                self._can_read,
                self._agent_reliability_map(),
            )
            memory.metadata.update(trust_results[memory.memory_id].to_dict())

        decision = self.trust_engine.gate_action(action, task, combined_supporting, trust_results)
        decision["retrieved_supporting_memory_ids"] = retrieved_supporting_ids
        decision["contextual_supporting_memory_ids"] = [memory.memory_id for memory in contextual_supporting]
        action_claim_ids = (
            self._add_claims_for_action(
                agent_id=agent_id,
                action=action,
                decision=decision,
                supporting_memory_ids=[memory.memory_id for memory in combined_supporting],
            )
            if self._full_graph_enabled()
            else []
        )
        decision["supporting_claim_ids"] = action_claim_ids
        if self._graph_enabled():
            self.graph_builder.add_action(
                action_id=decision["action_id"],
                action_type=str(action_type),
                risk_level=str(risk_level),
                decision=decision["decision"],
                supporting_memory_ids=decision["supporting_memory_ids"],
                agent_id=agent_id,
                path_trust_min=decision.get("path_trust_min"),
                required_trust_threshold=decision.get("required_trust_threshold"),
            )
            for claim_id in action_claim_ids:
                self.graph_builder.add_claim_used_for_action(claim_id, decision["action_id"])
        self._gate_events.append(decision)
        return decision

    def finalize_task(
        self,
        task: dict,
        events: list[dict],
        decisions: list[dict],
        conversation: list[dict[str, str]],
    ) -> None:
        affected_roots = []
        for source_id, source in self._source_catalog.items():
            if self._payload_is_affected(source):
                affected_roots.append(source_id)
        for memory in self._records.values():
            if memory.metadata.get("affected"):
                affected_roots.append(memory.memory_id)
        if self._graph_enabled() and self.ablation.enable_contamination_containment:
            self.graph_builder.propagate_affected(affected_roots)

    def export_graph(self) -> dict[str, Any]:
        payload = self.graph_builder.export()
        payload["method"] = self.method_name
        payload["ablation"] = self.ablation.to_dict()
        payload["retrieval_events"] = deepcopy(self._retrieval_events)
        payload["gate_events"] = [
            self._strip_evaluator_fields(event)
            for event in self._gate_events
        ]
        return payload

    def _strip_evaluator_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = deepcopy(payload)
        for key in (
            "expected_decision",
            "ground_truth_decision",
            "should_block",
            "risk_label",
            "experiment_group",
            "case_family",
            "generation_spec",
            "diversity_dimensions",
        ):
            cleaned.pop(key, None)
        return cleaned

    def graph_summary(self) -> dict[str, Any]:
        graph = self.graph_builder.export()
        return {
            "task_id": graph["task_id"],
            "method": self.method_name,
            "ablation": self.ablation.to_dict(),
            "node_counts": graph["node_counts"],
            "edge_counts": graph["edge_counts"],
            "retrieval_events": len(self._retrieval_events),
            "gate_events": len(self._gate_events),
        }

    def _ensure_task_seeded(self, task: dict) -> None:
        task_id = task.get("task_id", getattr(self, "task_id", "unknown_task"))
        if self._seeded_task_id == task_id:
            return
        self._task_agents = [str(agent) for agent in task.get("agent_roles", [])]
        if self._graph_enabled():
            self.graph_builder.register_task(task)
        for source in task.get("source_documents", []):
            self._register_source(source)
        for event in task.get("initial_memory_events", task.get("memory_events", [])):
            self._seed_memory_event(event, task)
        self._seeded_task_id = task_id

    def _seed_memory_event(self, event: dict, task: dict) -> None:
        memory_id = event.get("memory_id")
        content = str(event.get("content", ""))
        if not memory_id or memory_id in self._records or not content.strip():
            return
        owner = str(event.get("owner_agent") or event.get("agent_id") or "unknown_owner")
        source_id = event.get("source_id")
        derived_from = [str(item) for item in event.get("derived_from") or [] if item]
        if source_id and str(source_id) not in derived_from:
            derived_from.append(str(source_id))
        record = MemoryRecord(
            memory_id=_safe_id(str(memory_id)),
            content=content,
            owner_agent=owner,
            visibility=str(event.get("visibility", "shared")),
            source_id=str(source_id) if source_id else None,
            derived_from=derived_from,
            trust_label=event.get("trust_label"),
            metadata={
                "task_id": task.get("task_id"),
                "domain": task.get("domain"),
                "origin": "seed_memory_event",
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "permission_scope": self._permission_scope(event, task, default_owner=owner),
                "raw_visibility": event.get("visibility"),
                "raw_permission_scope": event.get("permission_scope"),
            },
        )
        self._add_record(record, current_agent=owner, action=None, fixed_id=True)
        if self._full_graph_enabled() and (event.get("event_type") == "revoke_permission" or event.get("visibility") == "revoked"):
            for ref in derived_from:
                self.graph_builder.add_invalidated_by(record.memory_id, ref, event_id=event.get("event_id"))

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

    def _add_record(
        self,
        record: MemoryRecord,
        current_agent: str,
        action: dict | None,
        fixed_id: bool = False,
    ) -> None:
        if self._collection is None:
            raise RuntimeError("MAPGraph vector collection has not been initialized. Call reset_task first.")
        if not fixed_id:
            record.memory_id = _safe_id(record.memory_id)

        trust_result = self.trust_engine.compute_memory_trust(
            record,
            current_agent,
            action,
            self._source_catalog,
            self._records,
            self._can_read,
            self._agent_reliability_map(),
        )
        record.metadata.update(trust_result.to_dict())
        record.metadata["affected"] = bool(
            self.ablation.enable_contamination_containment
            and trust_result.affected_ancestors
        )
        record.metadata["map_graph_components"] = {
            "graph_builder": self.ablation.enable_graph_builder,
            "permission_filter": self.ablation.enable_permission_filter,
            "path_trust_propagation": self.ablation.enable_trust_propagation,
            "trust_aware_reranking": self.ablation.enable_trust_aware_reranking,
            "action_time_gate": self.ablation.enable_action_gate,
            "contamination_containment": self.ablation.enable_contamination_containment,
            "contextual_action_support": self.ablation.enable_contextual_action_support,
        }
        self._records[record.memory_id] = record
        self._collection.add(
            ids=[record.memory_id],
            embeddings=[self.embedder.embed(record.content)[0]],
            documents=[record.content],
            metadatas=[self._metadata_for_chroma(record)],
        )
        if self._graph_enabled():
            self.graph_builder.add_memory(
                record.memory_id,
                record.content,
                record.owner_agent,
                record.trust_label,
                record.visibility,
                record.derived_from,
                record.metadata,
            )
        if self._full_graph_enabled():
            self._add_verification_edges(record)

    def _register_source(self, source: dict) -> None:
        source_id = source.get("source_id")
        if not source_id:
            return
        self._source_catalog[str(source_id)] = source
        if self._graph_enabled():
            self.graph_builder.add_resource(source)

    def _permission_scope(self, payload: dict, task: dict, default_owner: str) -> list[str]:
        scope = payload.get("permission_scope")
        if scope:
            return [str(item) for item in scope]
        if payload.get("visibility") == "shared":
            return [str(agent) for agent in task.get("agent_roles", [])]
        return [str(default_owner)]

    def _derived_permission_scope(self, references: list[str], default_owner: str) -> list[str]:
        scopes: list[set[str]] = []
        all_agents = set(self._task_agents or [default_owner])
        for ref in references:
            source = self._source_catalog.get(str(ref))
            if source is not None:
                source_scope = set(str(item) for item in source.get("permission_scope") or [])
                if not source_scope:
                    source_scope = all_agents if source.get("visibility") == "shared" else {str(default_owner)}
                scopes.append(source_scope)
                continue

            parent = self._records.get(str(ref))
            if parent is None:
                continue
            parent_scope = set(str(item) for item in parent.metadata.get("permission_scope") or [])
            if not parent_scope:
                parent_scope = all_agents if parent.visibility == "shared" else {str(parent.owner_agent)}
            scopes.append(parent_scope)

        if not scopes:
            return sorted(all_agents)
        allowed = set.intersection(*scopes)
        return sorted(allowed or {str(default_owner)})

    def _can_read(self, agent_id: str, record: MemoryRecord) -> bool:
        agent_id = str(agent_id)
        if record.metadata.get("permission_validity") == 0.0:
            pass
        if not self.ablation.enable_permission_filter:
            return True
        permission_scope = set(str(item) for item in record.metadata.get("permission_scope") or [])
        if not permission_scope:
            permission_scope = {str(record.owner_agent)}
        visibility = str(record.visibility or record.metadata.get("raw_visibility") or "").lower()
        if visibility == "shared":
            own_record_allowed = agent_id in permission_scope or permission_scope >= set(self._task_agents)
        else:
            own_record_allowed = agent_id in permission_scope or agent_id == str(record.owner_agent)
        if visibility == "revoked":
            own_record_allowed = False
        if not own_record_allowed:
            return False

        for ref in record.derived_from:
            source = self._source_catalog.get(str(ref))
            if source is None:
                continue
            source_visibility = str(source.get("visibility", "shared")).lower()
            source_scope = set(str(item) for item in source.get("permission_scope") or [])
            if source_visibility == "revoked" or source.get("is_revoked"):
                return False
            if source_visibility != "shared" and agent_id not in source_scope:
                return False
        return True

    def _payload_is_affected(self, payload: dict) -> bool:
        tags = set(payload.get("risk_tags") or [])
        trust_label = str(payload.get("trust_label") or "").lower()
        visibility = str(payload.get("visibility") or payload.get("raw_visibility") or "").lower()
        trust = float(payload.get("trust", 1.0) if payload.get("trust") is not None else 1.0)
        return (
            bool(payload.get("is_poisoned"))
            or bool(payload.get("is_private"))
            or bool(payload.get("is_revoked"))
            or visibility in {"private", "hr_only", "executor_only", "verifier_only", "revoked"}
            or trust_label in {"poisoned", "private", "revoked", "withdrawn", "untrusted", "risky"}
            or bool(tags & {"poisoned", "private", "revoked", "withdrawn", "untrusted", "stale", "credential", "dangerous_command", "risky_command", "privacy_violation"})
            or trust < 0.6
        )

    def _trust_label_from_payload(self, payload: dict) -> str | None:
        if payload.get("trust_label"):
            return str(payload.get("trust_label"))
        if self._payload_is_affected(payload):
            return "affected"
        if "trusted" in set(payload.get("risk_tags") or []):
            return "trusted"
        return None

    def _agent_reliability_map(self) -> dict[str, float]:
        reliability = {agent: 0.95 for agent in self._task_agents}
        reliability["unknown"] = 0.50
        reliability["external"] = 0.50
        return reliability

    def _graph_enabled(self) -> bool:
        return self.ablation.enable_graph_builder and self.ablation.graph_detail != "none"

    def _full_graph_enabled(self) -> bool:
        return self._graph_enabled() and self.ablation.graph_detail == "full"

    def _add_claims_for_message(
        self,
        agent_id: str,
        message_id: str,
        content: str,
        derived_from: list[str],
    ) -> list[str]:
        if not self._full_graph_enabled():
            return []
        claims = self._extract_claim_texts(content)
        claim_ids: list[str] = []
        for claim_text in claims:
            claim_id = self._new_claim_id("message")
            confidence = self._claim_confidence(derived_from)
            self.graph_builder.add_claim(
                claim_id=claim_id,
                text=claim_text,
                confidence=confidence,
                derived_from=[message_id] + derived_from,
                owner_agent=agent_id,
                origin="agent_message",
            )
            self._add_claim_verification_edges(claim_id, derived_from)
            claim_ids.append(claim_id)
        self._message_claim_ids[message_id] = claim_ids
        return claim_ids

    def _add_claims_for_action(
        self,
        agent_id: str,
        action: dict,
        decision: dict,
        supporting_memory_ids: list[str],
    ) -> list[str]:
        if not self._full_graph_enabled():
            return []
        action_text = str(action.get("content") or action.get("rationale") or decision.get("reason") or "")
        if not action_text.strip():
            action_text = f"{decision.get('decision')} {decision.get('action_type')} based on MAP-Graph gate."
        claims = self._extract_claim_texts(action_text)
        if not claims:
            claims = [action_text[:300]]
        claim_ids: list[str] = []
        for claim_text in claims[:3]:
            claim_id = self._new_claim_id("action")
            confidence = max(0.0, min(1.0, float(decision.get("path_trust_max", 0.0) or 0.0)))
            self.graph_builder.add_claim(
                claim_id=claim_id,
                text=claim_text,
                confidence=confidence,
                derived_from=supporting_memory_ids,
                owner_agent=agent_id,
                origin="proposed_action",
                decision=decision.get("decision"),
            )
            self._add_claim_verification_edges(claim_id, supporting_memory_ids)
            claim_ids.append(claim_id)
        return claim_ids

    def _extract_claim_texts(self, content: str) -> list[str]:
        text = re.sub(r"\s+", " ", str(content or "")).strip()
        if not text:
            return []
        parts = re.split(r"(?<=[。.!?？])\s+|[；;]\s*", text)
        claims = []
        for part in parts:
            part = part.strip(" -•\t")
            if len(part) < 12:
                continue
            claims.append(part[:300])
            if len(claims) >= 3:
                break
        if not claims and len(text) >= 12:
            claims.append(text[:300])
        return claims

    def _claim_confidence(self, derived_from: list[str]) -> float:
        scores = []
        for ref in derived_from:
            memory = self._records.get(str(ref))
            if memory is not None:
                scores.append(float(memory.metadata.get("path_trust", 0.5) or 0.5))
                continue
            source = self._source_catalog.get(str(ref))
            if source is not None:
                scores.append(float(source.get("trust", 0.5) if source.get("trust") is not None else 0.5))
        return round(min(scores) if scores else 0.5, 6)

    def _new_claim_id(self, prefix: str) -> str:
        self._claim_counter += 1
        task_id = getattr(self, "task_id", "task")
        return _safe_id(f"claim_{prefix}_{task_id}_{self._claim_counter:04d}")

    def _add_verification_edges(self, record: MemoryRecord) -> None:
        if not self._full_graph_enabled():
            return
        verified = False
        for ref in record.derived_from:
            source = self._source_catalog.get(str(ref))
            if source is None:
                continue
            trust = float(source.get("trust", 0.0) if source.get("trust") is not None else 0.0)
            if trust >= 0.9 or "trusted" in set(source.get("risk_tags") or []):
                self.graph_builder.add_verified_by(
                    record.memory_id,
                    str(ref),
                    verifier_type="Resource",
                    source_trust=trust,
                )
                verified = True
        if not verified and record.metadata.get("path_trust", 0.0) >= 0.85:
            verifier_id = "mapgraph_path_trust_verifier"
            self.graph_builder.add_verified_by(
                record.memory_id,
                verifier_id,
                verifier_type="Tool",
                tool_type="trust_verifier",
                risk_level="low",
                path_trust=record.metadata.get("path_trust"),
            )

    def _add_claim_verification_edges(self, claim_id: str, derived_from: list[str]) -> None:
        if not self._full_graph_enabled():
            return
        for ref in derived_from:
            source = self._source_catalog.get(str(ref))
            if source is not None:
                trust = float(source.get("trust", 0.0) if source.get("trust") is not None else 0.0)
                if trust >= 0.9 or "trusted" in set(source.get("risk_tags") or []):
                    self.graph_builder.add_verified_by(
                        claim_id,
                        str(ref),
                        verifier_type="Resource",
                        source_trust=trust,
                    )
                continue
            memory = self._records.get(str(ref))
            if memory is not None and float(memory.metadata.get("path_trust", 0.0) or 0.0) >= 0.85:
                self.graph_builder.add_verified_by(
                    claim_id,
                    "mapgraph_path_trust_verifier",
                    verifier_type="Tool",
                    tool_type="trust_verifier",
                    risk_level="low",
                    path_trust=memory.metadata.get("path_trust"),
                )

    def _contextual_action_support(self, agent_id: str) -> list[MemoryRecord]:
        contextual = []
        for record in self._records.values():
            if record.metadata.get("origin") != "agent_message":
                continue
            if record.owner_agent == agent_id:
                continue
            trust_result = self.trust_engine.compute_memory_trust(
                record,
                agent_id,
                None,
                self._source_catalog,
                self._records,
                self._can_read,
                self._agent_reliability_map(),
            )
            if trust_result.affected_ancestors or trust_result.permission_validity == 0.0:
                record.metadata.update(trust_result.to_dict())
                contextual.append(record)
        return contextual

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
