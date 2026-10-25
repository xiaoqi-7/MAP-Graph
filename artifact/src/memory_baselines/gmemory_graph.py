from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any

import networkx as nx

from src.memory_baselines.base import BaseMemoryBackend, MemoryRecord
from src.utils.model_clients import EmbeddingClient, LLMClient, load_model_config


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class GMemoryGraphBackend(BaseMemoryBackend):
    """B3 G-Memory style graph memory baseline.

    This adapter follows G-Memory's three-layer design with a query graph,
    interaction graph, and insight graph. It reuses past collaboration
    trajectories as graph memories, but intentionally does not add MAP-Graph
    permission-aware retrieval, trust propagation, or action-time gates.
    """

    method_name = "B3"

    def __init__(
        self,
        top_k: int | None = None,
        similarity_threshold: float = 0.3,
        task_graph_edge_threshold: float = 0.7,
        hop: int = 1,
        max_graph_tasks: int | None = None,
        llm_augmentation: bool | None = None,
        insight_llm_augmentation: bool | None = None,
    ):
        config = load_model_config()
        self.top_k = top_k or int(config.get("retrieval", {}).get("top_k", 5))
        self.embedder = EmbeddingClient()
        self.similarity_threshold = similarity_threshold
        self.task_graph_edge_threshold = task_graph_edge_threshold
        self.hop = hop
        self.llm_augmentation = (
            llm_augmentation
            if llm_augmentation is not None
            else os.getenv("GMEMORY_STRICT_LLM", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        insight_llm_env = os.getenv("GMEMORY_INSIGHT_LLM")
        self.insight_llm_augmentation = (
            insight_llm_augmentation
            if insight_llm_augmentation is not None
            else (
                insight_llm_env.strip().lower() in {"1", "true", "yes", "on"}
                if insight_llm_env is not None
                else True
            )
        )
        self._llm_client: LLMClient | None = None
        self.start_insights_threshold = int(
            os.getenv("GMEMORY_START_INSIGHTS_THRESHOLD", "5")
        )
        self.rounds_per_insights = int(
            os.getenv("GMEMORY_ROUNDS_PER_INSIGHTS", "5")
        )
        self.insights_point_num = int(
            os.getenv("GMEMORY_INSIGHTS_POINT_NUM", "5")
        )
        configured_history = (
            max_graph_tasks
            if max_graph_tasks is not None
            else int(os.getenv("GMEMORY_MAX_GRAPH_TASKS", "512"))
        )
        self.max_graph_tasks = max(1, int(configured_history))
        self.max_similarity_edges = max(2, self.top_k * 4)

        self.query_graph = nx.Graph()
        self.interaction_graph = nx.DiGraph()
        self.insight_graph = nx.Graph()

        self._task_records: list[MemoryRecord] = []
        self._retrieved_records: dict[str, MemoryRecord] = {}
        self._current_sources: list[dict[str, Any]] = []
        self._current_tool_outputs: list[dict[str, Any]] = []
        self._current_memory_events: list[dict[str, Any]] = []
        self._current_messages: list[dict[str, Any]] = []
        self._observed_source_keys: set[tuple[str, str]] = set()
        self._observed_tool_output_keys: set[tuple[str, str]] = set()
        self._seeded_task_id: str | None = None
        self._counter = 0
        self._graph_task_order: list[str] = []
        # Run-scoped count used for G-Memory's periodic insight updates.  It
        # intentionally survives reset_task(), which only resets per-task
        # working state.  Using the bounded query-graph size here would stop
        # updates once max_graph_tasks is reached.
        self._completed_task_count = 0
        self._task_retrieval_cache: list[MemoryRecord] | None = None

    def reset_task(self, task_id: str) -> None:
        super().reset_task(task_id)
        self._task_records = []
        self._retrieved_records = {}
        self._current_sources = []
        self._current_tool_outputs = []
        self._current_memory_events = []
        self._current_messages = []
        self._observed_source_keys = set()
        self._observed_tool_output_keys = set()
        self._seeded_task_id = None
        self._counter = 0
        self._task_retrieval_cache = None

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
            self._current_sources.append(
                {
                    "agent_id": agent_id,
                    "source_id": source_id,
                    "content": str(source.get("content", "")),
                    "metadata": metadata,
                }
            )
            record = self._make_record(
                owner_agent=agent_id,
                content=str(source.get("content", "")),
                visibility="graph_interaction",
                source_id=None,
                derived_from=[],
                trust_label=None,
                metadata={**metadata, "origin": "source_observation"},
            )
            self._task_records.append(record)
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

            self._current_tool_outputs.append(
                {
                    "agent_id": agent_id,
                    "tool_output_id": str(output_id),
                    "content": str(content),
                    "metadata": metadata,
                }
            )
            record = self._make_record(
                owner_agent=agent_id,
                content=str(content),
                visibility="graph_interaction",
                source_id=None,
                derived_from=[],
                trust_label=None,
                metadata={
                    **metadata,
                    "origin": "tool_output_observation",
                    "tool_output_id": str(output_id),
                },
            )
            self._task_records.append(record)
            written.append(record)
        return written

    def retrieve(self, agent_id: str, query: str, task: dict) -> list[MemoryRecord]:
        self._ensure_task_seeded(task)
        if self.query_graph.number_of_nodes() == 0 and self.insight_graph.number_of_nodes() == 0:
            return []

        # Official G-Memory retrieves once from the incoming task, before the
        # MAS starts its role turns. Reuse the same task-level result for all
        # roles instead of re-querying with role-specific prompts.
        if self._task_retrieval_cache is not None:
            return [
                self._clone_graph_record_for_agent(record, agent_id)
                for record in self._task_retrieval_cache
            ]

        task_query = self._task_main(task)
        query_embedding = self.embedder.embed(task_query)[0]
        candidates: list[tuple[float, MemoryRecord]] = []
        candidates.extend(self._retrieve_from_query_graph(agent_id, query_embedding))
        candidates.extend(self._retrieve_from_insight_graph(agent_id, query_embedding))

        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = self._rerank_successful_candidates(
            candidates,
            task_query,
        )
        selected: list[MemoryRecord] = []
        seen: set[str] = set()

        def take(role: str, limit: int) -> None:
            for _, record in candidates:
                if len([item for item in selected if item.metadata.get("gmemory_retrieval_role") == role]) >= limit:
                    return
                if record.metadata.get("gmemory_retrieval_role") != role:
                    continue
                if record.memory_id in seen:
                    continue
                seen.add(record.memory_id)
                selected.append(record)
                self._retrieved_records[record.memory_id] = record

        take("successful_case", 2)
        take("failed_case", 1)
        take("insight", max(0, self.top_k - len(selected)))

        for _, record in candidates:
            if len(selected) >= self.top_k:
                break
            if record.memory_id in seen:
                continue
            seen.add(record.memory_id)
            selected.append(record)
            self._retrieved_records[record.memory_id] = record
        self._task_retrieval_cache = list(selected)
        return selected

    def write(self, agent_id: str, content: str, metadata: dict) -> MemoryRecord | None:
        if not content or not content.strip():
            return None
        self._current_messages.append(
            {
                "agent_id": agent_id,
                "content": content,
                "metadata": metadata,
            }
        )
        record = self._make_record(
            owner_agent=agent_id,
            content=content,
            visibility="graph_interaction",
            source_id=None,
            derived_from=[],
            trust_label=None,
            metadata={**metadata, "origin": "agent_message"},
        )
        self._task_records.append(record)
        return record

    def _clone_graph_record_for_agent(
        self,
        record: MemoryRecord,
        agent_id: str,
    ) -> MemoryRecord:
        return MemoryRecord(
            memory_id=record.memory_id,
            content=record.content,
            owner_agent=agent_id,
            visibility=record.visibility,
            source_id=record.source_id,
            derived_from=list(record.derived_from),
            trust_label=record.trust_label,
            metadata=dict(record.metadata),
        )

    def _rerank_successful_candidates(
        self,
        candidates: list[tuple[float, MemoryRecord]],
        query_task: str,
    ) -> list[tuple[float, MemoryRecord]]:
        """Apply G-Memory's generative relevance ranking to success cases."""
        if not self.llm_augmentation:
            return candidates
        success = [
            (score, record)
            for score, record in candidates
            if record.metadata.get("gmemory_retrieval_role") == "successful_case"
        ]
        if not success:
            return candidates
        reranked: dict[str, float] = {}
        # Official retrieve_memory first collects 2x the requested successful
        # top-k, then applies generative ranking.
        for similarity, record in success[:4]:
            response = self._llm().chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Score how relevant and helpful a successful "
                            "multi-agent trajectory is for an ongoing task."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Successful case:\n"
                            f"{record.content}\n\n"
                            "Ongoing task:\n"
                            f"{query_task}\n\n"
                            "Return only: Score: <integer from 1 to 10>"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=32,
            )
            match = re.search(r"\b(10|[1-9])\b", response)
            generative_score = int(match.group(1)) / 10.0 if match else 0.0
            reranked[record.memory_id] = generative_score
            record.metadata["gmemory_generative_relevance"] = generative_score

        return sorted(
            candidates,
            key=lambda item: (
                reranked.get(item[1].memory_id, -1.0),
                item[0],
            ),
            reverse=True,
        )

    def all_memories(self) -> list[MemoryRecord]:
        records = list(self._retrieved_records.values())
        records.extend(self._task_records)
        return records

    def drain_model_call_stats(self) -> list[dict[str, Any]]:
        if self._llm_client is None:
            return []
        return [
            {**stats, "memory_operation": "b3_gmemory_reasoning"}
            for stats in self._llm_client.drain_call_stats()
        ]

    def export_graph(self) -> dict[str, Any]:
        return {
            "task_id": getattr(self, "task_id", "unknown_task"),
            "method": self.method_name,
            "graph_type": "gmemory_adapter",
            "node_counts": {
                "query_graph": self.query_graph.number_of_nodes(),
                "interaction_graph": self.interaction_graph.number_of_nodes(),
                "insight_graph": self.insight_graph.number_of_nodes(),
            },
            "edge_counts": {
                "query_graph": self.query_graph.number_of_edges(),
                "interaction_graph": self.interaction_graph.number_of_edges(),
                "insight_graph": self.insight_graph.number_of_edges(),
            },
            "bounded_history": {
                "max_graph_tasks": self.max_graph_tasks,
                "retained_graph_tasks": len(self._graph_task_order),
                "completed_graph_tasks": self._completed_task_count,
                "max_similarity_edges_per_new_node": self.max_similarity_edges,
            },
            "paper_alignment": {
                "official_repository": "https://github.com/bingreeky/GMemory",
                "official_commit": "7b581c5",
                "trajectory_sparsification": (
                    "llm" if self.llm_augmentation else "deterministic"
                ),
                "insight_update": (
                    "llm"
                    if self.insight_llm_augmentation
                    else "deterministic"
                ),
                "task_graph_edge_threshold": self.task_graph_edge_threshold,
                "retrieval_threshold": self.similarity_threshold,
                "hop": self.hop,
                "start_insights_threshold": self.start_insights_threshold,
                "rounds_per_insights": self.rounds_per_insights,
                "insights_point_num": self.insights_point_num,
            },
            # The insight layer is capped at ten rules.  Persisting this small
            # state makes --skip-existing resume exact without replaying paid
            # LLM insight-generation calls.
            "insight_state": [
                {
                    "node_id": str(node_id),
                    "task_id": data.get("task_id"),
                    "insight": str(data.get("insight") or ""),
                    "domain": data.get("domain"),
                    "score": int(data.get("score", 0)),
                    "positive_correlation_tasks": list(
                        data.get("positive_correlation_tasks") or []
                    ),
                    "negative_correlation_tasks": list(
                        data.get("negative_correlation_tasks") or []
                    ),
                }
                for node_id, data in self.insight_graph.nodes(data=True)
                if data.get("node_type") == "insight"
            ],
        }

    def restore_insight_state(self, state: list[dict[str, Any]]) -> None:
        """Restore the bounded insight layer saved by export_graph()."""
        self.insight_graph = nx.Graph()
        for item in state:
            insight = str(item.get("insight") or "").strip()
            node_id = str(item.get("node_id") or "").strip()
            if not insight or not node_id:
                continue
            self.insight_graph.add_node(
                node_id,
                node_type="insight",
                task_id=item.get("task_id"),
                insight=insight,
                embedding=self.embedder.embed(insight)[0],
                domain=item.get("domain"),
                score=int(item.get("score", 0)),
                positive_correlation_tasks=list(
                    item.get("positive_correlation_tasks") or []
                ),
                negative_correlation_tasks=list(
                    item.get("negative_correlation_tasks") or []
                ),
            )

        insight_nodes = list(self.insight_graph.nodes(data=True))
        for index, (node_id, data) in enumerate(insight_nodes):
            candidates: list[tuple[float, str]] = []
            for other_id, other_data in insight_nodes[index + 1 :]:
                score = _cosine(
                    data.get("embedding", []),
                    other_data.get("embedding", []),
                )
                if score >= self.similarity_threshold:
                    candidates.append((score, str(other_id)))
            for score, other_id in sorted(candidates, reverse=True)[
                : self.max_similarity_edges
            ]:
                self.insight_graph.add_edge(
                    node_id,
                    other_id,
                    weight=score,
                    edge_type="insight_similarity",
                )

    def finalize_task(
        self,
        task: dict,
        events: list[dict],
        decisions: list[dict],
        conversation: list[dict[str, str]],
    ) -> None:
        self._ensure_task_seeded(task)
        task_id = task.get("task_id", getattr(self, "task_id", "unknown_task"))
        task_main = self._task_main(task)
        task_description = self._task_description(task)
        trajectory = self._interaction_trace_to_gmemory_trajectory(task, events, decisions, conversation)
        label = self._is_successful(decisions)
        key_steps = self._extract_key_steps(
            trajectory,
            task_description=task_description,
        )
        fail_reason = None if label else self._detect_failure_reason(task, decisions)

        task_node_id = _safe_id(f"task:{task_id}")
        task_embedding = self.embedder.embed(task_main)[0]
        mas_message = {
            "task_main": task_main,
            "task_description": task_description,
            "task_trajectory": trajectory,
            "label": label,
            "extra_fields": {
                "key_steps": key_steps,
                "fail_reason": fail_reason,
            },
            "state_chain": self._state_chain_compatible(events),
        }
        self.query_graph.add_node(
            task_node_id,
            task_id=task_id,
            task_main=task_main,
            task_description=task_description,
            trajectory=trajectory,
            label=label,
            key_steps=key_steps,
            fail_reason=fail_reason,
            mas_message=mas_message,
            official_gmemory_adapter=True,
            embedding=task_embedding,
            domain=task.get("domain"),
        )
        self._connect_query_graph(task_node_id, task_embedding)
        self._add_interaction_graph(task_node_id, task, events, decisions, trajectory)
        self._completed_task_count += 1
        self._add_insight_graph(task_node_id, task, decisions, trajectory)
        self._graph_task_order.append(str(task_id))
        self._prune_graph_history()

    def _ensure_task_seeded(self, task: dict) -> None:
        task_id = task.get("task_id", getattr(self, "task_id", "unknown_task"))
        if self._seeded_task_id == task_id:
            return

        for event in task.get("initial_memory_events", task.get("memory_events", [])):
            content = str(event.get("content", ""))
            if not content.strip():
                continue
            owner = str(event.get("owner_agent") or event.get("agent_id") or "unknown_owner")
            self._current_memory_events.append(
                {
                    "agent_id": str(event.get("agent_id") or owner),
                    "owner_agent": owner,
                    "memory_id": str(event.get("memory_id")),
                    "event_id": event.get("event_id"),
                    "event_type": event.get("event_type"),
                    "content": content,
                    "source_id": event.get("source_id"),
                    "visibility": event.get("visibility"),
                    "permission_scope": event.get("permission_scope"),
                }
            )
            record = MemoryRecord(
                memory_id=_safe_id(str(event.get("memory_id") or f"seed_{task_id}_{owner}_{len(self._task_records)+1}")),
                content=content,
                owner_agent=owner,
                visibility="graph_interaction",
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
                    "note": "B3 converts seeded memory events into the G-Memory interaction trajectory without MAP-Graph gates.",
                },
            )
            self._task_records.append(record)

        self._seeded_task_id = task_id

    def _retrieve_from_query_graph(self, agent_id: str, query_embedding: list[float]) -> list[tuple[float, MemoryRecord]]:
        scored_tasks = []
        for node_id, data in self.query_graph.nodes(data=True):
            score = _cosine(query_embedding, data.get("embedding", []))
            if score >= self.similarity_threshold:
                scored_tasks.append((score, node_id, data))
        scored_tasks.sort(key=lambda item: item[0], reverse=True)

        expanded: dict[str, float] = {}
        for score, node_id, _ in scored_tasks[: self.top_k]:
            expanded[node_id] = max(expanded.get(node_id, 0.0), score)
            if self.hop > 0:
                lengths = nx.single_source_shortest_path_length(self.query_graph, node_id, cutoff=self.hop)
                for neighbor_id, distance in lengths.items():
                    neighbor_data = self.query_graph.nodes[neighbor_id]
                    neighbor_score = _cosine(query_embedding, neighbor_data.get("embedding", []))
                    expanded[neighbor_id] = max(expanded.get(neighbor_id, 0.0), neighbor_score / (distance + 1))

        records = []
        for node_id, score in expanded.items():
            data = self.query_graph.nodes[node_id]
            record = self._graph_record(
                graph_type="query_graph",
                owner_agent=agent_id,
                node_id=node_id,
                content=(
                    "Prior collaboration task\n"
                    f"task_summary: {str(data.get('task_main') or '')[:300]}\n"
                    f"runtime_outcome: {'completed' if data.get('label') else 'incomplete'}\n"
                    f"key_steps: {'; '.join(str(item) for item in (data.get('key_steps') or [])[:4])[:500]}"
                ),
                score=score,
                related_task_id=data.get("task_id"),
                label=data.get("label"),
                retrieval_role="successful_case" if data.get("label") else "failed_case",
                mas_message=data.get("mas_message"),
            )
            records.append((score, record))
        return records

    def _retrieve_from_insight_graph(self, agent_id: str, query_embedding: list[float]) -> list[tuple[float, MemoryRecord]]:
        records = []
        for node_id, data in self.insight_graph.nodes(data=True):
            if data.get("node_type") != "insight":
                continue
            score = _cosine(query_embedding, data.get("embedding", []))
            if score < self.similarity_threshold:
                continue
            record = self._graph_record(
                graph_type="insight_graph",
                owner_agent=agent_id,
                node_id=node_id,
                content=(
                    "Prior collaboration insight\n"
                    f"insight: {str(data.get('insight') or '')[:400]}"
                ),
                score=score,
                related_task_id=data.get("task_id"),
                label=None,
                retrieval_role="insight",
                mas_message=None,
            )
            records.append((score, record))
        return records

    def _connect_query_graph(self, new_node_id: str, new_embedding: list[float]) -> None:
        scored: list[tuple[float, str]] = []
        for node_id, data in list(self.query_graph.nodes(data=True)):
            if node_id == new_node_id:
                continue
            score = _cosine(new_embedding, data.get("embedding", []))
            if score >= self.task_graph_edge_threshold:
                scored.append((score, node_id))
        for score, node_id in sorted(scored, reverse=True)[: self.max_similarity_edges]:
            self.query_graph.add_edge(new_node_id, node_id, weight=score, edge_type="task_similarity")

    def _add_interaction_graph(
        self,
        task_node_id: str,
        task: dict,
        events: list[dict],
        decisions: list[dict],
        trajectory: str,
    ) -> None:
        self.interaction_graph.add_node(
            task_node_id,
            node_type="task",
            task_id=task.get("task_id"),
            domain=task.get("domain"),
            trajectory=trajectory,
        )
        previous_node = task_node_id
        for index, memory_event in enumerate(self._current_memory_events, start=1):
            node_id = _safe_id(f"interaction:{task.get('task_id')}:seed_memory:{index}:{memory_event.get('memory_id')}")
            self.interaction_graph.add_node(
                node_id,
                node_type="seed_memory_event",
                task_id=task.get("task_id"),
                agent_id=memory_event.get("agent_id"),
                owner_agent=memory_event.get("owner_agent"),
                memory_id=memory_event.get("memory_id"),
                event_type=memory_event.get("event_type"),
                content=memory_event.get("content", ""),
                source_id=memory_event.get("source_id"),
            )
            self.interaction_graph.add_edge(previous_node, node_id, edge_type="seed_memory")
            previous_node = node_id
        for event in events:
            if event.get("event_type") != "message":
                continue
            step = event.get("step")
            agent_id = event.get("agent_id")
            node_id = _safe_id(f"interaction:{task.get('task_id')}:{step}:{agent_id}")
            self.interaction_graph.add_node(
                node_id,
                node_type="agent_message",
                task_id=task.get("task_id"),
                step=step,
                agent_id=agent_id,
                content=event.get("content", ""),
                retrieved_memory_ids=event.get("retrieved_memory_ids", []),
                visible_source_ids=event.get("visible_source_ids", []),
            )
            self.interaction_graph.add_edge(previous_node, node_id, edge_type="temporal")
            previous_node = node_id
        for decision in decisions:
            node_id = _safe_id(f"interaction:{task.get('task_id')}:decision:{decision.get('agent_id')}")
            self.interaction_graph.add_node(
                node_id,
                node_type="decision",
                task_id=task.get("task_id"),
                agent_id=decision.get("agent_id"),
                decision=decision.get("decision"),
            )
            self.interaction_graph.add_edge(previous_node, node_id, edge_type="decision")

    def _add_insight_graph(self, task_node_id: str, task: dict, decisions: list[dict], trajectory: str) -> None:
        insights = self._derive_insights(task, decisions, trajectory)
        for index, insight in enumerate(insights, start=1):
            existing_id = self._matching_insight_node(insight)
            if existing_id is not None:
                data = self.insight_graph.nodes[existing_id]
                data["score"] = int(data.get("score", 2)) + 1
                correlated = list(data.get("positive_correlation_tasks") or [])
                correlated.append(str(task.get("task_id")))
                data["positive_correlation_tasks"] = list(dict.fromkeys(correlated))
                self.insight_graph.add_edge(
                    existing_id,
                    task_node_id,
                    edge_type="positive_correlation",
                )
                continue

            node_id = _safe_id(f"insight:{task.get('task_id')}:{index}")
            embedding = self.embedder.embed(insight)[0]
            self.insight_graph.add_node(
                node_id,
                node_type="insight",
                task_id=task.get("task_id"),
                insight=insight,
                embedding=embedding,
                domain=task.get("domain"),
                score=2,
                positive_correlation_tasks=[str(task.get("task_id"))],
                negative_correlation_tasks=[],
            )
            self.insight_graph.add_edge(node_id, task_node_id, edge_type="derived_from_task")

            similarity_candidates: list[tuple[float, str]] = []
            for other_id, data in list(self.insight_graph.nodes(data=True)):
                if other_id == node_id or data.get("node_type") != "insight":
                    continue
                score = _cosine(embedding, data.get("embedding", []))
                if score >= self.similarity_threshold:
                    similarity_candidates.append((score, other_id))
            for score, other_id in sorted(similarity_candidates, reverse=True)[: self.max_similarity_edges]:
                self.insight_graph.add_edge(node_id, other_id, weight=score, edge_type="insight_similarity")

        # Match the official manager's bounded rule list and remove rules whose
        # feedback score is no longer positive.
        insight_nodes = [
            (node_id, data)
            for node_id, data in self.insight_graph.nodes(data=True)
            if data.get("node_type") == "insight"
        ]
        stale = [
            node_id for node_id, data in insight_nodes
            if int(data.get("score", 0)) <= 0
        ]
        self.insight_graph.remove_nodes_from(stale)
        remaining = [
            (node_id, data)
            for node_id, data in self.insight_graph.nodes(data=True)
            if data.get("node_type") == "insight"
        ]
        if len(remaining) > 10:
            remove = sorted(
                remaining,
                key=lambda item: int(item[1].get("score", 0)),
            )[: len(remaining) - 10]
            self.insight_graph.remove_nodes_from([node_id for node_id, _ in remove])

    def _prune_graph_history(self) -> None:
        """Keep recent trajectories while preserving the bounded insight layer."""
        self._graph_task_order = self._graph_task_order[-self.max_graph_tasks :]
        keep = set(self._graph_task_order)
        for graph in (self.query_graph, self.interaction_graph):
            graph.remove_nodes_from(
                [
                    node_id
                    for node_id, data in graph.nodes(data=True)
                    if str(data.get("task_id") or "") not in keep
                ]
            )

        # Insight rules are long-term abstractions rather than task-history
        # nodes.  Keep them across trajectory pruning; _add_insight_graph()
        # already bounds the rule set to ten nodes.  Only task anchor nodes in
        # the insight graph are pruned with the trajectory window.
        self.insight_graph.remove_nodes_from(
            [
                node_id
                for node_id, data in self.insight_graph.nodes(data=True)
                if data.get("node_type") != "insight"
                and str(data.get("task_id") or "") not in keep
            ]
        )

    def _derive_insights(self, task: dict, decisions: list[dict], trajectory: str) -> list[str]:
        if self._completed_task_count < self.start_insights_threshold:
            return []
        if self._completed_task_count % self.rounds_per_insights != 0:
            return []

        recent_ids = self._graph_task_order[-self.insights_point_num :]
        histories: list[str] = []
        for task_id in recent_ids:
            node_id = _safe_id(f"task:{task_id}")
            if not self.query_graph.has_node(node_id):
                continue
            data = self.query_graph.nodes[node_id]
            histories.append(
                f"Task: {data.get('task_main')}\n"
                f"Trajectory: {str(data.get('trajectory') or '')[:2400]}"
            )
        histories.append(
            f"Task: {self._task_main(task)}\nTrajectory: {trajectory[:2400]}"
        )

        existing_rules = [
            str(data.get("insight"))
            for _, data in self.insight_graph.nodes(data=True)
            if data.get("node_type") == "insight"
        ]
        if self.insight_llm_augmentation:
            response = self._llm().chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Derive concise, general multi-agent collaboration "
                            "rules from task trajectories. Rules must be grounded "
                            "in the trajectories and useful across tasks."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Recent trajectories:\n"
                            f"{chr(10).join(histories)}\n\n"
                            "Existing rules:\n"
                            f"{chr(10).join(existing_rules) or '(none)'}\n\n"
                            "Return at most four numbered rules. Each rule must "
                            "be one sentence and must not mention task IDs."
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=384,
            )
            parsed = [
                item.strip()
                for item in re.findall(
                    r"(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\Z)",
                    response,
                    flags=re.DOTALL,
                )
                if item.strip()
            ]
            if parsed:
                return parsed[:4]

        roles = task.get("agent_roles") or task.get("agents") or []
        return [
            (
                "Assign evidence gathering, independent checking, and final "
                "action proposal to explicit roles so the collaboration path "
                "remains auditable."
            ),
            (
                "Ground the final action in cited sources and retrieved "
                "trajectory evidence because unsupported coordination messages "
                "are difficult to verify."
            ),
            (
                f"Use role-specialized review across {len(roles)} participants "
                "before committing a consequential action."
            ),
        ]

    def _matching_insight_node(self, insight: str) -> str | None:
        normalized = re.sub(r"\s+", " ", insight).strip().lower()
        for node_id, data in self.insight_graph.nodes(data=True):
            if data.get("node_type") != "insight":
                continue
            existing = re.sub(
                r"\s+",
                " ",
                str(data.get("insight") or ""),
            ).strip().lower()
            if existing == normalized:
                return str(node_id)
        return None

    def _interaction_trace_to_gmemory_trajectory(
        self,
        task: dict,
        events: list[dict],
        decisions: list[dict],
        conversation: list[dict[str, str]],
    ) -> str:
        lines = [
            "### G-Memory interaction trajectory",
            f"Task: {self._task_main(task)}",
            f"Domain: {task.get('domain')}",
        ]
        for memory_event in self._current_memory_events:
            content = str(memory_event.get("content", "")).replace("\n", " ")[:900]
            lines.append(
                "> memory_event: {event_type}; agent={agent}; owner={owner}; memory_id={memory_id}; source_id={source_id}".format(
                    event_type=memory_event.get("event_type"),
                    agent=memory_event.get("agent_id"),
                    owner=memory_event.get("owner_agent"),
                    memory_id=memory_event.get("memory_id"),
                    source_id=memory_event.get("source_id"),
                )
            )
            lines.append(f"observation: {content}")
        for event in events:
            if event.get("event_type") == "message":
                content = str(event.get("content", "")).replace("\n", " ")[:900]
                lines.append(
                    "> action: {agent} observes sources={sources} retrieves memories={memories}".format(
                        agent=event.get("agent_id"),
                        sources=event.get("visible_source_ids", []),
                        memories=event.get("retrieved_memory_ids", []),
                    )
                )
                lines.append(f"observation: {content}")
            elif event.get("event_type") == "proposed_action":
                lines.append(f"> action: {event.get('agent_id')} proposes final action")
                lines.append(
                    "observation: decision={decision}; supporting_memories={memories}".format(
                        decision=event.get("decision"),
                        memories=event.get("supporting_memory_ids", []),
                    )
                )
        for decision in decisions:
            lines.append(
                "> final_decision: actual={actual}; agent={agent}".format(
                    actual=decision.get("decision"),
                    agent=decision.get("agent_id"),
                )
            )
        return "\n".join(lines)

    def _task_main(self, task: dict) -> str:
        return str(task.get("task") or task.get("instruction") or task.get("objective") or task.get("task_id"))

    def _task_description(self, task: dict) -> str:
        return json.dumps(
            {
                "task_id": task.get("task_id"),
                "domain": task.get("domain"),
                "expected_action": task.get("expected_action"),
            },
            ensure_ascii=False,
        )

    def _is_successful(self, decisions: list[dict]) -> bool:
        if not decisions:
            return False
        return all(decision.get("decision") not in {None, "NO_DECISION"} for decision in decisions)

    def _extract_key_steps(
        self,
        trajectory: str,
        task_description: str = "",
    ) -> list[str]:
        if self.llm_augmentation:
            response = self._llm().chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract the critical steps from a successful "
                            "multi-agent trajectory. Do not add steps that are "
                            "not present. Remove failed or irrelevant steps."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Task:\n{task_description}\n\n"
                            f"Trajectory:\n{trajectory}\n\n"
                            "Return at most eight concise numbered key steps."
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=384,
            )
            parsed = [
                item.strip()
                for item in re.findall(
                    r"(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\Z)",
                    response,
                    flags=re.DOTALL,
                )
                if item.strip()
            ]
            if parsed:
                return parsed[:8]

        steps = []
        for line in trajectory.splitlines():
            line = line.strip()
            if not line.startswith(">"):
                continue
            if any(token in line for token in ("memory_event", "action", "final_decision")):
                steps.append(line[:240])
            if len(steps) >= 8:
                break
        return steps

    def _detect_failure_reason(self, task: dict, decisions: list[dict]) -> str | None:
        if not decisions:
            if self.llm_augmentation:
                return self._llm().chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Identify the most likely reason a multi-agent "
                                "trajectory failed to produce a final action."
                            ),
                        },
                        {
                            "role": "user",
                            "content": self._task_description(task),
                        },
                    ],
                    temperature=0.1,
                    max_tokens=128,
                )
            return "No action decision was produced for this trajectory."
        return None

    def _llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def _state_chain_compatible(self, events: list[dict]) -> list[dict[str, Any]]:
        states = []
        for event in events:
            if event.get("event_type") != "message":
                continue
            states.append(
                {
                    "graph": {
                        "action": f"{event.get('agent_id')} message",
                        "observation": str(event.get("content", ""))[:1000],
                        "reward": 1,
                    },
                    "nodes": [
                        {
                            "id": f"{event.get('agent_id')}-{event.get('step')}",
                            "agent_name": event.get("agent_id"),
                            "message": event.get("content"),
                            "extra_fields": {
                                "retrieved_memory_ids": event.get("retrieved_memory_ids", []),
                                "visible_source_ids": event.get("visible_source_ids", []),
                            },
                        }
                    ],
                    "edges": [],
                }
            )
        return states

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
        prefix = str(metadata.get("origin") or "graph_memory")
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

    def _graph_record(
        self,
        graph_type: str,
        owner_agent: str,
        node_id: str,
        content: str,
        score: float,
        related_task_id: str | None,
        label: bool | None = None,
        retrieval_role: str | None = None,
        mas_message: dict | None = None,
    ) -> MemoryRecord:
        prefix = "gmq" if graph_type == "query_graph" else "gmi"
        memory_id = f"{prefix}_{hashlib.sha1(str(node_id).encode('utf-8')).hexdigest()[:12]}"
        return MemoryRecord(
            memory_id=memory_id,
            content=content,
            owner_agent=owner_agent,
            visibility="gmemory_graph",
            source_id=None,
            derived_from=[node_id],
            trust_label=None,
            metadata={
                "origin": graph_type,
                "graph_type": graph_type,
                "graph_node_id": node_id,
                "related_task_id": related_task_id,
                "similarity": round(score, 6),
                "label": label,
                "gmemory_retrieval_role": retrieval_role,
                "mas_message": mas_message,
                "official_gmemory_concepts": {
                    "task_layer": graph_type == "query_graph",
                    "insights_layer": graph_type == "insight_graph",
                    "interaction_graph": True,
                    "successful_topk": 2,
                    "failed_topk": 1,
                    "hop": self.hop,
                    "threshold": self.similarity_threshold,
                },
            },
        )
