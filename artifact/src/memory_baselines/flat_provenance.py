from __future__ import annotations

import json
from typing import Any

from src.map_graph.trust import (
    PRIVATE_RISK_TAGS,
    PRIVATE_TRUST_LABELS,
    UNTRUSTED_RISK_TAGS,
    UNTRUSTED_TRUST_LABELS,
)
from src.memory_baselines.base import MemoryRecord
from src.memory_baselines.shared_vector import SharedVectorMemoryBackend, _safe_id


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


class FlatProvenanceMemoryBackend(SharedVectorMemoryBackend):
    """Provenance-metadata control without a provenance graph.

    The backend retains the same direct fields used by MAP-Graph—visibility,
    owner, permission scope, source trust, private/revoked/poisoned state, and
    action risk. Retrieval filtering and the action gate inspect only the
    current memory's own flat metadata. The backend never constructs a graph,
    traverses parent records, propagates path trust, or recursively follows
    ancestry.
    """

    method_name = "FlatProvenance"

    RESTRICTED_VISIBILITIES = {
        "private",
        "hr_only",
        "executor_only",
        "verifier_only",
        "reviewer_only",
        "isolated",
    }

    def __init__(self, top_k: int | None = None):
        super().__init__(top_k=top_k)
        self._task_agents: list[str] = []
        self._source_catalog: dict[str, dict[str, Any]] = {}
        self._current_action_risk = "medium"
        self._last_retrieval: dict[str, Any] = {}

    def reset_task(self, task_id: str) -> None:
        super().reset_task(task_id)
        self._task_agents = []
        self._source_catalog = {}
        self._current_action_risk = "medium"
        self._last_retrieval = {}

    def observe_sources(
        self,
        agent_id: str,
        sources: list[dict],
        task: dict,
        metadata: dict,
    ) -> list[MemoryRecord]:
        self._ensure_task_seeded(task)
        written: list[MemoryRecord] = []
        for source in sources:
            source_id = str(source.get("source_id") or "")
            if not source_id:
                continue
            self._source_catalog[source_id] = dict(source)
            source_key = (agent_id, source_id)
            if source_key in self._observed_source_keys:
                continue
            self._observed_source_keys.add(source_key)

            fields = self._fields_from_payload(
                source,
                default_owner=agent_id,
            )
            record = self._make_record(
                owner_agent=agent_id,
                content=str(source.get("content", "")),
                visibility=str(fields["visibility"]),
                source_id=source_id,
                derived_from=[source_id],
                trust_label=str(fields["trust_label"])
                if fields["trust_label"]
                else None,
                metadata={
                    **metadata,
                    "origin": "source_document",
                    **fields,
                    "direct_input_ids": [source_id],
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
        written: list[MemoryRecord] = []
        for output in tool_outputs:
            output_id = str(
                output.get("tool_output_id")
                or output.get("output_id")
                or output.get("id")
                or ""
            )
            if not output_id:
                continue
            output_key = (agent_id, output_id)
            if output_key in self._observed_tool_output_keys:
                continue
            self._observed_tool_output_keys.add(output_key)
            self._source_catalog[output_id] = dict(output)

            content = output.get("content")
            if content is None:
                content = output.get("output")
            if content is None:
                content = output.get("result")
            if content is None:
                content = json.dumps(output, ensure_ascii=False)

            fields = self._fields_from_payload(
                output,
                default_owner=agent_id,
            )
            record = self._make_record(
                owner_agent=agent_id,
                content=str(content),
                visibility=str(fields["visibility"]),
                source_id=None,
                derived_from=[output_id],
                trust_label=str(fields["trust_label"])
                if fields["trust_label"]
                else None,
                metadata={
                    **metadata,
                    "origin": "tool_output",
                    "tool_output_id": output_id,
                    **fields,
                    "direct_input_ids": [output_id],
                },
            )
            self._add_record(record)
            written.append(record)
        return written

    def retrieve(
        self,
        agent_id: str,
        query: str,
        task: dict,
    ) -> list[MemoryRecord]:
        self._ensure_task_seeded(task)
        if not self._records:
            self._last_retrieval = {
                "agent_id": agent_id,
                "candidate_memory_ids": [],
                "permission_denied_memory_ids": [],
                "selected_memory_ids": [],
            }
            return []
        if self._collection is None:
            raise RuntimeError(
                "Flat provenance collection has not been initialized. "
                "Call reset_task first."
            )

        query_embedding = self.embedder.embed(query)[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=len(self._records),
            include=["distances"],
        )
        ids = [str(value) for value in result.get("ids", [[]])[0]]
        distances = result.get("distances", [[]])[0]
        allowed: list[tuple[float, MemoryRecord]] = []
        denied: list[str] = []
        for memory_id, distance in zip(ids, distances):
            record = self._records.get(memory_id)
            if record is None:
                continue
            if not self._can_read_flat(agent_id, record):
                denied.append(memory_id)
                continue
            allowed.append((1.0 - float(distance), record))
        allowed.sort(key=lambda item: item[0], reverse=True)
        selected = [record for _, record in allowed[: self.top_k]]
        self._last_retrieval = {
            "agent_id": agent_id,
            "candidate_memory_ids": ids,
            "permission_denied_memory_ids": denied,
            "selected_memory_ids": [
                record.memory_id for record in selected
            ],
        }
        return selected

    def write(
        self,
        agent_id: str,
        content: str,
        metadata: dict,
    ) -> MemoryRecord | None:
        if not content or not content.strip():
            return None
        references = _unique(
            [
                str(value)
                for value in metadata.get("retrieved_memory_ids") or []
            ]
            + [
                str(value)
                for value in metadata.get("visible_source_ids") or []
            ]
        )
        fields = self._fields_from_direct_inputs(
            references,
            default_owner=agent_id,
        )
        record = self._make_record(
            owner_agent=agent_id,
            content=content,
            visibility=str(fields["visibility"]),
            source_id=None,
            derived_from=references,
            trust_label=str(fields["trust_label"])
            if fields["trust_label"]
            else None,
            metadata={
                **metadata,
                "origin": "agent_message",
                **fields,
                "direct_input_ids": references,
            },
        )
        self._add_record(record)
        return record

    def filter_conversation(
        self,
        agent_id: str,
        conversation: list[dict[str, Any]],
        task: dict,
    ) -> list[dict[str, Any]]:
        return [
            item
            for item in conversation
            if "permission_scope" not in item
            or str(agent_id)
            in {
                str(agent)
                for agent in item.get("permission_scope") or []
            }
        ]

    def apply_timeline_event(self, event: dict, task: dict) -> None:
        """Update only records directly tied to the revoked source."""
        if event.get("event_type") != "revoke_permission":
            return
        source_id = str(event.get("source_id") or "")
        if not source_id:
            return
        source = dict(self._source_catalog.get(source_id) or {})
        source.update(event)
        source["source_id"] = source_id
        source["visibility"] = "revoked"
        source["permission_scope"] = []
        source["is_revoked"] = True
        self._source_catalog[source_id] = source

        for record in self._records.values():
            direct_ids = {
                str(value)
                for value in record.metadata.get("direct_input_ids") or []
            }
            if record.source_id != source_id and source_id not in direct_ids:
                continue
            record.visibility = "revoked"
            record.metadata["visibility"] = "revoked"
            record.metadata["permission_scope"] = []
            record.metadata["is_revoked"] = True
            record.metadata["is_untrusted"] = True

    def decide_action(
        self,
        action: dict,
        task: dict,
        supporting_memories: list[MemoryRecord],
    ) -> dict:
        self._ensure_task_seeded(task)
        action_type = str(
            action.get("action_type")
            or action.get("type")
            or (task.get("expected_action") or {}).get("type")
            or "other"
        )
        risk_level = str(
            action.get("risk_level")
            or action.get("risk")
            or (task.get("expected_action") or {}).get("risk_level")
            or "medium"
        ).lower()
        agent_id = str(action.get("agent_id") or "unknown_agent")
        threshold = self._threshold_for_action(action_type, risk_level)
        summaries = [
            self._self_metadata_summary(agent_id, memory)
            for memory in supporting_memories
        ]
        trust_values = [
            float(item["flat_trust"]) for item in summaries
        ]
        min_trust = min(trust_values, default=0.0)
        max_trust = max(trust_values, default=0.0)

        decision = "ALLOW"
        reason = (
            "FlatProvenance allowed the action because supporting memories' "
            "own metadata satisfies the direct permission and trust checks."
        )
        high_risk = risk_level == "high"
        revoked = any(item["is_revoked"] for item in summaries)
        private = any(item["is_private"] for item in summaries)
        poisoned = any(
            item["is_poisoned"] or item["is_untrusted"]
            for item in summaries
        )
        permission_denied = any(
            not item["permission_allowed"] for item in summaries
        )

        if not supporting_memories and risk_level in {"medium", "high"}:
            decision = "ASK_USER" if high_risk else "REVERIFY"
            reason = (
                "FlatProvenance escalated because no supporting memory "
                "survived direct metadata filtering."
            )
        elif high_risk and (
            revoked
            or private
            or poisoned
            or permission_denied
            or max_trust < threshold
        ):
            decision = "BLOCK"
            reason = (
                "FlatProvenance blocked the high-risk action because a "
                "supporting memory's own metadata is private, revoked, "
                "poisoned, untrusted, or permission-invalid."
            )
        elif revoked:
            decision = "BLOCK"
            reason = (
                "FlatProvenance blocked the action because a supporting "
                "memory's own metadata is revoked."
            )
        elif private or permission_denied:
            decision = "REDACT"
            reason = (
                "FlatProvenance requires redaction because a supporting "
                "memory's own metadata is private or permission-invalid."
            )
        elif poisoned:
            decision = "REVERIFY"
            reason = (
                "FlatProvenance requires re-verification because a "
                "supporting memory's own metadata is poisoned or untrusted."
            )
        elif max_trust < threshold:
            decision = "ASK_USER"
            reason = (
                "FlatProvenance asks for authorization because direct source "
                "trust is below the action-risk threshold."
            )

        return {
            "action_id": action.get("action_id")
            or f"{task.get('task_id', 'task')}_{agent_id}_action",
            "agent_id": agent_id,
            "action_type": action_type,
            "risk_level": risk_level,
            "supporting_memory_ids": [
                memory.memory_id for memory in supporting_memories
            ],
            "supporting_source_ids": list(
                action.get("supporting_source_ids") or []
            ),
            "decision": decision,
            "flat_trust_min": round(min_trust, 6),
            "flat_trust_max": round(max_trust, 6),
            "required_trust_threshold": threshold,
            "direct_metadata_checks": summaries,
            "reason": reason,
        }

    def last_retrieval_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_retrieval)

    def _ensure_task_seeded(self, task: dict) -> None:
        task_id = str(
            task.get("task_id", getattr(self, "task_id", "unknown_task"))
        )
        if self._seeded_task_id == task_id:
            return
        self._task_agents = [
            str(agent) for agent in task.get("agent_roles") or []
        ]
        self._current_action_risk = str(
            (task.get("expected_action") or {}).get("risk_level")
            or "medium"
        ).lower()
        for source in task.get("source_documents") or []:
            source_id = str(source.get("source_id") or "")
            if source_id:
                self._source_catalog[source_id] = dict(source)

        for event in task.get(
            "initial_memory_events",
            task.get("memory_events", []),
        ):
            memory_id = str(event.get("memory_id") or "")
            content = str(event.get("content", ""))
            if not memory_id or not content.strip():
                continue
            owner = str(
                event.get("owner_agent")
                or event.get("agent_id")
                or "unknown_owner"
            )
            source_id = str(event.get("source_id") or "") or None
            references = _unique(
                [str(value) for value in event.get("derived_from") or []]
                + ([source_id] if source_id else [])
            )
            fields = self._fields_from_payload(
                event,
                default_owner=owner,
            )
            record = MemoryRecord(
                memory_id=_safe_id(memory_id),
                content=content,
                owner_agent=owner,
                visibility=str(fields["visibility"]),
                source_id=source_id,
                derived_from=references,
                trust_label=str(fields["trust_label"])
                if fields["trust_label"]
                else None,
                metadata={
                    "task_id": task_id,
                    "domain": task.get("domain"),
                    "origin": "seed_memory_event",
                    "event_id": event.get("event_id"),
                    "event_type": event.get("event_type"),
                    **fields,
                    "direct_input_ids": references,
                },
            )
            self._add_record(record)
        self._seeded_task_id = task_id

    def _fields_from_payload(
        self,
        payload: dict,
        default_owner: str,
    ) -> dict[str, Any]:
        visibility = str(
            payload.get("visibility")
            or payload.get("raw_visibility")
            or "shared"
        ).lower()
        owner = str(
            payload.get("owner")
            or payload.get("source_owner")
            or payload.get("owner_agent")
            or default_owner
        )
        if "permission_scope" in payload:
            scope = _unique(
                [str(value) for value in payload.get("permission_scope") or []]
            )
        elif visibility == "shared":
            scope = list(self._task_agents)
        else:
            scope = [owner]
        tags = {
            str(tag).lower() for tag in payload.get("risk_tags") or []
        }
        label = str(payload.get("trust_label") or "").lower()
        try:
            source_trust = float(
                payload.get(
                    "source_trust",
                    payload.get("trust", 1.0),
                )
            )
        except (TypeError, ValueError):
            source_trust = 1.0

        is_private = bool(payload.get("is_private")) or (
            visibility in self.RESTRICTED_VISIBILITIES
            or bool(tags & PRIVATE_RISK_TAGS)
            or label in PRIVATE_TRUST_LABELS
        )
        is_revoked = (
            bool(payload.get("is_revoked"))
            or visibility == "revoked"
            or "revoked" in tags
            or label == "revoked"
        )
        is_poisoned = (
            bool(payload.get("is_poisoned"))
            or "poisoned" in tags
            or label == "poisoned"
        )
        is_untrusted = (
            bool(payload.get("is_untrusted"))
            or bool(tags & UNTRUSTED_RISK_TAGS)
            or label in UNTRUSTED_TRUST_LABELS
            or source_trust < 0.6
        )
        if is_revoked:
            visibility = "revoked"
            scope = []
        return {
            "visibility": visibility,
            "owner": owner,
            "source_owner": owner,
            "permission_scope": scope,
            "source_trust": max(0.0, min(1.0, source_trust)),
            "risk_tags": sorted(tags),
            "trust_label": label or None,
            "is_private": is_private,
            "is_revoked": is_revoked,
            "is_poisoned": is_poisoned,
            "is_untrusted": is_untrusted,
            "action_risk": str(
                payload.get("action_risk")
                or payload.get("risk_level")
                or self._current_action_risk
            ).lower(),
        }

    def _fields_from_direct_inputs(
        self,
        references: list[str],
        default_owner: str,
    ) -> dict[str, Any]:
        payloads: list[dict[str, Any]] = []
        for reference in references:
            source = self._source_catalog.get(reference)
            if source is not None:
                payloads.append(
                    self._fields_from_payload(
                        source,
                        default_owner=default_owner,
                    )
                )
                continue
            parent = self._records.get(reference)
            if parent is None:
                continue
            # Deliberately inspect only the parent record's own stored fields.
            # Do not follow parent.derived_from or any ancestor identifiers.
            payloads.append(
                self._fields_from_payload(
                    {
                        **parent.metadata,
                        "visibility": parent.visibility,
                        "owner_agent": parent.owner_agent,
                        "trust_label": parent.trust_label,
                    },
                    default_owner=default_owner,
                )
            )

        if not payloads:
            return {
                "visibility": "shared",
                "owner": default_owner,
                "source_owner": default_owner,
                "permission_scope": list(self._task_agents),
                "source_trust": 0.8,
                "risk_tags": [],
                "trust_label": None,
                "is_private": False,
                "is_revoked": False,
                "is_poisoned": False,
                "is_untrusted": False,
                "action_risk": self._current_action_risk,
            }

        scopes = [
            set(str(value) for value in item["permission_scope"])
            for item in payloads
        ]
        allowed = set.intersection(*scopes) if scopes else set()
        if not allowed:
            allowed = {default_owner}
        revoked = any(bool(item["is_revoked"]) for item in payloads)
        private = any(bool(item["is_private"]) for item in payloads)
        visibility = (
            "revoked"
            if revoked
            else (
                "shared"
                if allowed >= set(self._task_agents) and not private
                else "private"
            )
        )
        tags = sorted(
            {
                str(tag)
                for item in payloads
                for tag in item["risk_tags"]
            }
        )
        return {
            "visibility": visibility,
            "owner": default_owner,
            "source_owner": default_owner,
            "permission_scope": [] if revoked else sorted(allowed),
            "source_trust": min(
                float(item["source_trust"]) for item in payloads
            ),
            "risk_tags": tags,
            "trust_label": next(
                (
                    item["trust_label"]
                    for item in payloads
                    if item["trust_label"]
                ),
                None,
            ),
            "is_private": private,
            "is_revoked": revoked,
            "is_poisoned": any(
                bool(item["is_poisoned"]) for item in payloads
            ),
            "is_untrusted": any(
                bool(item["is_untrusted"]) for item in payloads
            ),
            "action_risk": self._current_action_risk,
        }

    def _can_read_flat(
        self,
        agent_id: str,
        record: MemoryRecord,
    ) -> bool:
        if bool(record.metadata.get("is_revoked")):
            return False
        visibility = str(
            record.metadata.get("visibility")
            or record.visibility
            or "shared"
        ).lower()
        if visibility == "revoked":
            return False
        scope = {
            str(value)
            for value in record.metadata.get("permission_scope") or []
        }
        if not scope:
            scope = {str(record.owner_agent)}
        if visibility == "shared":
            return (
                str(agent_id) in scope
                or scope >= set(self._task_agents)
            )
        return (
            str(agent_id) in scope
            or str(agent_id) == str(record.owner_agent)
        )

    def _self_metadata_summary(
        self,
        agent_id: str,
        memory: MemoryRecord,
    ) -> dict[str, Any]:
        permission_allowed = self._can_read_flat(agent_id, memory)
        source_trust = float(memory.metadata.get("source_trust", 0.8))
        flat_trust = source_trust if permission_allowed else 0.0
        if bool(memory.metadata.get("is_poisoned")) or bool(
            memory.metadata.get("is_untrusted")
        ):
            flat_trust *= 0.25
        if bool(memory.metadata.get("is_private")):
            flat_trust *= 0.45
        if bool(memory.metadata.get("is_revoked")):
            flat_trust *= 0.15
        return {
            "memory_id": memory.memory_id,
            "visibility": memory.metadata.get(
                "visibility",
                memory.visibility,
            ),
            "owner": memory.metadata.get(
                "owner",
                memory.owner_agent,
            ),
            "source_trust": round(source_trust, 6),
            "is_private": bool(memory.metadata.get("is_private")),
            "is_revoked": bool(memory.metadata.get("is_revoked")),
            "is_poisoned": bool(memory.metadata.get("is_poisoned")),
            "is_untrusted": bool(memory.metadata.get("is_untrusted")),
            "action_risk": memory.metadata.get(
                "action_risk",
                self._current_action_risk,
            ),
            "permission_allowed": permission_allowed,
            "flat_trust": round(max(0.0, min(1.0, flat_trust)), 6),
        }

    @staticmethod
    def _threshold_for_action(
        action_type: str,
        risk_level: str,
    ) -> float:
        if str(action_type).lower() == "answer":
            return 0.30
        if str(risk_level).lower() == "high":
            return 0.85
        return 0.60
