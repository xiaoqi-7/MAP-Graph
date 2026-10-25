from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.memory_baselines.base import MemoryRecord


UNTRUSTED_RISK_TAGS = {
    "poisoned",
    "untrusted",
    "withdrawn",
    "stale",
    "revoked",
    "dangerous_command",
    "risky_command",
    "privacy_violation",
    "credential",
}

PRIVATE_RISK_TAGS = {"private", "salary", "medical", "credential", "confidential"}

UNTRUSTED_TRUST_LABELS = {
    "poisoned",
    "untrusted",
    "withdrawn",
    "stale",
    "revoked",
    "risky",
}

PRIVATE_TRUST_LABELS = {"private", "credential", "confidential"}


@dataclass(frozen=True)
class TrustResult:
    trust: float
    source_trust: float
    path_integrity: float
    transformation_factor: float
    permission_validity: float
    verification_bonus: float
    agent_reliability: float
    transformation_loss: float
    affected_ancestors: list[str]
    contamination_reasons: dict[str, Any]
    untrusted_ancestors: list[str]
    private_ancestors: list[str]
    revoked_ancestors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_trust": self.trust,
            "source_trust": self.source_trust,
            "path_integrity": self.path_integrity,
            "transformation_factor": self.transformation_factor,
            "permission_validity": self.permission_validity,
            "verification_bonus": self.verification_bonus,
            "agent_reliability": self.agent_reliability,
            "transformation_loss": self.transformation_loss,
            "affected_ancestors": self.affected_ancestors,
            "contamination_reasons": self.contamination_reasons,
            "untrusted_ancestors": self.untrusted_ancestors,
            "private_ancestors": self.private_ancestors,
            "revoked_ancestors": self.revoked_ancestors,
        }


class MAPGraphTrustEngine:
    """Path-based trust propagation and action gate rules."""

    def __init__(
        self,
        low_trust_threshold: float = 0.6,
        enable_trust_propagation: bool = True,
        enable_action_gate: bool = True,
        enable_contamination_containment: bool = True,
    ) -> None:
        self.low_trust_threshold = low_trust_threshold
        self.enable_trust_propagation = enable_trust_propagation
        self.enable_action_gate = enable_action_gate
        self.enable_contamination_containment = enable_contamination_containment

    def threshold_for_action(self, action_type: str, risk_level: str) -> float:
        risk = str(risk_level or "").lower()
        action = str(action_type or "").lower()
        if action == "answer":
            return 0.30
        if risk == "low":
            return 0.60
        if risk == "high":
            return 0.85
        return 0.60

    def compute_memory_trust(
        self,
        memory: MemoryRecord,
        agent_id: str,
        action: dict | None,
        source_catalog: dict[str, dict[str, Any]],
        records: dict[str, MemoryRecord],
        permission_checker,
        agent_reliability: dict[str, float] | None = None,
    ) -> TrustResult:
        if not self.enable_trust_propagation:
            permission_validity = 1.0 if permission_checker(agent_id, memory) else 0.0
            return TrustResult(
                trust=permission_validity,
                source_trust=1.0,
                path_integrity=1.0,
                transformation_factor=1.0,
                permission_validity=permission_validity,
                verification_bonus=1.0,
                agent_reliability=1.0,
                transformation_loss=0.0,
                affected_ancestors=[],
                contamination_reasons={},
                untrusted_ancestors=[],
                private_ancestors=[],
                revoked_ancestors=[],
            )

        ancestors = self._ancestor_payloads(memory, source_catalog, records)
        source_trust_values = [float(item["payload"].get("trust", 1.0)) for item in ancestors if item["kind"] == "source"]
        source_trust = min(source_trust_values) if source_trust_values else 0.80

        untrusted: list[str] = []
        private: list[str] = []
        revoked: list[str] = []
        reasons: dict[str, Any] = {}
        verification_hits = 0

        for item in ancestors:
            ref = item["id"]
            payload = item["payload"]
            tags = set(payload.get("risk_tags") or [])
            label = str(payload.get("trust_label") or "").lower()
            visibility = str(payload.get("visibility") or payload.get("source_visibility") or "").lower()
            trust = float(payload.get("trust", 1.0) if payload.get("trust") is not None else 1.0)
            item_reasons: list[str] = []
            if payload.get("is_poisoned") or tags & UNTRUSTED_RISK_TAGS or label in UNTRUSTED_TRUST_LABELS:
                item_reasons.append("untrusted_or_poisoned")
                untrusted.append(ref)
            if payload.get("is_private") or tags & PRIVATE_RISK_TAGS or label in PRIVATE_TRUST_LABELS or visibility in {"private", "hr_only", "executor_only", "verifier_only"}:
                item_reasons.append("private_or_sensitive")
                private.append(ref)
            if payload.get("is_revoked") or visibility == "revoked" or label == "revoked":
                item_reasons.append("revoked_or_invalidated")
                revoked.append(ref)
            if trust < self.low_trust_threshold:
                item_reasons.append("low_source_trust")
                untrusted.append(ref)
            if "trusted" in tags or label == "trusted" or trust >= 0.9:
                verification_hits += 1
            if item_reasons:
                reasons[ref] = {
                    "reasons": sorted(set(item_reasons)),
                    "trust": payload.get("trust"),
                    "risk_tags": sorted(tags),
                    "visibility": payload.get("visibility"),
                    "trust_label": payload.get("trust_label"),
                }

        affected = sorted(set(untrusted + private + revoked))
        path_integrity = 1.0
        if untrusted:
            path_integrity *= 0.25
        if private:
            path_integrity *= 0.45
        if revoked:
            path_integrity *= 0.15

        transformation_loss = self._transformation_loss(memory, records)
        transformation_factor = max(0.35, 1.0 - transformation_loss)
        permission_validity = 1.0 if permission_checker(agent_id, memory) else 0.0
        verification_bonus = min(1.20, 1.0 + 0.05 * verification_hits)
        reliability = (agent_reliability or {}).get(memory.owner_agent, 0.95)

        trust = (
            source_trust
            * path_integrity
            * transformation_factor
            * permission_validity
            * verification_bonus
            * reliability
        )
        trust = max(0.0, min(1.0, trust))

        return TrustResult(
            trust=trust,
            source_trust=source_trust,
            path_integrity=path_integrity,
            transformation_factor=transformation_factor,
            permission_validity=permission_validity,
            verification_bonus=verification_bonus,
            agent_reliability=reliability,
            transformation_loss=transformation_loss,
            affected_ancestors=affected,
            contamination_reasons=reasons,
            untrusted_ancestors=sorted(set(untrusted)),
            private_ancestors=sorted(set(private)),
            revoked_ancestors=sorted(set(revoked)),
        )

    def gate_action(
        self,
        action: dict,
        task: dict,
        supporting_memories: list[MemoryRecord],
        trust_results: dict[str, TrustResult],
    ) -> dict[str, Any]:
        action_type = action.get("action_type") or action.get("type") or (task.get("expected_action") or {}).get("type") or "other"
        risk_level = action.get("risk_level") or action.get("risk") or (task.get("expected_action") or {}).get("risk_level") or "medium"
        action_id = action.get("action_id") or f"{task.get('task_id', 'task')}_{action.get('agent_id', 'agent')}_action"
        threshold = self.threshold_for_action(str(action_type), str(risk_level))
        supporting_ids = [memory.memory_id for memory in supporting_memories]
        relevant = [trust_results[memory.memory_id] for memory in supporting_memories if memory.memory_id in trust_results]
        min_trust = min((item.trust for item in relevant), default=0.0)
        max_trust = max((item.trust for item in relevant), default=0.0)
        high_risk = str(risk_level).lower() == "high"

        affected = [
            {
                "memory_id": memory.memory_id,
                **trust_results[memory.memory_id].to_dict(),
            }
            for memory in supporting_memories
            if memory.memory_id in trust_results and trust_results[memory.memory_id].affected_ancestors
        ]

        if not self.enable_action_gate:
            return self._decision(
                action,
                task,
                action_id,
                action_type,
                risk_level,
                supporting_ids,
                "ALLOW",
                threshold,
                min_trust,
                "MAP-Graph ablation allowed action because the action-time gate is disabled.",
                affected,
                max_trust=max_trust,
            )

        if self.enable_contamination_containment and high_risk and affected:
            return self._decision(
                action,
                task,
                action_id,
                action_type,
                risk_level,
                supporting_ids,
                "BLOCK",
                threshold,
                min_trust,
                "MAP-Graph hard rule blocked high-risk action because a supporting memory has an untrusted/private/revoked ancestor.",
                affected,
            )

        if not supporting_memories and str(risk_level).lower() in {"medium", "high"}:
            context_decision, context_reason = self._no_support_decision(task, str(risk_level))
            return self._decision(
                action,
                task,
                action_id,
                action_type,
                risk_level,
                supporting_ids,
                context_decision,
                threshold,
                min_trust,
                context_reason,
                affected,
                max_trust=max_trust,
            )

        private_only = any(item.private_ancestors for item in relevant)
        revoked = any(item.revoked_ancestors for item in relevant)
        untrusted = any(item.untrusted_ancestors for item in relevant)
        permission_denied = any(item.permission_validity == 0.0 for item in relevant)
        revoked_or_poisoned = revoked or untrusted
        if self.enable_contamination_containment and relevant and (private_only or revoked_or_poisoned or permission_denied):
            if high_risk or revoked:
                decision = "BLOCK"
                reason = "MAP-Graph blocked because at least one supporting memory has an untrusted/private/revoked ancestor or invalid permission path."
            elif private_only or permission_denied:
                decision = "REDACT"
                reason = "MAP-Graph requires redaction because at least one supporting memory depends on private data or an invalid permission path."
            else:
                decision = "REVERIFY"
                reason = "MAP-Graph requires re-verification because at least one supporting memory depends on untrusted ancestors."
            return self._decision(
                action,
                task,
                action_id,
                action_type,
                risk_level,
                supporting_ids,
                decision,
                threshold,
                min_trust,
                reason,
                affected,
                max_trust=max_trust,
            )
        if max_trust < threshold:
            if private_only:
                decision = "REDACT"
                reason = "MAP-Graph requires redaction because supporting memory depends on private or sensitive ancestors."
            elif revoked_or_poisoned:
                decision = "REVERIFY"
                reason = "MAP-Graph requires re-verification because path trust is below the action threshold."
            else:
                decision = "ASK_USER"
                reason = "MAP-Graph asks for user authorization because path trust is below the action threshold."
            return self._decision(
                action,
                task,
                action_id,
                action_type,
                risk_level,
                supporting_ids,
                decision,
                threshold,
                min_trust,
                reason,
                affected,
                max_trust=max_trust,
            )

        return self._decision(
            action,
            task,
            action_id,
            action_type,
            risk_level,
            supporting_ids,
            "ALLOW",
            threshold,
            min_trust,
            "MAP-Graph allowed action because all supporting memories satisfy permission and path-trust requirements.",
            affected,
            max_trust=max_trust,
        )

    def _no_support_decision(self, task: dict, risk_level: str) -> tuple[str, str]:
        return (
            "ASK_USER" if str(risk_level).lower() == "high" else "REVERIFY",
            "MAP-Graph escalated because no supporting memory survived retrieval and permission filtering; no task-level risk label is used as a substitute for provenance evidence.",
        )

    def _task_context_flags(self, task: dict) -> dict[str, bool]:
        flags = {"private": False, "revoked": False, "poisoned": False, "dangerous": False}
        for payload in list(task.get("source_documents", [])) + list(task.get("memory_events", [])):
            tags = set(payload.get("risk_tags") or [])
            label = str(payload.get("trust_label") or "").lower()
            visibility = str(payload.get("visibility") or "").lower()
            if payload.get("is_private") or visibility in {"private", "hr_only", "executor_only", "verifier_only"} or tags & PRIVATE_RISK_TAGS or label in PRIVATE_TRUST_LABELS:
                flags["private"] = True
            if payload.get("is_revoked") or visibility == "revoked" or "revoked" in tags or label == "revoked":
                flags["revoked"] = True
            if payload.get("is_poisoned") or "poisoned" in tags or label == "poisoned":
                flags["poisoned"] = True
            if tags & {"dangerous_command", "risky_command", "privacy_violation", "credential"} or label == "risky":
                flags["dangerous"] = True
            if "withdrawn" in tags or label == "withdrawn":
                flags["poisoned"] = True
        return flags

    def _ancestor_payloads(
        self,
        memory: MemoryRecord,
        source_catalog: dict[str, dict[str, Any]],
        records: dict[str, MemoryRecord],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[str] = set()

        def visit(ref: str) -> None:
            if not ref or ref in seen:
                return
            seen.add(ref)
            if ref in source_catalog:
                output.append({"kind": "source", "id": ref, "payload": source_catalog[ref]})
                return
            parent = records.get(ref)
            if parent is None:
                return
            output.append(
                {
                    "kind": "memory",
                    "id": ref,
                    "payload": {
                        **parent.metadata,
                        "trust_label": parent.trust_label,
                        "visibility": parent.visibility,
                        "owner_agent": parent.owner_agent,
                    },
                }
            )
            for parent_ref in parent.derived_from:
                visit(str(parent_ref))

        for ref in memory.derived_from:
            visit(str(ref))
        if memory.source_id:
            visit(str(memory.source_id))
        return output

    def _transformation_loss(self, memory: MemoryRecord, records: dict[str, MemoryRecord]) -> float:
        loss = 0.0
        seen: set[str] = set()

        def visit(record: MemoryRecord, depth: int) -> None:
            nonlocal loss
            if record.memory_id in seen:
                return
            seen.add(record.memory_id)
            origin = str(record.metadata.get("origin") or "")
            event_type = str(record.metadata.get("event_type") or "")
            if origin == "agent_message":
                loss += 0.08
            if event_type == "summarize_memory":
                loss += 0.10
            if depth > 0:
                loss += 0.03
            for ref in record.derived_from:
                parent = records.get(str(ref))
                if parent is not None:
                    visit(parent, depth + 1)

        visit(memory, 0)
        return min(0.65, loss)

    def _decision(
        self,
        action: dict,
        task: dict,
        action_id: str,
        action_type: str,
        risk_level: str,
        supporting_ids: list[str],
        decision: str,
        threshold: float,
        min_trust: float,
        reason: str,
        affected: list[dict[str, Any]],
        max_trust: float | None = None,
    ) -> dict[str, Any]:
        return {
            "action_id": action_id,
            "agent_id": action.get("agent_id", "unknown_agent"),
            "action_type": action_type,
            "risk_level": str(risk_level).lower(),
            "supporting_memory_ids": supporting_ids,
            "supporting_source_ids": action.get("supporting_source_ids", []),
            "decision": decision,
            "path_trust_min": round(min_trust, 6),
            "path_trust_max": round(max_trust if max_trust is not None else min_trust, 6),
            "required_trust_threshold": threshold,
            "affected_supporting_memories": affected,
            "reason": reason,
        }
