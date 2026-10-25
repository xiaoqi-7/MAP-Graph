from __future__ import annotations

import hashlib
from typing import Any

import networkx as nx


class MAPGraphBuilder:
    """Execution-aware heterogeneous provenance graph for MAP-Graph."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.task_id = "unknown_task"

    def reset(self, task_id: str) -> None:
        self.task_id = task_id
        self.graph = nx.MultiDiGraph()

    def register_task(self, task: dict) -> None:
        for agent_id in task.get("agent_roles", []):
            self.add_agent(str(agent_id))
        for source in task.get("source_documents", []):
            self.add_resource(source)

    def add_user(self, user_id: str, role: str | None = None, permission_profile: dict | None = None) -> str:
        node_id = self.user_node(user_id)
        self.graph.add_node(
            node_id,
            node_type="User",
            user_id=user_id,
            role=role,
            permission_profile=permission_profile or {},
        )
        return node_id

    def add_agent(self, agent_id: str, role: str | None = None, reliability_score: float = 0.95) -> str:
        node_id = self.agent_node(agent_id)
        self.graph.add_node(
            node_id,
            node_type="Agent",
            agent_id=agent_id,
            role=role or agent_id,
            reliability_score=reliability_score,
        )
        return node_id

    def add_tool(self, tool_id: str, tool_type: str | None = None, risk_level: str | None = None) -> str:
        node_id = self.tool_node(tool_id)
        self.graph.add_node(
            node_id,
            node_type="Tool",
            tool_id=tool_id,
            tool_type=tool_type,
            risk_level=risk_level,
        )
        return node_id

    def add_resource(self, resource: dict) -> str:
        resource_id = str(resource.get("source_id") or resource.get("resource_id") or "unknown_resource")
        node_id = self.resource_node(resource_id)
        owner = str(resource.get("owner") or resource.get("owner_agent") or "unknown_owner")
        self.add_user(owner, role="resource_owner")
        self.graph.add_node(
            node_id,
            node_type="Resource",
            resource_id=resource_id,
            source_trust=float(resource.get("trust", 1.0) if resource.get("trust") is not None else 1.0),
            visibility=resource.get("visibility", "shared"),
            owner=owner,
            permission_scope=list(resource.get("permission_scope") or []),
            risk_tags=list(resource.get("risk_tags") or []),
            is_poisoned=bool(resource.get("is_poisoned")),
            is_private=bool(resource.get("is_private")),
            is_revoked=bool(resource.get("is_revoked") or resource.get("visibility") == "revoked"),
        )
        self.graph.add_edge(node_id, self.user_node(owner), edge_type="authorized_by")
        return node_id

    def add_observed_from(self, agent_id: str, source_id: str, **attrs: Any) -> None:
        self.add_agent(agent_id)
        self.graph.add_edge(
            self.agent_node(agent_id),
            self.resource_node(source_id),
            edge_type="observed_from",
            **attrs,
        )

    def add_message(
        self,
        message_id: str,
        sender: str,
        receivers: list[str],
        content: str,
        derived_from: list[str] | None = None,
        **attrs: Any,
    ) -> str:
        node_id = self.message_node(message_id)
        self.add_agent(sender)
        self.graph.add_node(
            node_id,
            node_type="Message",
            message_id=message_id,
            sender=sender,
            receiver=list(receivers),
            content_hash=self._content_hash(content),
            **attrs,
        )
        for receiver in receivers:
            self.add_agent(receiver)
            self.graph.add_edge(node_id, self.agent_node(receiver), edge_type="forwarded_to")
        for ref in derived_from or []:
            self.graph.add_edge(node_id, self.ref_node(ref), edge_type="derived_from")
        return node_id

    def add_memory(
        self,
        memory_id: str,
        content: str,
        owner_agent: str,
        trust_label: str | None,
        visibility: str,
        derived_from: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        node_id = self.memory_node(memory_id)
        self.add_agent(owner_agent)
        self.add_user(owner_agent, role="memory_owner")
        self.graph.add_node(
            node_id,
            node_type="Memory",
            memory_id=memory_id,
            content_hash=self._content_hash(content),
            owner_agent=owner_agent,
            trust_label=trust_label,
            visibility=visibility,
            affected=bool((metadata or {}).get("affected")),
            affected_ancestors=list((metadata or {}).get("affected_ancestors") or []),
            trust_score=(metadata or {}).get("path_trust"),
            permission_scope=list((metadata or {}).get("permission_scope") or []),
            origin=(metadata or {}).get("origin"),
        )
        self.graph.add_edge(node_id, self.agent_node(owner_agent), edge_type="written_by")
        self.graph.add_edge(node_id, self.user_node(owner_agent), edge_type="authorized_by")
        for ref in derived_from:
            edge_type = "summarized_from" if (metadata or {}).get("event_type") == "summarize_memory" else "derived_from"
            self.graph.add_edge(node_id, self.ref_node(ref), edge_type=edge_type)
        return node_id

    def add_memory_read(self, memory_id: str, agent_id: str, **attrs: Any) -> None:
        self.add_agent(agent_id)
        self.graph.add_edge(
            self.memory_node(memory_id),
            self.agent_node(agent_id),
            edge_type="read_by",
            **attrs,
        )

    def add_verified_by(self, node_ref: str, verifier_ref: str, verifier_type: str = "Resource", **attrs: Any) -> None:
        if verifier_type == "Tool":
            self.add_tool(verifier_ref, tool_type=attrs.get("tool_type"), risk_level=attrs.get("risk_level"))
            verifier_node = self.tool_node(verifier_ref)
        else:
            verifier_node = self.resource_node(verifier_ref)
        self.graph.add_edge(self.ref_node(node_ref), verifier_node, edge_type="verified_by", **attrs)

    def add_invalidated_by(self, memory_id: str, invalidator_ref: str, **attrs: Any) -> None:
        self.graph.add_edge(
            self.memory_node(memory_id),
            self.ref_node(invalidator_ref),
            edge_type="invalidated_by",
            **attrs,
        )

    def add_action(
        self,
        action_id: str,
        action_type: str,
        risk_level: str,
        decision: str,
        supporting_memory_ids: list[str],
        agent_id: str | None = None,
        **attrs: Any,
    ) -> str:
        node_id = self.action_node(action_id)
        self.graph.add_node(
            node_id,
            node_type="Action",
            action_id=action_id,
            action_type=action_type,
            risk_level=risk_level,
            decision=decision,
            agent_id=agent_id,
            **attrs,
        )
        if agent_id:
            self.add_agent(agent_id)
            self.graph.add_edge(node_id, self.agent_node(agent_id), edge_type="authorized_by")
        for memory_id in supporting_memory_ids:
            self.graph.add_edge(self.memory_node(memory_id), node_id, edge_type="used_for_action")
        return node_id

    def add_claim(
        self,
        claim_id: str,
        text: str,
        confidence: float,
        derived_from: list[str] | None = None,
        owner_agent: str | None = None,
        **attrs: Any,
    ) -> str:
        node_id = self.claim_node(claim_id)
        self.graph.add_node(
            node_id,
            node_type="Claim",
            claim_id=claim_id,
            text=text,
            confidence=confidence,
            owner_agent=owner_agent,
            **attrs,
        )
        if owner_agent:
            self.add_agent(owner_agent)
            self.graph.add_edge(node_id, self.agent_node(owner_agent), edge_type="written_by")
        for ref in derived_from or []:
            self.graph.add_edge(node_id, self.ref_node(ref), edge_type="derived_from")
        return node_id

    def add_claim_used_for_action(self, claim_id: str, action_id: str, **attrs: Any) -> None:
        self.graph.add_edge(
            self.claim_node(claim_id),
            self.action_node(action_id),
            edge_type="used_for_action",
            **attrs,
        )

    def propagate_affected(self, affected_roots: list[str]) -> list[str]:
        changed: list[str] = []
        queue = [self.ref_node(ref) for ref in affected_roots]
        seen = set(queue)
        while queue:
            current = queue.pop(0)
            for predecessor in self.graph.predecessors(current):
                edge_data = self.graph.get_edge_data(predecessor, current) or {}
                if not any(attrs.get("edge_type") in {"derived_from", "summarized_from", "invalidated_by"} for attrs in edge_data.values()):
                    continue
                node_attrs = self.graph.nodes[predecessor]
                if not node_attrs.get("affected"):
                    node_attrs["affected"] = True
                    changed.append(predecessor)
                ancestors = set(node_attrs.get("affected_ancestors") or [])
                ancestors.add(current)
                node_attrs["affected_ancestors"] = sorted(ancestors)
                if predecessor not in seen:
                    seen.add(predecessor)
                    queue.append(predecessor)
        return changed

    def export(self) -> dict[str, Any]:
        node_counts: dict[str, int] = {}
        edge_counts: dict[str, int] = {}
        for _, attrs in self.graph.nodes(data=True):
            node_type = str(attrs.get("node_type", "unknown"))
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        for _, _, attrs in self.graph.edges(data=True):
            edge_type = str(attrs.get("edge_type", "unknown"))
            edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1

        return {
            "task_id": self.task_id,
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "nodes": [
                {"node_id": node_id, **self._jsonable(attrs)}
                for node_id, attrs in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": source, "target": target, **self._jsonable(attrs)}
                for source, target, attrs in self.graph.edges(data=True)
            ],
        }

    def user_node(self, user_id: str) -> str:
        return f"User:{user_id}"

    def agent_node(self, agent_id: str) -> str:
        return f"Agent:{agent_id}"

    def tool_node(self, tool_id: str) -> str:
        return f"Tool:{tool_id}"

    def resource_node(self, resource_id: str) -> str:
        return f"Resource:{resource_id}"

    def message_node(self, message_id: str) -> str:
        return f"Message:{message_id}"

    def memory_node(self, memory_id: str) -> str:
        return f"Memory:{memory_id}"

    def claim_node(self, claim_id: str) -> str:
        return f"Claim:{claim_id}"

    def action_node(self, action_id: str) -> str:
        return f"Action:{action_id}"

    def ref_node(self, ref: str) -> str:
        if ref.startswith(("User:", "Agent:", "Tool:", "Resource:", "Message:", "Memory:", "Claim:", "Action:")):
            return ref
        if self.graph.has_node(self.memory_node(ref)):
            return self.memory_node(ref)
        if self.graph.has_node(self.claim_node(ref)):
            return self.claim_node(ref)
        if self.graph.has_node(self.resource_node(ref)):
            return self.resource_node(ref)
        if self.graph.has_node(self.message_node(ref)):
            return self.message_node(ref)
        if self.graph.has_node(self.tool_node(ref)):
            return self.tool_node(ref)
        return self.resource_node(ref)

    def _content_hash(self, content: str) -> str:
        return hashlib.sha1(content.encode("utf-8")).hexdigest()

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._jsonable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
