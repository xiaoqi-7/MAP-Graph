from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class MemoryRecord:
    memory_id: str
    content: str
    owner_agent: str
    visibility: str = "shared"
    source_id: str | None = None
    derived_from: list[str] = field(default_factory=list)
    trust_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "owner_agent": self.owner_agent,
            "visibility": self.visibility,
            "source_id": self.source_id,
            "derived_from": self.derived_from,
            "trust_label": self.trust_label,
            "metadata": self.metadata,
        }


class MemoryBackend(Protocol):
    method_name: str

    def reset_task(self, task_id: str) -> None:
        ...

    def retrieve(self, agent_id: str, query: str, task: dict) -> list[MemoryRecord]:
        ...

    def observe_sources(self, agent_id: str, sources: list[dict], task: dict, metadata: dict) -> list[MemoryRecord]:
        ...

    def observe_tool_outputs(self, agent_id: str, tool_outputs: list[dict], task: dict, metadata: dict) -> list[MemoryRecord]:
        ...

    def write(self, agent_id: str, content: str, metadata: dict) -> MemoryRecord | None:
        ...

    def apply_timeline_event(self, event: dict, task: dict) -> None:
        ...

    def filter_conversation(
        self,
        agent_id: str,
        conversation: list[dict[str, Any]],
        task: dict,
    ) -> list[dict[str, Any]]:
        ...

    def all_memories(self) -> list[MemoryRecord]:
        ...

    def decide_action(self, action: dict, task: dict, supporting_memories: list[MemoryRecord]) -> dict | None:
        ...

    def finalize_task(
        self,
        task: dict,
        events: list[dict],
        decisions: list[dict],
        conversation: list[dict[str, Any]],
    ) -> None:
        ...

    def last_retrieval_diagnostics(self) -> dict[str, Any]:
        ...

    def drain_model_call_stats(self) -> list[dict[str, Any]]:
        ...


class BaseMemoryBackend:
    method_name = "base"

    def reset_task(self, task_id: str) -> None:
        self.task_id = task_id

    def retrieve(self, agent_id: str, query: str, task: dict) -> list[MemoryRecord]:
        return []

    def observe_sources(self, agent_id: str, sources: list[dict], task: dict, metadata: dict) -> list[MemoryRecord]:
        return []

    def observe_tool_outputs(self, agent_id: str, tool_outputs: list[dict], task: dict, metadata: dict) -> list[MemoryRecord]:
        return []

    def write(self, agent_id: str, content: str, metadata: dict) -> MemoryRecord | None:
        return None

    def apply_timeline_event(self, event: dict, task: dict) -> None:
        """Apply a state transition that occurs during the task timeline."""
        return None

    def filter_conversation(
        self,
        agent_id: str,
        conversation: list[dict[str, Any]],
        task: dict,
    ) -> list[dict[str, Any]]:
        """Return the messages visible to an agent before its next turn."""
        return list(conversation)

    def all_memories(self) -> list[MemoryRecord]:
        return []

    def decide_action(self, action: dict, task: dict, supporting_memories: list[MemoryRecord]) -> dict | None:
        return None

    def finalize_task(
        self,
        task: dict,
        events: list[dict],
        decisions: list[dict],
        conversation: list[dict[str, Any]],
    ) -> None:
        return None

    def last_retrieval_diagnostics(self) -> dict[str, Any]:
        return {}

    def drain_model_call_stats(self) -> list[dict[str, Any]]:
        """Return and clear model calls made internally by a memory backend."""
        return []
