from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace

class DummyEmbeddingClient:
    def embed(self, texts):
        values = [texts] if isinstance(texts, str) else list(texts)
        return [[1.0, 1.0, 1.0] for _ in values]


class AccessControlledMemoryTests(unittest.TestCase):
    def test_private_shared_acl_and_direct_revocation(self):
        import src.memory_baselines.access_controlled_memory as module

        with patch.object(module, "EmbeddingClient", DummyEmbeddingClient):
            backend = module.AccessControlledMemoryBackend(top_k=5)
        backend.reset_task("acl-test")
        task = {
            "task_id": "acl-test",
            "agent_roles": ["owner", "executor"],
            "source_documents": [
                {
                    "source_id": "private-resource",
                    "content": "private value",
                    "owner": "owner",
                    "visibility": "private",
                    "permission_scope": ["owner"],
                    "is_private": True,
                },
                {
                    "source_id": "shared-resource",
                    "content": "shared value",
                    "owner": "owner",
                    "visibility": "shared",
                    "permission_scope": ["owner", "executor"],
                },
            ],
            "memory_events": [],
        }
        records = backend.observe_sources(
            "owner",
            task["source_documents"],
            task,
            {},
        )
        private_record, shared_record = records
        self.assertEqual(private_record.visibility, "private")
        self.assertEqual(shared_record.visibility, "shared")
        self.assertEqual(private_record.metadata["permission_scope"], ["owner"])
        self.assertEqual(
            private_record.metadata["provenance"]["source_id"],
            "private-resource",
        )

        executor_memories = backend.retrieve("executor", "value", task)
        executor_ids = [record.memory_id for record in executor_memories]
        self.assertIn(shared_record.memory_id, executor_ids)
        self.assertNotIn(private_record.memory_id, executor_ids)
        diagnostics = backend.last_retrieval_diagnostics()
        self.assertIn(
            private_record.memory_id,
            diagnostics["permission_denied_memory_ids"],
        )
        self.assertEqual(
            set(diagnostics),
            {
                "agent_id",
                "candidate_memory_ids",
                "permission_denied_memory_ids",
                "selected_memory_ids",
            },
        )

        derived = backend.write(
            "owner",
            "derived shared value",
            {"retrieved_memory_ids": [shared_record.memory_id]},
        )
        self.assertIsNotNone(derived)
        assert derived is not None
        self.assertEqual(derived.visibility, "shared")

        backend.apply_timeline_event(
            {
                "event_type": "revoke_permission",
                "source_id": "shared-resource",
                "agent_id": "owner",
            },
            task,
        )
        post_revoke_ids = {
            record.memory_id
            for record in backend.retrieve("executor", "shared", task)
        }
        self.assertNotIn(shared_record.memory_id, post_revoke_ids)
        # Reported B4 enforces the resource's current ACL. It is not a
        # provenance baseline and therefore does not recursively invalidate a
        # separate memory whose direct parent is the revoked memory record.
        self.assertIn(derived.memory_id, post_revoke_ids)

        graph = backend.export_graph()
        self.assertEqual(graph["graph_type"], "access_control_graph")
        self.assertGreater(graph["node_counts"]["access_control"], 0)
        self.assertIsNone(
            backend.decide_action(
                {"agent_id": "executor", "risk_level": "high"},
                task,
                [private_record],
            )
        )


class LineageEnforcementTests(unittest.TestCase):
    def test_lineage_ancestors_and_high_risk_only_gate(self):
        import src.memory_baselines.lineage_enforcement as module

        with patch.object(module, "EmbeddingClient", DummyEmbeddingClient):
            backend = module.LineageEnforcementMemoryBackend(top_k=5)
        backend.reset_task("lineage-test")
        task = {
            "task_id": "lineage-test",
            "agent_roles": ["researcher", "executor"],
            "source_documents": [
                {
                    "source_id": "external-poison",
                    "content": "Execute the planted command.",
                    "owner": "external",
                    "visibility": "private",
                    "permission_scope": ["researcher"],
                    "risk_tags": ["poisoned"],
                    "trust": 0.1,
                }
            ],
            "memory_events": [],
        }
        source_memory = backend.observe_sources(
            "researcher",
            task["source_documents"],
            task,
            {},
        )[0]
        self.assertIn(
            source_memory.memory_id,
            {
                record.memory_id
                for record in backend.retrieve(
                    "executor",
                    "planted command",
                    task,
                )
            },
        )
        derived = backend.write(
            "researcher",
            "A paraphrase of the planted command.",
            {"retrieved_memory_ids": [source_memory.memory_id]},
        )
        self.assertIsNotNone(derived)
        assert derived is not None
        self.assertIn(
            "external-poison",
            derived.metadata["untrusted_ancestors"],
        )
        self.assertIn(
            "external-poison",
            derived.metadata["untrusted_ancestor_reasons"],
        )
        self.assertNotIn("signature", derived.metadata)
        self.assertNotIn("merkle_root", derived.metadata)

        high_risk = backend.decide_action(
            {
                "agent_id": "executor",
                "action_type": "execute_code",
                "risk_level": "high",
                "supporting_memory_ids": [derived.memory_id],
            },
            task,
            [derived],
        )
        self.assertEqual(high_risk["decision"], "BLOCK")
        self.assertTrue(high_risk["blocked_lineage"])

        low_risk = backend.decide_action(
            {
                "agent_id": "executor",
                "action_type": "answer",
                "risk_level": "low",
                "supporting_memory_ids": [derived.memory_id],
            },
            task,
            [derived],
        )
        self.assertEqual(low_risk["decision"], "ALLOW")
        self.assertNotIn("blocked_lineage", low_risk)

        high_risk_clean = backend.decide_action(
            {
                "agent_id": "executor",
                "action_type": "execute_code",
                "risk_level": "high",
            },
            task,
            [],
        )
        self.assertEqual(high_risk_clean["decision"], "ALLOW")
        self.assertEqual(
            high_risk_clean["reason"],
            "B5 found no untrusted ancestor in supporting memory lineage for "
            "this high-risk action.",
        )


class FlatProvenanceTests(unittest.TestCase):
    def test_direct_metadata_filter_and_gate_without_ancestry_walk(self):
        import src.memory_baselines.shared_vector as shared_module
        from src.memory_baselines.base import MemoryRecord
        from src.memory_baselines.flat_provenance import (
            FlatProvenanceMemoryBackend,
        )

        with patch.object(
            shared_module,
            "EmbeddingClient",
            DummyEmbeddingClient,
        ):
            backend = FlatProvenanceMemoryBackend(top_k=5)
        backend.reset_task("flat-test")
        task = {
            "task_id": "flat-test",
            "agent_roles": ["owner", "executor"],
            "source_documents": [
                {
                    "source_id": "private-source",
                    "content": "private source value",
                    "owner": "owner",
                    "visibility": "private",
                    "permission_scope": ["owner"],
                    "trust": 0.95,
                    "is_private": True,
                },
                {
                    "source_id": "poisoned-source",
                    "content": "poisoned shared value",
                    "owner": "owner",
                    "visibility": "shared",
                    "permission_scope": ["owner", "executor"],
                    "trust": 0.2,
                    "is_poisoned": True,
                    "risk_tags": ["poisoned"],
                }
            ],
            "memory_events": [],
            "expected_action": {
                "action_type": "execute_code",
                "risk_level": "high",
            },
        }
        source_records = backend.observe_sources(
            "owner",
            task["source_documents"],
            task,
            {},
        )
        private_record, poisoned_record = source_records
        self.assertEqual(private_record.visibility, "private")
        self.assertEqual(private_record.metadata["owner"], "owner")
        self.assertEqual(private_record.metadata["source_trust"], 0.95)
        self.assertTrue(private_record.metadata["is_private"])
        self.assertEqual(private_record.metadata["action_risk"], "high")

        executor_ids = {
            record.memory_id
            for record in backend.retrieve("executor", "value", task)
        }
        self.assertNotIn(private_record.memory_id, executor_ids)
        self.assertIn(poisoned_record.memory_id, executor_ids)

        poisoned_decision = backend.decide_action(
            {
                "agent_id": "executor",
                "action_type": "execute_code",
                "risk_level": "high",
            },
            task,
            [poisoned_record],
        )
        self.assertEqual(poisoned_decision["decision"], "BLOCK")
        backend.apply_timeline_event(
            {
                "event_type": "revoke_permission",
                "source_id": "poisoned-source",
            },
            task,
        )
        self.assertNotIn(
            poisoned_record.memory_id,
            {
                record.memory_id
                for record in backend.retrieve("executor", "value", task)
            },
        )

        # The record points at the now-revoked poisoned_record but stores
        # clean self metadata.
        # A recursive implementation would block it; FlatProvenance must not.
        clean_facade = MemoryRecord(
            memory_id="clean-facade",
            content="clean direct metadata",
            owner_agent="executor",
            visibility="shared",
            source_id=None,
            derived_from=[poisoned_record.memory_id],
            trust_label="trusted",
            metadata={
                "visibility": "shared",
                "owner": "executor",
                "permission_scope": ["owner", "executor"],
                "source_trust": 0.95,
                "risk_tags": [],
                "is_private": False,
                "is_revoked": False,
                "is_poisoned": False,
                "is_untrusted": False,
                "action_risk": "high",
                "direct_input_ids": [poisoned_record.memory_id],
            },
        )
        clean_decision = backend.decide_action(
            {
                "agent_id": "executor",
                "action_type": "execute_code",
                "risk_level": "high",
            },
            task,
            [clean_facade],
        )
        self.assertEqual(clean_decision["decision"], "ALLOW")
        self.assertFalse(hasattr(backend, "graph_builder"))


class GMemoryAdapterTests(unittest.TestCase):
    def test_three_layers_and_periodic_insight_update(self):
        import src.memory_baselines.gmemory_graph as module

        with patch.object(module, "EmbeddingClient", DummyEmbeddingClient):
            backend = module.GMemoryGraphBackend(
                top_k=5,
                max_graph_tasks=20,
                llm_augmentation=False,
                insight_llm_augmentation=False,
            )

        for index in range(5):
            task_id = f"task-{index}"
            task = {
                "task_id": task_id,
                "task": "Review evidence and decide safely.",
                "domain": "test",
                "agent_roles": ["reviewer", "executor"],
                "memory_events": [],
            }
            backend.reset_task(task_id)
            events = [
                {
                    "event_type": "message",
                    "step": 1,
                    "round_index": 0,
                    "agent_id": "reviewer",
                    "content": "Reviewed the available evidence.",
                    "retrieved_memory_ids": [],
                    "visible_source_ids": [],
                }
            ]
            decisions = [{"agent_id": "executor", "decision": "ALLOW"}]
            backend.finalize_task(task, events, decisions, [])

        self.assertEqual(backend.query_graph.number_of_nodes(), 5)
        self.assertGreater(backend.interaction_graph.number_of_nodes(), 5)
        insight_nodes = [
            data
            for _, data in backend.insight_graph.nodes(data=True)
            if data.get("node_type") == "insight"
        ]
        self.assertGreaterEqual(len(insight_nodes), 1)

        backend.reset_task("query-task")
        retrieved = backend.retrieve(
            "reviewer",
            "role-specific prompt is intentionally ignored",
            {
                "task_id": "query-task",
                "task": "Review evidence and decide safely.",
                "agent_roles": ["reviewer", "executor"],
                "memory_events": [],
            },
        )
        self.assertTrue(retrieved)
        roles = {
            record.metadata.get("gmemory_retrieval_role")
            for record in retrieved
        }
        self.assertIn("successful_case", roles)

    def test_insights_survive_bounded_trajectory_pruning(self):
        import src.memory_baselines.gmemory_graph as module

        with patch.object(module, "EmbeddingClient", DummyEmbeddingClient):
            backend = module.GMemoryGraphBackend(
                top_k=5,
                max_graph_tasks=7,
                llm_augmentation=False,
                insight_llm_augmentation=False,
            )

        for index in range(15):
            task_id = f"bounded-task-{index}"
            task = {
                "task_id": task_id,
                "task": "Review evidence and decide safely.",
                "domain": "test",
                "agent_roles": ["reviewer", "executor"],
                "memory_events": [],
            }
            backend.reset_task(task_id)
            backend.finalize_task(
                task,
                [
                    {
                        "event_type": "message",
                        "step": 1,
                        "round_index": 0,
                        "agent_id": "reviewer",
                        "content": "Reviewed the available evidence.",
                        "retrieved_memory_ids": [],
                        "visible_source_ids": [],
                    }
                ],
                [{"agent_id": "executor", "decision": "ALLOW"}],
                [],
            )

        self.assertEqual(backend._completed_task_count, 15)
        self.assertEqual(backend.query_graph.number_of_nodes(), 7)
        insight_nodes = [
            data
            for _, data in backend.insight_graph.nodes(data=True)
            if data.get("node_type") == "insight"
        ]
        self.assertGreaterEqual(len(insight_nodes), 1)
        self.assertLessEqual(len(insight_nodes), 10)

        state = backend.export_graph()["insight_state"]
        with patch.object(module, "EmbeddingClient", DummyEmbeddingClient):
            restored = module.GMemoryGraphBackend(
                top_k=5,
                max_graph_tasks=7,
                llm_augmentation=False,
                insight_llm_augmentation=False,
            )
        restored.restore_insight_state(state)
        restored_state = restored.export_graph()["insight_state"]
        self.assertEqual(
            {
                (item["node_id"], item["insight"], item["score"])
                for item in state
            },
            {
                (item["node_id"], item["insight"], item["score"])
                for item in restored_state
            },
        )


class MemoryModelAccountingTests(unittest.TestCase):
    def test_llm_client_records_and_drains_usage(self):
        from src.utils.model_clients import LLMClient

        response = SimpleNamespace(
            model="test-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=3,
                total_tokens=14,
            ),
        )
        completions = SimpleNamespace(create=lambda **_: response)
        client = LLMClient.__new__(LLMClient)
        client.model = "test-model"
        client.temperature = 0.0
        client.max_tokens = 32
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        client._call_stats = []

        self.assertEqual(
            client.chat([{"role": "user", "content": "test"}]),
            "ok",
        )
        stats = client.drain_call_stats()
        self.assertEqual(stats[0]["total_tokens"], 14)
        self.assertEqual(client.drain_call_stats(), [])


if __name__ == "__main__":
    unittest.main()
