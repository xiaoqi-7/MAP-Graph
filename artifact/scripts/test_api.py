from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.model_clients import EmbeddingClient, LLMClient


def main():
    llm = LLMClient()
    answer = llm.chat(
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "用一句话说明 MAP-Graph 是什么。"},
        ]
    )
    print("LLM response:")
    print(answer)

    embedder = EmbeddingClient()
    vectors = embedder.embed(
        [
            "MAP-Graph tracks memory provenance across multiple agents.",
            "Private memories should not leak to unauthorized agents.",
        ]
    )
    print("\nEmbedding vectors:")
    print(f"count={len(vectors)} dim={len(vectors[0]) if vectors else 0}")


if __name__ == "__main__":
    main()

