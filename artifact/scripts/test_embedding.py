from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.model_clients import EmbeddingClient


def main():
    embedder = EmbeddingClient()
    vectors = embedder.embed(
        [
            "MAP-Graph tracks memory provenance across multiple agents.",
            "Private memories should not leak to unauthorized agents.",
        ]
    )
    print(f"provider={embedder.provider}")
    print(f"model={embedder.model}")
    print(f"count={len(vectors)}")
    print(f"dim={len(vectors[0]) if vectors else 0}")


if __name__ == "__main__":
    main()

