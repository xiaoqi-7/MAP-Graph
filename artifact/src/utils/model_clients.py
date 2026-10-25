import os
import time
from pathlib import Path
from typing import Iterable

import yaml
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_model_config(config_path: str | Path | None = None) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")

    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "model.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _env_value(config: dict, key: str, default_key: str | None = None):
    env_name = config.get(key)
    if env_name:
        value = os.getenv(env_name)
        if value not in (None, ""):
            return value
    if default_key:
        return config.get(default_key)
    return None


def _env_bool(config: dict, key: str, default_key: str | None = None) -> bool:
    value = _env_value(config, key, default_key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _openai_compatible_client(base_url: str, api_key: str) -> OpenAI:
    if not base_url:
        raise ValueError("Missing base_url. Check LLM_BASE_URL or EMBEDDING_BASE_URL.")
    if not api_key:
        raise ValueError("Missing api_key. Check LLM_API_KEY or EMBEDDING_API_KEY.")
    return OpenAI(api_key=api_key, base_url=base_url)


class LLMClient:
    def __init__(self, config_path: str | Path | None = None):
        config = load_model_config(config_path)["llm"]
        self.model = _env_value(config, "model_env", "default_model")
        self.temperature = float(_env_value(config, "temperature_env", "default_temperature"))
        self.max_tokens = int(_env_value(config, "max_tokens_env", "default_max_tokens"))
        self.client = _openai_compatible_client(
            base_url=_env_value(config, "base_url_env"),
            api_key=_env_value(config, "api_key_env"),
        )
        self._call_stats: list[dict] = []

    def chat(self, messages: list[dict], **kwargs) -> str:
        call_start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=kwargs.pop("model", self.model),
            messages=messages,
            temperature=kwargs.pop("temperature", self.temperature),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            **kwargs,
        )
        usage = getattr(response, "usage", None)
        prompt_tokens = self._usage_value(
            usage,
            "prompt_tokens",
            "input_tokens",
        )
        completion_tokens = self._usage_value(
            usage,
            "completion_tokens",
            "output_tokens",
        )
        total_tokens = self._usage_value(usage, "total_tokens")
        if (
            total_tokens is None
            and prompt_tokens is not None
            and completion_tokens is not None
        ):
            total_tokens = prompt_tokens + completion_tokens
        self._call_stats.append(
            {
                "latency_ms": round(
                    (time.perf_counter() - call_start) * 1000,
                    3,
                ),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model": getattr(response, "model", self.model),
            }
        )
        return response.choices[0].message.content or ""

    def drain_call_stats(self) -> list[dict]:
        stats = list(self._call_stats)
        self._call_stats.clear()
        return stats

    @staticmethod
    def _usage_value(usage, *names: str) -> int | None:
        if usage is None:
            return None
        for name in names:
            value = (
                usage.get(name)
                if isinstance(usage, dict)
                else getattr(usage, name, None)
            )
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        return None


class EmbeddingClient:
    _sentence_transformer_cache = {}

    def __init__(self, config_path: str | Path | None = None):
        config = load_model_config(config_path)["embedding"]
        self.provider = _env_value(config, "provider_env", "default_provider")
        self.model = _env_value(config, "model_env", "default_model")
        self.normalize = _env_bool(config, "normalize_env", "default_normalize")
        self.local_files_only = _env_bool(config, "local_files_only_env", "default_local_files_only")

        if self.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            device = _env_value(config, "device_env", "default_device")
            local_model_path = PROJECT_ROOT / self.model
            model_name_or_path = str(local_model_path) if local_model_path.exists() else self.model
            cache_key = (model_name_or_path, device, self.local_files_only)
            if cache_key not in self._sentence_transformer_cache:
                self._sentence_transformer_cache[cache_key] = SentenceTransformer(
                    model_name_or_path,
                    device=device,
                    local_files_only=self.local_files_only,
                )
            self.client = self._sentence_transformer_cache[cache_key]
        elif self.provider == "openai_compatible":
            self.client = _openai_compatible_client(
                base_url=_env_value(config, "base_url_env"),
                api_key=_env_value(config, "api_key_env"),
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def embed(self, texts: str | Iterable[str]) -> list[list[float]]:
        inputs = [texts] if isinstance(texts, str) else list(texts)
        if self.provider == "sentence_transformers":
            embeddings = self.client.encode(
                inputs,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )
            return embeddings.tolist()

        response = self.client.embeddings.create(
            model=self.model,
            input=inputs,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]
