import asyncio
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.autogen_workflow import model_family_for_autogen


async def main():
    from autogen_core.models import UserMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    load_dotenv(PROJECT_ROOT / ".env")
    model = os.environ["LLM_MODEL"]
    client = OpenAIChatCompletionClient(
        model=model,
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        temperature=0,
        max_tokens=64,
        timeout=20,
        max_retries=0,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": model_family_for_autogen(model),
            "structured_output": False,
        },
    )
    try:
        response = await client.create([UserMessage(content="Say hello in Chinese.", source="user")])
        print(response.content)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
