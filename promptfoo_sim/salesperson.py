"""
Salesperson LLM agent for the promptfoo simulation.

Maintains the salesperson's side of the conversation using gpt-4o-mini.
The conversation history is from the salesperson's perspective:
  role="assistant" → salesperson's previous turns
  role="user"      → persona's responses
"""
import sys
from pathlib import Path

_PARENT = Path(__file__).parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from openai import OpenAI
import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def generate_salesperson_response(
    system_prompt: str,
    conversation_history: list[dict],
    temperature: float = 0.7,
) -> tuple[str, dict]:
    """
    Generate the next salesperson message.

    Returns (response_text, token_metadata).
    """
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    resp = _get_client().chat.completions.create(
        model=config.MODEL,
        messages=messages,
        temperature=temperature,
    )
    text = resp.choices[0].message.content.strip()
    meta = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }
    return text, meta
