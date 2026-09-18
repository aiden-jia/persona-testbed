from abc import ABC, abstractmethod
from openai import OpenAI
import config


class BasePromptMethod(ABC):
    name: str = "base"
    description: str = "Abstract base prompting method."
    uses_rag: bool = True

    def __init__(self):
        self._client: OpenAI = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        return self._client

    @abstractmethod
    def build_messages(
        self,
        persona_info: dict,
        rag_examples: list[dict],
        conversation_history: list[dict],
        user_input: str,
    ) -> list[dict]:
        """Construct the messages list to send to the OpenAI API."""
        ...

    def generate(
        self,
        persona_info: dict,
        rag_examples: list[dict],
        conversation_history: list[dict],
        user_input: str,
        temperature: float = 0.7,
        **kwargs,
    ) -> tuple[str, dict]:
        """
        Generate a persona response.
        Returns (response_text, metadata_dict).
        Subclasses may override this entirely (e.g. for multi-call methods).
        """
        messages = self.build_messages(
            persona_info, rag_examples, conversation_history, user_input
        )
        resp = self.client.chat.completions.create(
            model=config.MODEL,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        text = resp.choices[0].message.content.strip()
        meta = {
            "method": self.name,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "rag_examples_used": len(rag_examples),
            "api_calls": 1,
        }
        return text, meta

    def generate_streamed(
        self,
        persona_info: dict,
        rag_examples: list[dict],
        conversation_history: list[dict],
        user_input: str,
        temperature: float = 0.7,
        **kwargs,
    ):
        """
        Streaming variant of generate().
        Returns (stream, metadata_dict); caller iterates the stream for chunks.
        Subclasses with multi-call flows should override this to run prep calls
        synchronously and stream only the final generation.
        """
        messages = self.build_messages(
            persona_info, rag_examples, conversation_history, user_input
        )
        stream = self.client.chat.completions.create(
            model=config.MODEL,
            messages=messages,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        meta = {
            "method": self.name,
            "rag_examples_used": len(rag_examples),
            "api_calls": 1,
        }
        return stream, meta

    # ------------------------------------------------------------------ #
    # Shared formatting helpers
    # ------------------------------------------------------------------ #

    def _build_system_core(self, persona_info: dict) -> str:
        """
        The foundational system prompt injected at the top of every conversation.
        Uses the [ROLE_PROMPT] section if present — a rich narrative character
        brief covering personality, speaking style, behavior, goals, and tone.
        Falls back to a structured key-value format if no role prompt is defined.
        """
        role_prompt = persona_info.get("role_prompt", "").strip()
        if role_prompt:
            return role_prompt

        # Structured fallback for personas without a [ROLE_PROMPT] section
        name = persona_info.get("name", "the persona")
        return (
            f"You are roleplaying as {name}, a potential car buyer visiting a dealership.\n\n"
            f"PERSONA PROFILE:\n{self._format_persona_block(persona_info)}\n\n"
            "Stay in character at all times. Do not break character or acknowledge you are an AI."
        )

    def _format_persona_block(self, persona_info: dict) -> str:
        skip = {"type", "persona_name", "role_prompt"}
        lines = []
        for k, v in persona_info.items():
            if k not in skip and v:
                label = k.replace("_", " ").title()
                lines.append(f"- {label}: {v}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Walk-out detection
    # ------------------------------------------------------------------ #

    _WALKOUT_DETECT_PROMPT = """\
You are deciding whether a persona has actually ended a negotiation and left.

PERSONA: {name}
PERSONA'S RESPONSE: "{response}"

Did the persona definitively end the interaction — said goodbye and is leaving, \
explicitly stated they will not continue this transaction, or made unmistakably \
clear the conversation is over?

Do NOT answer yes for:
- Frustration, skepticism, or coolness that still leaves room for the deal
- A threat to leave framed as a condition ("if you can't do X I'll have to go")
- Ultimatums where the persona is still present and waiting for a reply

Answer with ONLY the single word "yes" or "no"."""

    def detect_walkout(self, response_text: str, persona_info: dict) -> bool:
        """Return True if the persona's response represents an actual departure."""
        name = persona_info.get("name", "the persona")
        prompt = self._WALKOUT_DETECT_PROMPT.format(name=name, response=response_text)
        try:
            resp = self.client.chat.completions.create(
                model=config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            return resp.choices[0].message.content.strip().lower().startswith("yes")
        except Exception:
            return False

    def _format_examples_block(
        self, examples: list[dict], persona_name: str = "Persona"
    ) -> str:
        if not examples:
            return ""

        is_cold_call = "caller_turn" in examples[0]

        if is_cold_call:
            parts = ["BEHAVIORAL GROUND TRUTH — real recipient responses from similar cold calls. Your response must draw directly from these:\n"]
            for i, ex in enumerate(examples, 1):
                parts.append(f"Example {i}:")
                prior = ex.get("prior_context", [])
                if prior:
                    parts.append("  [Earlier in this call:]")
                    for line in prior[-2:]:
                        parts.append(f"    {line}")
                parts.append(f'  Caller said: "{ex["caller_turn"]}"')
                parts.append(f'  You could say: "{ex["recipient_turn"]}"')
                parts.append("")
        else:
            parts = [f"EXAMPLE RESPONSES ({persona_name} in similar situations):\n"]
            for i, ex in enumerate(examples, 1):
                parts.append(f"Example {i}:")
                parts.append(f'  Salesperson: "{ex["user_msg"]}"')
                parts.append(f'  {persona_name}: "{ex["persona_response"]}"')
                parts.append("")

        return "\n".join(parts)
