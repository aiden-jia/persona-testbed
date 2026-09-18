"""
Reflection prompting for persona simulation.

Process (3 API calls):
  1. DRAFT   — generate an initial in-character response.
  2. CRITIQUE — critique the draft against the persona's traits.
  3. REFINE  — produce an improved response incorporating the critique.
"""
from prompting.base import BasePromptMethod
import config


class ReflectionPrompt(BasePromptMethod):
    name = "reflection"
    description = (
        "Reflection: draft → self-critique against persona → refined response (3 API calls)."
    )
    uses_rag = True

    _CRITIQUE_PROMPT = """\
You are evaluating a roleplay response for persona authenticity.

PERSONA: {name}
CHARACTER BRIEF:
{profile}

OTHER PARTY SAID: "{user_input}"
DRAFT RESPONSE: "{draft}"

Critically evaluate the draft on:
1. Does the tone match the personality and speaking style described in the character brief above?
2. Does the response reflect the persona's stated priorities and concerns?
3. Is the level of detail and specificity appropriate — not too verbose, not too thin?
4. Is there anything that feels out-of-character (too compliant, too hostile, too generic, or too polished)?
5. What one or two concrete changes would make it more authentic?

Be concise and specific."""

    # ------------------------------------------------------------------ #

    def build_messages(self, persona_info, rag_examples, conversation_history, user_input):
        # Not used directly — generate() controls the full flow.
        raise NotImplementedError("ReflectionPrompt uses generate() directly.")

    def generate(self, persona_info, rag_examples, conversation_history, user_input, **kwargs):
        name = persona_info.get("name", "the persona")
        # Use role_prompt as the profile text for critique (richer context for evaluation)
        profile = persona_info.get("role_prompt") or self._format_persona_block(persona_info)
        examples_block = self._format_examples_block(rag_examples, name)

        # System core built from role_prompt (persists across all three steps)
        system_core = self._build_system_core(persona_info)
        if examples_block:
            system_core += f"\n\n{examples_block}"

        # Step 1: Draft
        draft_system = system_core + "\n\nProvide an initial response. Stay fully in character."
        draft_messages = [{"role": "system", "content": draft_system}]
        draft_messages.extend(conversation_history)
        draft_messages.append({"role": "user", "content": user_input})

        r1 = self.client.chat.completions.create(
            model=config.MODEL, messages=draft_messages, temperature=0.7
        )
        draft = r1.choices[0].message.content.strip()

        # Step 2: Critique — analytical prompt, not a roleplay system prompt
        critique_prompt = self._CRITIQUE_PROMPT.format(
            name=name, profile=profile, user_input=user_input, draft=draft
        )
        r2 = self.client.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": critique_prompt}],
            temperature=0.3,
        )
        critique = r2.choices[0].message.content.strip()

        # Step 3: Refine — same system core + critique guidance
        refine_system = (
            system_core
            + f'\n\nYour first draft was: "{draft}"'
            + f'\n\nCritique of that draft: "{critique}"'
            + "\n\nNow produce a refined, more authentic response that addresses the critique. "
            "Output only the final response — no labels or meta-commentary."
        )
        refine_messages = [{"role": "system", "content": refine_system}]
        refine_messages.extend(conversation_history)
        refine_messages.append({"role": "user", "content": user_input})

        r3 = self.client.chat.completions.create(
            model=config.MODEL, messages=refine_messages, temperature=0.7
        )
        final = r3.choices[0].message.content.strip()

        total_tokens = sum(
            r.usage.prompt_tokens + r.usage.completion_tokens for r in [r1, r2, r3]
        )
        meta = {
            "method": self.name,
            "draft": draft,
            "critique": critique,
            "total_tokens": total_tokens,
            "rag_examples_used": len(rag_examples),
            "api_calls": 3,
        }
        return final, meta

    def generate_streamed(self, persona_info, rag_examples, conversation_history, user_input, **kwargs):
        """Draft and critique run synchronously; only the final refine call streams."""
        name = persona_info.get("name", "the persona")
        profile = persona_info.get("role_prompt") or self._format_persona_block(persona_info)
        examples_block = self._format_examples_block(rag_examples, name)

        system_core = self._build_system_core(persona_info)
        if examples_block:
            system_core += f"\n\n{examples_block}"

        # Step 1: Draft (sync)
        draft_messages = [{"role": "system", "content": system_core + "\n\nProvide an initial response. Stay fully in character."}]
        draft_messages.extend(conversation_history)
        draft_messages.append({"role": "user", "content": user_input})
        r1 = self.client.chat.completions.create(model=config.MODEL, messages=draft_messages, temperature=0.7)
        draft = r1.choices[0].message.content.strip()

        # Step 2: Critique (sync)
        critique_prompt = self._CRITIQUE_PROMPT.format(
            name=name, profile=profile, user_input=user_input, draft=draft
        )
        r2 = self.client.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": critique_prompt}],
            temperature=0.3,
        )
        critique = r2.choices[0].message.content.strip()

        # Step 3: Refine (streamed)
        refine_system = (
            system_core
            + f'\n\nYour first draft was: "{draft}"'
            + f'\n\nCritique of that draft: "{critique}"'
            + "\n\nNow produce a refined, more authentic response that addresses the critique. "
            "Output only the final response — no labels or meta-commentary."
        )
        refine_messages = [{"role": "system", "content": refine_system}]
        refine_messages.extend(conversation_history)
        refine_messages.append({"role": "user", "content": user_input})
        stream = self.client.chat.completions.create(
            model=config.MODEL, messages=refine_messages, temperature=0.7, stream=True
        )

        meta = {
            "method": self.name,
            "draft": draft,
            "critique": critique,
            "rag_examples_used": len(rag_examples),
            "api_calls": 3,
        }
        return stream, meta
