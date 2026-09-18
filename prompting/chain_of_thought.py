from prompting.base import BasePromptMethod
import config


class ChainOfThoughtPrompt(BasePromptMethod):
    name = "chain_of_thought"
    description = (
        "CoT: model reasons step-by-step through persona psychology before producing the response."
    )
    uses_rag = True

    _COT_STEPS = """\
Before responding, silently work through these steps:
1. What is the salesperson trying to accomplish with this message?
2. How does {name}'s background, priorities, and personality shape their reaction?
3. What emotional state is {name} likely in right now given the conversation so far?
4. What key concerns or questions would {name} want to raise?
5. What tone should the response carry (skeptical, curious, pragmatic, guarded, enthusiastic)?
6. How long should this response be? Err toward the shorter end — real cold-call recipients don't explain themselves in paragraphs.

Then output ONLY the in-character response as {name}. Do not include the reasoning steps."""

    def build_messages(self, persona_info, rag_examples, conversation_history, user_input):
        name = persona_info.get("name", "the persona")
        system = self._build_system_core(persona_info)

        examples_block = self._format_examples_block(rag_examples, name)
        if examples_block:
            system += f"\n\nREFERENCE EXAMPLES (calibrate your tone from these):\n{examples_block}"

        system += f"\n\nINSTRUCTIONS:\n{self._COT_STEPS.format(name=name)}"

        messages = [{"role": "system", "content": system}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def generate(self, persona_info, rag_examples, conversation_history, user_input, **kwargs):
        # Slightly lower temperature keeps step-wise reasoning coherent
        return super().generate(
            persona_info, rag_examples, conversation_history, user_input,
            temperature=0.6, **kwargs
        )

    def generate_streamed(self, persona_info, rag_examples, conversation_history, user_input, **kwargs):
        return super().generate_streamed(
            persona_info, rag_examples, conversation_history, user_input,
            temperature=0.6, **kwargs
        )
