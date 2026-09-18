from prompting.base import BasePromptMethod


class BasicPrompt(BasePromptMethod):
    name = "basic"
    description = "Zero-shot: persona description only, no examples, no structured reasoning."
    uses_rag = False

    def build_messages(self, persona_info, rag_examples, conversation_history, user_input):
        system = self._build_system_core(persona_info)
        messages = [{"role": "system", "content": system}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})
        return messages
