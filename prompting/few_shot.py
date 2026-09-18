from prompting.base import BasePromptMethod


class FewShotPrompt(BasePromptMethod):
    name = "few_shot"
    description = "RAG few-shot: retrieves similar examples from Pinecone and injects them as demonstrations."
    uses_rag = True

    def build_messages(self, persona_info, rag_examples, conversation_history, user_input):
        name = persona_info.get("name", "the persona")
        system = self._build_system_core(persona_info)

        examples_block = self._format_examples_block(rag_examples, name)
        if examples_block:
            system += (
                f"\n\n{examples_block}\n"
                "Your response must follow the vocabulary, length, and register of these examples. Do not elaborate beyond what they demonstrate."
            )

        messages = [{"role": "system", "content": system}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})
        return messages
