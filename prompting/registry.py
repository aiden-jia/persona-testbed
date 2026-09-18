from prompting.basic import BasicPrompt
from prompting.few_shot import FewShotPrompt
from prompting.chain_of_thought import ChainOfThoughtPrompt
from prompting.tree_of_thought import TreeOfThoughtPrompt
from prompting.tree_of_thought_coldcall import TreeOfThoughtColdCallPrompt
from prompting.reflection import ReflectionPrompt
from prompting.base import BasePromptMethod

# Ordered dict — insertion order defines display order in menus.
METHODS: dict[str, BasePromptMethod] = {
    "basic": BasicPrompt(),
    "few_shot": FewShotPrompt(),
    "chain_of_thought": ChainOfThoughtPrompt(),
    "tree_of_thought": TreeOfThoughtPrompt(),
    "tree_of_thought_coldcall": TreeOfThoughtColdCallPrompt(),
    "reflection": ReflectionPrompt(),
}


def get_method(name: str) -> BasePromptMethod:
    if name not in METHODS:
        raise ValueError(
            f"Unknown method '{name}'. Available: {list(METHODS.keys())}"
        )
    return METHODS[name]


def list_methods() -> list[dict]:
    return [
        {"name": m.name, "description": m.description, "uses_rag": m.uses_rag}
        for m in METHODS.values()
    ]
