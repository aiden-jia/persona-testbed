"""
promptfoo custom Python provider — Persona Prompting Testbed simulation.

For each (method, scenario) pair promptfoo calls call_api() once.
The provider runs a complete two-sided conversation internally:
  - Persona side  : one of the 5 prompting methods from may_term/
  - Salesperson   : gpt-4o-mini with a dealership sales system prompt

Returns the full transcript + per-turn stats as structured output so promptfoo's
LLM-rubric evaluators can grade the conversation.

may_term/ is never modified — only imported via sys.path injection.
"""
import sys
import json
import time
from pathlib import Path

_THIS_DIR = Path(__file__).parent
_PARENT = _THIS_DIR.parent

for _p in [str(_THIS_DIR), str(_PARENT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_persona_info(persona_name: str) -> dict:
    """Load persona from Pinecone, with local .txt fallback."""
    import config
    from upload_persona import parse_persona_file

    try:
        from rag.retriever import retrieve_persona_info
        info = retrieve_persona_info(persona_name)
        if info:
            return info
    except Exception:
        pass

    slug = persona_name.lower().replace(" ", "_")
    for path in [
        config.PERSONAS_DIR / f"{slug}.txt",
        config.PERSONAS_DIR / f"{persona_name}.txt",
    ]:
        if path.exists():
            info, _ = parse_persona_file(path)
            return info

    return {"name": persona_name}


def _get_rag_examples(
    salesperson_msg: str,
    persona_history: list[dict],
    method,
    persona_name: str,
    persona_info: dict = None,
) -> list[dict]:
    """Retrieve RAG examples using the same query construction as chat.py."""
    if not method.uses_rag:
        return []
    try:
        from rag.retriever import retrieve_examples, retrieve_cold_call_examples
        if (persona_info or {}).get("rag_namespace") == "cold_calls":
            return retrieve_cold_call_examples(salesperson_msg)
        query = salesperson_msg
        if persona_history:
            recent = persona_history[-6:]
            lines = []
            for m in recent:
                prefix = "SALESPERSON:" if m["role"] == "user" else "PERSONA:"
                lines.append(f"{prefix} {m['content']}")
            query = "\n".join(lines) + "\n" + salesperson_msg
        return retrieve_examples(query, persona_name)
    except Exception as e:
        print(f"[RAG error] {type(e).__name__}: {e}", flush=True)
        return []


def _extract_tokens(meta: dict) -> dict:
    """Normalize token counts across methods.

    basic/few_shot/chain_of_thought: prompt_tokens + completion_tokens
    reflection: total_tokens (sum of 3 calls, no breakdown)
    tree_of_thought: no token tracking in meta
    """
    prompt = meta.get("prompt_tokens", 0)
    completion = meta.get("completion_tokens", 0)
    total = meta.get("total_tokens", prompt + completion)
    return {"total": total, "prompt": prompt, "completion": completion}


def _detect_objection(response_text: str, client, model: str) -> bool:
    """Return True if the persona raised a meaningful objection or pushback."""
    prompt = (
        f'Did the persona in this response raise a clear objection, pushback, or resistance '
        f'(e.g. "we already have something", "bad timing", "not interested", "I\'m busy", '
        f'"send me an email", "who is this", or expressing skepticism)?\n\n'
        f'PERSONA\'S RESPONSE: "{response_text}"\n\n'
        "Answer YES if they expressed hesitation, skepticism, a competing solution, or a reason not to continue.\n"
        "Answer NO if they are simply asking a clarifying question or engaging neutrally.\n"
        "Answer with ONLY the word 'yes' or 'no'."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        return resp.choices[0].message.content.strip().lower().startswith("yes")
    except Exception:
        return False


def _detect_deal_closed(response_text: str, client, model: str, criteria: str = None) -> bool:
    """Return True if the persona unambiguously accepted the deal.

    Pass `criteria` to override the default car-dealership framing for other
    scenario types (e.g. 'agreed to a discovery call' for cold calls).
    """
    condition = criteria or "accepted the deal"
    prompt = (
        f'Did the persona definitively and unconditionally commit to the following in this response: {condition}?\n\n'
        f'PERSONA\'S RESPONSE: "{response_text}"\n\n'
        "Answer YES only if they clearly committed with no conditions attached.\n"
        "Answer NO if the response:\n"
        "- Contains any condition ('if', 'as long as', 'provided that', 'once', 'assuming')\n"
        "- Is still asking questions or requesting information before deciding\n"
        "- Expresses willingness or readiness without an explicit commitment\n"
        "- Uses hedging language ('I think', 'I could see', 'maybe', 'probably')\n"
        "Answer with ONLY the word 'yes' or 'no'."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        return resp.choices[0].message.content.strip().lower().startswith("yes")
    except Exception:
        return False


def _format_transcript(transcript: list[dict], method_name: str) -> str:
    lines = [f"=== SIMULATED CONVERSATION ({method_name.upper()}) ===", ""]
    for entry in transcript:
        seq = entry["seq"]
        role = entry["role"]
        content = entry["content"]
        label = "SALESPERSON" if role == "salesperson" else f"PERSONA ({method_name})"
        lines.append(f"[{seq}] {label}:")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def _format_stats_block(stats: dict) -> str:
    outcome = (
        "WALKED OUT" if stats["walked_out"]
        else "DEAL CLOSED" if stats["deal_closed"]
        else "MAX TURNS REACHED"
    )
    lines = [
        "=== SIMULATION STATS ===",
        f"Method              : {stats['method']}",
        f"Persona             : {stats['persona']}",
        f"Scenario            : {stats['scenario']}",
        f"Persona turns       : {stats['persona_turns']}",
        f"Salesperson turns   : {stats['salesperson_turns']}",
        f"Total tokens        : {stats['total_tokens']}",
        f"Total API calls     : {stats['total_api_calls']}",
        f"Total time          : {stats['total_time_s']:.2f}s",
        f"Avg persona turn    : {stats['avg_time_per_persona_turn']:.2f}s",
        f"Outcome             : {outcome}",
    ]
    if stats.get("walkout_turn"):
        lines.append(f"Walkout at turn     : {stats['walkout_turn']}")
    if stats.get("close_turn"):
        lines.append(f"Deal closed at turn : {stats['close_turn']}")

    lines += [
        "",
        "--- PER PERSONA TURN ---",
        f"{'Turn':>4}  {'Calls':>5}  {'Tokens':>7}  {'Time(s)':>8}  {'RAG':>4}  {'Phase':<12}  {'Tactic':<18}  Outcome",
    ]
    for t in stats["per_turn"]:
        outcome = "walked_out" if t.get("walked_out") else "deal_closed" if t.get("deal_closed") else ""
        phase = t.get("negotiation_phase") or ""
        tactic = t.get("selected_tactic") or ""
        lines.append(
            f"{t['turn']:>4}  {t['api_calls']:>5}  {t['tokens']['total']:>7}  "
            f"{t['time_s']:>8.3f}  {t['rag_examples_used']:>4}  {phase:<12}  {tactic:<18}  {outcome}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# promptfoo entry point
# ---------------------------------------------------------------------------

def call_api(prompt: str, options: dict, context: dict) -> dict:
    """Run one complete two-sided conversation and return transcript + stats."""
    import config
    from salesperson import generate_salesperson_response
    from prompting.basic import BasicPrompt
    from prompting.few_shot import FewShotPrompt
    from prompting.chain_of_thought import ChainOfThoughtPrompt
    from prompting.tree_of_thought import TreeOfThoughtPrompt
    from prompting.tree_of_thought_coldcall import TreeOfThoughtColdCallPrompt
    from prompting.reflection import ReflectionPrompt

    METHOD_CLASSES = {
        "basic": BasicPrompt,
        "few_shot": FewShotPrompt,
        "chain_of_thought": ChainOfThoughtPrompt,
        "tree_of_thought": TreeOfThoughtPrompt,
        "tree_of_thought_coldcall": TreeOfThoughtColdCallPrompt,
        "reflection": ReflectionPrompt,
    }

    # --- Read config ---
    provider_config = options.get("config", {})
    method_name = provider_config.get("method", "basic")

    vars_ = context.get("vars", {})
    persona_name = vars_.get("persona_name", "Car Buyer")
    scenario_id = vars_.get("scenario_id", "standard_negotiation")

    if method_name not in METHOD_CLASSES:
        return {"error": f"Unknown method '{method_name}'. Options: {list(METHOD_CLASSES)}"}

    # --- Load scenario ---
    scenarios_path = _THIS_DIR / "scenarios.json"
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)
    scenario = next((s for s in scenarios if s["id"] == scenario_id), scenarios[0])

    # Scenario-level overrides take precedence over provider config defaults.
    max_turns_each = scenario.get("max_turns_each", provider_config.get("max_turns_each", 30))
    deal_closed_criteria = scenario.get("deal_closed_criteria")
    min_objections_before_walkout = scenario.get("min_objections_before_walkout", 0)

    # --- Load persona info ---
    persona_info = _load_persona_info(persona_name)
    # Use the canonical name from the persona file for RAG namespace lookups,
    # not the generic label from the YAML vars (e.g. "Alex Chen" not "Car Buyer").
    rag_persona_name = persona_info.get("name", persona_name)

    # Fresh method instance per conversation (ensures TreeOfThought._last_state is clean)
    method = METHOD_CLASSES[method_name]()

    # --- Conversation state ---
    # persona_history: OpenAI format from persona's POV (user=salesperson, assistant=persona)
    # salesperson_history: OpenAI format from salesperson's POV (user=persona, assistant=salesperson)
    persona_history: list[dict] = []
    salesperson_history: list[dict] = []
    transcript: list[dict] = []
    seq = 1  # sequential message index across both sides

    stats = {
        "method": method_name,
        "persona": persona_name,
        "scenario": scenario_id,
        "persona_turns": 0,
        "salesperson_turns": 1,
        "total_tokens": 0,
        "total_api_calls": 0,
        "total_time_s": 0.0,
        "avg_time_per_persona_turn": 0.0,
        "walked_out": False,
        "deal_closed": False,
        "walkout_turn": None,
        "close_turn": None,
        "per_turn": [],
    }

    # --- Opening salesperson message (from scenario — no API call) ---
    salesperson_msg = scenario["salesperson_opening"]
    salesperson_history.append({"role": "assistant", "content": salesperson_msg})
    transcript.append({"seq": seq, "role": "salesperson", "content": salesperson_msg})
    seq += 1

    objections_raised = 0

    # --- Main loop ---
    for persona_turn in range(1, max_turns_each + 1):

        # ===== PERSONA TURN =====
        rag_examples = _get_rag_examples(salesperson_msg, persona_history, method, rag_persona_name, persona_info)
        history_snapshot = list(persona_history)

        t0 = time.perf_counter()
        try:
            persona_response, meta = method.generate(
                persona_info=persona_info,
                rag_examples=rag_examples,
                conversation_history=history_snapshot,
                user_input=salesperson_msg,
            )
        except Exception as e:
            return {"error": f"Persona generation failed (turn {persona_turn}): {e}"}
        elapsed = time.perf_counter() - t0

        # Check deal_closed first — farewell language in an acceptance response
        # would otherwise false-trigger the walkout detector.
        tactic_walkout = meta.get("walked_out", False)
        deal_closed = False
        if not tactic_walkout:
            deal_closed = _detect_deal_closed(persona_response, method.client, config.MODEL, deal_closed_criteria)

        # Only run walkout detection if no deal was closed.
        walked_out = False
        if not deal_closed:
            walked_out = tactic_walkout or method.detect_walkout(persona_response, persona_info)

        # Track objections before applying the walk-out floor so the departing
        # turn itself counts toward the threshold.
        if not deal_closed and min_objections_before_walkout > 0:
            if _detect_objection(persona_response, method.client, config.MODEL):
                objections_raised += 1
            # Suppress the walk-out until the persona has raised enough resistance
            # naturally — prevents premature exits on weak openers.
            if walked_out and objections_raised < min_objections_before_walkout:
                walked_out = False

        tok = _extract_tokens(meta)
        turn_stat = {
            "turn": persona_turn,
            "api_calls": meta.get("api_calls", 1),
            "tokens": tok,
            "time_s": round(elapsed, 3),
            "rag_examples_used": meta.get("rag_examples_used", 0),
            "negotiation_phase": meta.get("negotiation_phase"),
            "selected_tactic": meta.get("selected_tactic"),
            "walked_out": walked_out,
            "deal_closed": deal_closed,
        }
        stats["per_turn"].append(turn_stat)
        stats["total_tokens"] += tok["total"]
        stats["total_api_calls"] += meta.get("api_calls", 1)
        stats["total_time_s"] += elapsed
        stats["persona_turns"] = persona_turn

        # Update histories
        persona_history.append({"role": "user", "content": salesperson_msg})
        persona_history.append({"role": "assistant", "content": persona_response})
        salesperson_history.append({"role": "user", "content": persona_response})

        transcript.append({"seq": seq, "role": "persona", "content": persona_response})
        seq += 1

        if walked_out:
            stats["walked_out"] = True
            stats["walkout_turn"] = persona_turn
            break

        if deal_closed:
            stats["deal_closed"] = True
            stats["close_turn"] = persona_turn
            break

        if persona_turn >= max_turns_each:
            break

        # ===== SALESPERSON TURN =====
        try:
            sales_response, sales_meta = generate_salesperson_response(
                system_prompt=scenario["salesperson_system_prompt"],
                conversation_history=salesperson_history,
            )
        except Exception as e:
            return {"error": f"Salesperson generation failed (turn {persona_turn}): {e}"}

        salesperson_history.append({"role": "assistant", "content": sales_response})
        salesperson_msg = sales_response
        stats["salesperson_turns"] += 1
        stats["total_tokens"] += (
            sales_meta.get("prompt_tokens", 0) + sales_meta.get("completion_tokens", 0)
        )
        stats["total_api_calls"] += 1

        transcript.append({"seq": seq, "role": "salesperson", "content": sales_response})
        seq += 1

    # --- Finalize stats ---
    if stats["persona_turns"] > 0:
        stats["avg_time_per_persona_turn"] = stats["total_time_s"] / stats["persona_turns"]

    # --- Build output ---
    transcript_text = _format_transcript(transcript, method_name)
    stats_text = _format_stats_block(stats)
    output = transcript_text + "\n\n" + stats_text

    return {
        "output": output,
        "tokenUsage": {
            "total": stats["total_tokens"],
            "prompt": sum(t["tokens"]["prompt"] for t in stats["per_turn"]),
            "completion": sum(t["tokens"]["completion"] for t in stats["per_turn"]),
        },
        "metadata": stats,
    }
