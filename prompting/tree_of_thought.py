"""
Tree-of-Thought Negotiation prompting for persona simulation.

Process (2-3 API calls, down from 4):
  1. ANALYZE      — extract state from full history (turn 1) or diff-update from
                    the previous cached state (turn 2+).
  2. BRANCH+SELECT — generate N tactics, score each, and pick the winner in one
                    call. Skipped entirely on trivial turns (opening/exploring,
                    far/moderate distance, no prior commitments).
  3. GENERATE     — produce the final in-character response. Streamed in real-time
                    when called via generate_streamed().

Trivial turns collapse to 2 API calls; high-stakes turns use 3.
"""
import json
from prompting.base import BasePromptMethod
import config


class TreeOfThoughtPrompt(BasePromptMethod):
    name = "tree_of_thought"
    description = (
        "ToT Negotiation: analyzes deal state, generates & scores negotiation tactics, "
        "then streams the final response (2-3 API calls)."
    )
    uses_rag = True

    def __init__(self):
        super().__init__()
        self._last_state: dict | None = None

    # ------------------------------------------------------------------ #
    # Prompt templates
    # ------------------------------------------------------------------ #

    _ANALYZE_PROMPT = """\
You are a negotiation analyst. Study the persona brief and the full conversation \
history to extract the current negotiation state.

PERSONA: {name}
CHARACTER BRIEF:
{profile}

CONVERSATION SO FAR:
{history}

SALESPERSON'S LATEST MESSAGE: "{user_input}"

Return ONLY valid JSON — no commentary:
{{
  "negotiation_phase": "opening|exploring|bargaining|closing|stalled",
  "price_on_table": <number or null — most recent price quoted by the dealer>,
  "persona_target_price": <number or null — what the persona is trying to pay>,
  "trade_in_on_table": <number or null — dealer's current trade-in offer>,
  "persona_target_trade_in": <number or null — what the persona wants for trade-in>,
  "persona_posture": "open|skeptical|firm|cooling|walking",
  "prior_commitments": [<firm positions the persona has already stated, e.g. "won't sign today without wife seeing the car">],
  "seller_concessions": [<concessions the seller has made so far>],
  "persona_concessions": [<ground the persona has already given up>],
  "deal_distance": "far|moderate|close|very_close",
  "active_leverage": [<concrete leverage the persona currently holds, e.g. "competing quote $1,100 lower", "can sell trade-in privately">]
}}"""

    _UPDATE_STATE_PROMPT = """\
You are a negotiation analyst. Update the negotiation state based on the latest exchange.

PREVIOUS STATE:
{prev_state}

PERSONA'S LAST RESPONSE (what the persona said after that state was recorded):
"{last_persona_response}"

SALESPERSON'S NEW MESSAGE:
"{user_input}"

Update only the fields that changed. Return the complete updated state as ONLY valid JSON:
{{
  "negotiation_phase": "opening|exploring|bargaining|closing|stalled",
  "price_on_table": <number or null>,
  "persona_target_price": <number or null>,
  "trade_in_on_table": <number or null>,
  "persona_target_trade_in": <number or null>,
  "persona_posture": "open|skeptical|firm|cooling|walking",
  "prior_commitments": [...],
  "seller_concessions": [...],
  "persona_concessions": [...],
  "deal_distance": "far|moderate|close|very_close",
  "active_leverage": [...]
}}"""

    _BRANCH_AND_SELECT_PROMPT = """\
You are a negotiation strategist. Generate exactly {n} distinct tactics for {name}'s \
next response, score each one, and identify the winner.

PERSONA: {name}
CHARACTER BRIEF:
{profile}

NEGOTIATION STATE:
{state}

SALESPERSON SAID: "{user_input}"

Each tactic must be a genuinely different move. Draw from this repertoire:
  push_harder       — anchor to research/data, demand a better number, cite competing offers
  soft_push         — hint at alternatives or hesitation without a direct ultimatum
  conditional_accept — agree IF a specific condition is met (price, trade-in, add-on removed, etc.)
  backtrack         — soften a prior firm position because the seller has moved enough
  walk_away_signal  — clearly signal readiness to leave; still in the room, waiting for a response
  walk_out          — end the conversation; only when a firm line has been crossed or a walk_away_signal was ignored
  genuine_interest  — show real enthusiasm for a feature while noting remaining concerns
  accept            — agree to current terms because they are genuinely satisfactory

Score each tactic 1–10 across:
  1. Authenticity  — matches persona's voice, values, and decision-making style
  2. Consistency   — respects prior commitments and the conversation arc so far
  3. Fit           — right move for this phase and deal distance
  4. Goal progress — advances the persona's actual objectives

Penalise tactics that:
  - Suddenly capitulate without a justifying concession from the seller
  - Contradict a firm prior commitment with no explanation
  - Use walk_out without a prior walk_away_signal, or when the deal is close
  - Use walk_away_signal or walk_out when the seller just made a meaningful concession

Return ONLY valid JSON:
{{"branches": [
  {{
    "id": 1,
    "tactic": "<tactic name from list above>",
    "label": "<3-5 word label>",
    "approach": "<how to execute — specific, not generic>",
    "tone": "<emotional quality>",
    "what_persona_concedes": "<ground given up, or 'nothing'>",
    "what_persona_demands": "<what is asked in return, or 'nothing'>",
    "score": <1-10>,
    "score_reason": "<one sentence>"
  }}
],
"winner_id": <id of the highest-scoring branch>
}}"""

    # ------------------------------------------------------------------ #
    # BasePromptMethod interface
    # ------------------------------------------------------------------ #

    def build_messages(self, persona_info, rag_examples, conversation_history, user_input):
        raise NotImplementedError("TreeOfThoughtPrompt uses generate() directly.")

    def generate(self, persona_info, rag_examples, conversation_history, user_input, **kwargs):
        messages, meta = self._prepare_generation(
            persona_info, rag_examples, conversation_history, user_input
        )
        resp = self.client.chat.completions.create(
            model=config.MODEL, messages=messages, temperature=0.7
        )
        text = resp.choices[0].message.content.strip()
        meta["api_calls"] += 1
        return text, meta

    def generate_streamed(self, persona_info, rag_examples, conversation_history, user_input, **kwargs):
        """Prep calls run synchronously; returns the live stream for the final response."""
        messages, meta = self._prepare_generation(
            persona_info, rag_examples, conversation_history, user_input
        )
        stream = self.client.chat.completions.create(
            model=config.MODEL, messages=messages, temperature=0.7, stream=True
        )
        meta["api_calls"] += 1
        return stream, meta

    # ------------------------------------------------------------------ #
    # Core preparation (shared by generate and generate_streamed)
    # ------------------------------------------------------------------ #

    def _prepare_generation(
        self,
        persona_info: dict,
        rag_examples: list[dict],
        conversation_history: list[dict],
        user_input: str,
    ) -> tuple[list[dict], dict]:
        name = persona_info.get("name", "the persona")
        profile = persona_info.get("role_prompt") or self._format_persona_block(persona_info)

        last_persona_resp = next(
            (m["content"] for m in reversed(conversation_history) if m["role"] == "assistant"),
            None,
        )

        history_text = self._format_history(conversation_history)
        neg_state = self._analyze_state(
            name, profile, history_text, user_input,
            prev_state=self._last_state,
            last_persona_response=last_persona_resp,
        )
        self._last_state = neg_state

        if self._is_trivial_turn(neg_state):
            best = {
                "tactic": "soft_push",
                "label": "measured response",
                "approach": (
                    "acknowledge their point and respond naturally; "
                    "hint at remaining concerns without pressure"
                ),
                "tone": "calm and direct",
                "what_persona_concedes": "nothing",
                "what_persona_demands": "nothing",
            }
            n_branches = 0
            prep_calls = 1
        else:
            n = self._n_branches(neg_state)
            best, branches = self._branch_and_select(name, profile, neg_state, user_input, n)
            n_branches = len(branches)
            prep_calls = 2

        examples_block = self._format_examples_block(rag_examples, name)
        system = self._build_system_core(persona_info)
        if examples_block:
            system += f"\n\n{examples_block}"
        system += self._build_negotiation_context(neg_state, best)

        messages = [{"role": "system", "content": system}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})

        meta = {
            "method": self.name,
            "negotiation_phase": neg_state.get("negotiation_phase"),
            "deal_distance": neg_state.get("deal_distance"),
            "persona_posture": neg_state.get("persona_posture"),
            "selected_tactic": best.get("tactic"),
            "selected_branch": best,
            "branches_generated": n_branches,
            "rag_examples_used": len(rag_examples),
            "api_calls": prep_calls,
            "walked_out": best.get("tactic") == "walk_out",
        }
        return messages, meta

    # ------------------------------------------------------------------ #
    # Step implementations
    # ------------------------------------------------------------------ #

    def _analyze_state(
        self,
        name: str,
        profile: str,
        history_text: str,
        user_input: str,
        prev_state: dict | None = None,
        last_persona_response: str | None = None,
    ) -> dict:
        if prev_state is not None and last_persona_response is not None:
            prompt = self._UPDATE_STATE_PROMPT.format(
                prev_state=json.dumps(prev_state, indent=2),
                last_persona_response=last_persona_response,
                user_input=user_input,
            )
        else:
            prompt = self._ANALYZE_PROMPT.format(
                name=name,
                profile=profile,
                history=history_text or "(conversation just started)",
                user_input=user_input,
            )
        resp = self.client.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, KeyError):
            return prev_state or {
                "negotiation_phase": "opening",
                "deal_distance": "far",
                "persona_posture": "open",
                "prior_commitments": [],
                "seller_concessions": [],
                "persona_concessions": [],
                "active_leverage": [],
            }

    def _branch_and_select(
        self,
        name: str,
        profile: str,
        neg_state: dict,
        user_input: str,
        n: int,
    ) -> tuple[dict, list[dict]]:
        state_text = self._state_to_text(neg_state)
        prompt = self._BRANCH_AND_SELECT_PROMPT.format(
            n=n,
            name=name,
            profile=profile,
            state=state_text,
            user_input=user_input,
        )
        resp = self.client.chat.completions.create(
            model=config.MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            branches = data.get("branches", [])
            winner_id = data.get("winner_id")
            if branches:
                if winner_id is not None:
                    for b in branches:
                        if b.get("id") == winner_id:
                            return b, branches
                return max(branches, key=lambda b: b.get("score", 0)), branches
        except (json.JSONDecodeError, KeyError):
            pass
        fallback = {
            "tactic": "soft_push",
            "label": "measured response",
            "approach": "respond naturally, reference research if relevant",
            "tone": "calm and direct",
            "what_persona_concedes": "nothing",
            "what_persona_demands": "nothing",
        }
        return fallback, [fallback]

    # ------------------------------------------------------------------ #
    # Classification helpers
    # ------------------------------------------------------------------ #

    def _is_trivial_turn(self, neg_state: dict) -> bool:
        return (
            neg_state.get("negotiation_phase") in ("opening", "exploring")
            and neg_state.get("deal_distance") in ("far", "moderate")
            and neg_state.get("persona_posture") in ("open", "skeptical")
            and not neg_state.get("prior_commitments")
        )

    def _n_branches(self, neg_state: dict) -> int:
        if (
            neg_state.get("deal_distance") in ("very_close", "close")
            or neg_state.get("persona_posture") in ("walking", "cooling")
        ):
            return 4
        return 3

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _format_history(self, conversation_history: list[dict]) -> str:
        if not conversation_history:
            return ""
        lines = []
        for msg in conversation_history:
            role = "SALESPERSON" if msg["role"] == "user" else "PERSONA"
            lines.append(f'{role}: "{msg["content"]}"')
        return "\n".join(lines)

    def _state_to_text(self, state: dict) -> str:
        lines = [
            f"Phase: {state.get('negotiation_phase', 'unknown')}",
            f"Deal distance: {state.get('deal_distance', 'unknown')}",
            f"Persona posture: {state.get('persona_posture', 'unknown')}",
        ]
        if state.get("price_on_table"):
            lines.append(f"Price on table: ${state['price_on_table']:,}")
        if state.get("persona_target_price"):
            lines.append(f"Persona target price: ${state['persona_target_price']:,}")
        if state.get("trade_in_on_table"):
            lines.append(f"Trade-in offered: ${state['trade_in_on_table']:,}")
        if state.get("persona_target_trade_in"):
            lines.append(f"Persona target trade-in: ${state['persona_target_trade_in']:,}")
        if state.get("prior_commitments"):
            lines.append("Prior commitments: " + "; ".join(str(x) for x in state["prior_commitments"]))
        if state.get("seller_concessions"):
            lines.append("Seller has conceded: " + "; ".join(str(x) for x in state["seller_concessions"]))
        if state.get("persona_concessions"):
            lines.append("Persona has conceded: " + "; ".join(str(x) for x in state["persona_concessions"]))
        if state.get("active_leverage"):
            lines.append("Active leverage: " + "; ".join(str(x) for x in state["active_leverage"]))
        return "\n".join(lines)

    def _build_negotiation_context(self, neg_state: dict, best_branch: dict) -> str:
        tactic = best_branch.get("tactic", "soft_push")
        label = best_branch.get("label", "natural")
        approach = best_branch.get("approach", "respond naturally")
        tone = best_branch.get("tone", "conversational")
        concedes = best_branch.get("what_persona_concedes", "nothing")
        demands = best_branch.get("what_persona_demands", "nothing")

        tactic_guidance = {
            "push_harder": (
                "You are not satisfied with the current terms. Anchor back to your research "
                "or a competing offer. Name a specific number or condition and hold it. "
                "Do not soften your position — be direct but not aggressive."
            ),
            "soft_push": (
                "You have concerns but aren't ready to draw a hard line yet. Hint at "
                "alternatives (other dealers, selling trade-in privately) without issuing "
                "an ultimatum. Show mild hesitation."
            ),
            "conditional_accept": (
                "You are willing to move forward IF — and only if — a specific condition "
                "is met. State the condition clearly and precisely. Make it clear this is "
                "your threshold, not a negotiating opener."
            ),
            "backtrack": (
                "The seller has moved enough to warrant softening a position you held earlier. "
                "Acknowledge the movement they have made, then revise your prior stance "
                "in a measured way. Don't completely cave — link the backtrack to what "
                "they have conceded."
            ),
            "walk_away_signal": (
                "You are genuinely prepared to leave but you are giving the dealer one last "
                "chance. Name the specific issue clearly. Stay calm and factual — no anger, "
                "no ultimatum theatre. You are still in the room, but make it plain that "
                "without a resolution you will not be."
            ),
            "walk_out": (
                "You are leaving. The line has been crossed — whether that is a firm ADM, "
                "repeated dishonesty, an insulting number, or a walk_away_signal that was "
                "ignored. Your response is a clean, composed farewell: thank them briefly "
                "for their time, state the single reason the deal did not work (one sentence, "
                "factual, no heat), and make clear you are done. Do not issue any further "
                "conditions or openings. This is the end of the conversation."
            ),
            "genuine_interest": (
                "Something the seller said or showed has actually impressed you. Let a "
                "brief moment of authentic enthusiasm show before pivoting back to your "
                "remaining practical concerns. Don't overdo it — one sentence of genuine "
                "reaction, then back to specifics."
            ),
            "accept": (
                "The terms are genuinely satisfactory. Accept clearly but without excessive "
                "enthusiasm. Note any next steps (final paperwork review, etc.) to stay "
                "consistent with your prior commitments."
            ),
        }.get(tactic, "Respond naturally and in character.")

        block = (
            f"\n\nNEGOTIATION CONTEXT:\n"
            f"Current phase: {neg_state.get('negotiation_phase', 'unknown')} | "
            f"Deal distance: {neg_state.get('deal_distance', 'unknown')} | "
            f"Your posture: {neg_state.get('persona_posture', 'open')}\n\n"
            f"CHOSEN NEGOTIATION MOVE: {label} (tactic: {tactic})\n"
            f"Approach: {approach}\n"
            f"Tone: {tone}\n"
            f"What you concede this turn: {concedes}\n"
            f"What you demand in return: {demands}\n\n"
            f"EXECUTION GUIDANCE:\n{tactic_guidance}\n\n"
        )

        if neg_state.get("prior_commitments"):
            block += (
                "PRIOR COMMITMENTS TO HONOUR:\n"
                + "\n".join(f"- {c}" for c in neg_state["prior_commitments"])
                + "\n\n"
            )

        if neg_state.get("active_leverage"):
            block += (
                "LEVERAGE YOU CAN USE IF RELEVANT:\n"
                + "\n".join(f"- {lv}" for lv in neg_state["active_leverage"])
                + "\n\n"
            )

        block += (
            "Output only the in-character response — no labels, no meta-commentary, "
            "no stage directions."
        )
        return block
