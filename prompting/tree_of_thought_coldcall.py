"""
Tree-of-Thought Cold Call prompting for the recipient persona.

Process (2-3 API calls, mirroring tree_of_thought.py):
  1. ANALYZE      — extract call reception state from full history (turn 1) or
                    diff-update from the previous cached state (turn 2+).
  2. BRANCH+SELECT — generate N response tactics, score each, and pick the winner
                    in one call. Skipped on trivial turns (opener, not-yet-pitched,
                    neutral/curious engagement).
  3. GENERATE     — produce the final in-character response. Streamed in real-time
                    when called via generate_streamed().

Trivial turns collapse to 2 API calls; high-stakes turns use 3.
"""
import json
from prompting.base import BasePromptMethod
import config


class TreeOfThoughtColdCallPrompt(BasePromptMethod):
    name = "tree_of_thought_coldcall"
    description = (
        "ToT Cold Call: analyzes call reception state, generates & scores response tactics, "
        "then streams the final reply (2-3 API calls)."
    )
    uses_rag = True
    _ANALYSIS_MODEL = "gpt-5.4"

    def __init__(self):
        super().__init__()
        self._last_state: dict | None = None

    # ------------------------------------------------------------------ #
    # Prompt templates
    # ------------------------------------------------------------------ #

    _ANALYZE_PROMPT = """\
You are a conversation analyst. Study the persona brief and the full call history \
to extract the current cold call reception state.

PERSONA: {name}
CHARACTER BRIEF:
{profile}

CONVERSATION SO FAR:
{history}

CALLER'S LATEST MESSAGE: "{user_input}"

Return ONLY valid JSON — no commentary:
{{
  "call_phase": "opener|pitch|objection|recovery|close_attempt",
  "recipient_engagement": "curious|skeptical|flat|interested|checking_out",
  "value_clarity": "not_pitched_yet|vague|somewhat_clear|clear",
  "caller_specificity": "generic|somewhat_specific|highly_specific",
  "prior_objections": [<objections or pushbacks the recipient has already raised>],
  "caller_has_addressed": [<prior concerns the caller has since addressed>],
  "call_feel": "fresh|going_on|overstaying",
  "key_hook": <the single most specific or relevant thing the caller has said so far, or null>
}}"""

    _UPDATE_STATE_PROMPT = """\
You are a conversation analyst. Update the call reception state based on the latest exchange.

PREVIOUS STATE:
{prev_state}

RECIPIENT'S LAST RESPONSE (what the recipient said after that state was recorded):
"{last_persona_response}"

CALLER'S NEW MESSAGE:
"{user_input}"

Update only the fields that changed. Return the complete updated state as ONLY valid JSON:
{{
  "call_phase": "opener|pitch|objection|recovery|close_attempt",
  "recipient_engagement": "curious|skeptical|flat|interested|checking_out",
  "value_clarity": "not_pitched_yet|vague|somewhat_clear|clear",
  "caller_specificity": "generic|somewhat_specific|highly_specific",
  "prior_objections": [...],
  "caller_has_addressed": [...],
  "call_feel": "fresh|going_on|overstaying",
  "key_hook": <string or null>
}}"""

    _BRANCH_AND_SELECT_PROMPT = """\
You are a conversation strategist. Generate exactly {n} distinct response tactics for {name}'s \
next reply, score each one, and identify the winner.

PERSONA: {name}
CHARACTER BRIEF:
{profile}

CALL RECEPTION STATE:
{state}

CALLER JUST SAID: "{user_input}"

Each tactic must be a genuinely different move. Draw from this repertoire:
  engage              — show brief, genuine interest; ask a follow-up question
  short_redirect      — give a polite but brief pushback without ending the call
  clarifying_question — ask for specifics on something the caller said
  skeptical_probe     — challenge a vague claim, generic assumption, or scripted line
  polite_stall        — express mild openness without committing; buy time
  booking_signal      — indicate you might be open to a next step or follow-up
  wrap_up             — end the call cleanly because it's not a fit

Score each tactic 1–10 across:
  1. Authenticity  — matches how a real busy professional would actually respond
  2. Consistency   — respects prior objections and the conversation arc so far
  3. Fit           — right move for this call phase and engagement level
  4. Realism       — natural and unpolished; not over-explained or over-thought

Penalise tactics that:
  - Suddenly become warm or interested when the caller hasn't earned it
  - Contradict a prior objection without the caller addressing it
  - Use wrap_up too early (before any pitch has been made) or too late (after clear engagement)
  - Produce a long, structured reply when the moment calls for brevity
  - Sound like an AI — any tactic whose execution would feel scripted or formal

Return ONLY valid JSON:
{{"branches": [
  {{
    "id": 1,
    "tactic": "<tactic name from list above>",
    "label": "<3-5 word label>",
    "approach": "<how to execute — specific, not generic>",
    "tone": "<emotional quality>",
    "what_recipient_concedes": "<ground given up, or 'nothing'>",
    "what_recipient_signals": "<what this response communicates to the caller>",
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
        raise NotImplementedError("TreeOfThoughtColdCallPrompt uses generate() directly.")

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
        call_state = self._analyze_state(
            name, profile, history_text, user_input,
            prev_state=self._last_state,
            last_persona_response=last_persona_resp,
        )
        self._last_state = call_state

        if self._is_trivial_turn(call_state):
            best = {
                "tactic": "short_redirect",
                "label": "natural opener reply",
                "approach": (
                    "respond briefly and naturally to what was just said; "
                    "stay non-committal but not cold"
                ),
                "tone": "neutral and slightly distracted",
                "what_recipient_concedes": "nothing",
                "what_recipient_signals": "still listening, not yet engaged",
            }
            n_branches = 0
            prep_calls = 1
        else:
            n = self._n_branches(call_state)
            best, branches = self._branch_and_select(name, profile, call_state, user_input, n)
            n_branches = len(branches)
            prep_calls = 2

        examples_block = self._format_examples_block(rag_examples, name)
        system = self._build_system_core(persona_info)
        if examples_block:
            system += f"\n\n{examples_block}"
        system += self._build_call_context(call_state, best)

        messages = [{"role": "system", "content": system}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})

        meta = {
            "method": self.name,
            "call_phase": call_state.get("call_phase"),
            "recipient_engagement": call_state.get("recipient_engagement"),
            "call_feel": call_state.get("call_feel"),
            "selected_tactic": best.get("tactic"),
            "selected_branch": best,
            "branches_generated": n_branches,
            "rag_examples_used": len(rag_examples),
            "api_calls": prep_calls,
            "walked_out": best.get("tactic") == "wrap_up",
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
                history=history_text or "(call just started)",
                user_input=user_input,
            )
        resp = self.client.chat.completions.create(
            model=self._ANALYSIS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, KeyError):
            return prev_state or {
                "call_phase": "opener",
                "recipient_engagement": "skeptical",
                "value_clarity": "not_pitched_yet",
                "caller_specificity": "generic",
                "prior_objections": [],
                "caller_has_addressed": [],
                "call_feel": "fresh",
                "key_hook": None,
            }

    def _branch_and_select(
        self,
        name: str,
        profile: str,
        call_state: dict,
        user_input: str,
        n: int,
    ) -> tuple[dict, list[dict]]:
        state_text = self._state_to_text(call_state)
        prompt = self._BRANCH_AND_SELECT_PROMPT.format(
            n=n,
            name=name,
            profile=profile,
            state=state_text,
            user_input=user_input,
        )
        resp = self.client.chat.completions.create(
            model=self._ANALYSIS_MODEL,
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
            "tactic": "short_redirect",
            "label": "neutral reply",
            "approach": "respond briefly; neither warm nor cold",
            "tone": "neutral",
            "what_recipient_concedes": "nothing",
            "what_recipient_signals": "still listening",
        }
        return fallback, [fallback]

    # ------------------------------------------------------------------ #
    # Classification helpers
    # ------------------------------------------------------------------ #

    def _is_trivial_turn(self, call_state: dict) -> bool:
        return (
            call_state.get("call_phase") == "opener"
            and call_state.get("value_clarity") == "not_pitched_yet"
            and call_state.get("recipient_engagement") in ("curious", "skeptical", "flat")
        )

    def _n_branches(self, call_state: dict) -> int:
        if (
            call_state.get("call_phase") in ("close_attempt", "recovery")
            or call_state.get("recipient_engagement") in ("interested", "checking_out")
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
            role = "CALLER" if msg["role"] == "user" else "RECIPIENT"
            lines.append(f'{role}: "{msg["content"]}"')
        return "\n".join(lines)

    def _state_to_text(self, state: dict) -> str:
        lines = [
            f"Phase: {state.get('call_phase', 'unknown')}",
            f"Recipient engagement: {state.get('recipient_engagement', 'unknown')}",
            f"Value clarity: {state.get('value_clarity', 'unknown')}",
            f"Caller specificity: {state.get('caller_specificity', 'unknown')}",
            f"Call feel: {state.get('call_feel', 'unknown')}",
        ]
        if state.get("key_hook"):
            lines.append(f"Key hook so far: {state['key_hook']}")
        if state.get("prior_objections"):
            lines.append("Prior objections raised: " + "; ".join(str(x) for x in state["prior_objections"]))
        if state.get("caller_has_addressed"):
            lines.append("Caller has addressed: " + "; ".join(str(x) for x in state["caller_has_addressed"]))
        return "\n".join(lines)

    def _build_call_context(self, call_state: dict, best_branch: dict) -> str:
        tactic = best_branch.get("tactic", "short_redirect")
        label = best_branch.get("label", "natural")
        approach = best_branch.get("approach", "respond naturally")
        tone = best_branch.get("tone", "neutral")
        concedes = best_branch.get("what_recipient_concedes", "nothing")
        signals = best_branch.get("what_recipient_signals", "still listening")

        tactic_guidance = {
            "engage": (
                "Something the caller said genuinely caught your attention. Show brief, "
                "authentic curiosity — one short reaction, then a specific follow-up question. "
                "Don't over-commit or gush; you're interested, not sold."
            ),
            "short_redirect": (
                "You have a mild concern or this isn't quite landing yet. Give a brief, "
                "honest pushback — one or two sentences, no elaboration. Stay on the line "
                "but signal the caller needs to do better."
            ),
            "clarifying_question": (
                "The caller said something worth probing. Ask one clear, specific question "
                "about it. Don't ask broad questions — zero in on the one thing that would "
                "actually change your level of interest."
            ),
            "skeptical_probe": (
                "The caller made a claim that sounds generic, assumed, or vague. Call it out "
                "directly but without hostility — ask what they actually mean, or challenge "
                "the assumption. Keep it brief and factual."
            ),
            "polite_stall": (
                "You're not ruling it out, but you're not ready to move forward either. "
                "Acknowledge what they said, express mild openness, and leave room without "
                "committing to anything. Keep it short — one sentence of openness is enough."
            ),
            "booking_signal": (
                "You're genuinely open to hearing more at the right time. Signal that clearly "
                "but on your terms — suggest a next step, ask about format, or indicate when "
                "and how you'd be willing to follow up. Don't over-commit."
            ),
            "wrap_up": (
                "This call isn't going anywhere for you. End it cleanly: brief thanks, one "
                "honest reason it's not a fit right now, and a clear close. No harsh tone, "
                "no door slammed. Just done."
            ),
        }.get(tactic, "Respond naturally and in character.")

        block = (
            f"\n\nCALL RECEPTION STATE:\n"
            f"Phase: {call_state.get('call_phase', 'unknown')} | "
            f"Engagement: {call_state.get('recipient_engagement', 'unknown')} | "
            f"Call feel: {call_state.get('call_feel', 'fresh')}\n\n"
            f"CHOSEN RESPONSE TACTIC: {label} (tactic: {tactic})\n"
            f"Approach: {approach}\n"
            f"Tone: {tone}\n"
            f"What you concede this turn: {concedes}\n"
            f"What this signals to the caller: {signals}\n\n"
            f"EXECUTION GUIDANCE:\n{tactic_guidance}\n\n"
        )

        if call_state.get("prior_objections"):
            block += (
                "PRIOR OBJECTIONS YOU HAVE RAISED:\n"
                + "\n".join(f"- {o}" for o in call_state["prior_objections"])
                + "\n\n"
            )

        if call_state.get("key_hook"):
            block += f"KEY HOOK FROM THIS CALL: {call_state['key_hook']}\n\n"

        block += (
            "VARY YOUR OPENER: Do not start your response with the same word or phrase "
            "you used in your previous turn. Draw from: 'right,' 'mm-hmm,' 'got it,' "
            "'oh,' 'huh,' 'look,' 'well,' 'honestly' — never the same one twice in a row.\n\n"
            "Output only the in-character response — no labels, no meta-commentary, "
            "no stage directions."
        )
        return block
