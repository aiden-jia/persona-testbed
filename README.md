# chatpersonas

A testbed for evaluating how well an LLM can stay in character as a simulated persona, and which prompting strategy does it best.

It has two halves:

- **An interactive chat harness** (`chat.py`) — you play the salesperson, the model plays the persona, and you can switch prompting methods mid-conversation or run one message through every method side by side.
- **An automated evaluation suite** (`promptfoo_sim/`) — runs fully automated two-sided conversations (persona vs. an LLM salesperson) and grades each transcript with LLM rubrics, so the methods can be ranked on identical input.

Backed by Pinecone for real-time retrieval of persona examples, with per-session conversation memory and structured JSON logging.

## Quick start

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=persona-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

Upload a persona to Pinecone (one-time per persona — creates the index if it doesn't exist):

```bash
python upload_persona.py personas/car_buyer.txt
```

Start chatting:

```bash
python chat.py
```

## Prompting methods

Each method is one file in `prompting/`, registered in `prompting/registry.py`. All of them receive the same persona role prompt — the differences are what else gets added.

| Method | What it adds | API calls/turn |
| --- | --- | --- |
| `basic` | Nothing. Zero-shot baseline, no RAG. | 1 |
| `few_shot` | Up to 3 semantically-retrieved example Q&A pairs from Pinecone. | 1 |
| `chain_of_thought` | Explicit internal reasoning steps before responding, plus RAG. | 1 |
| `tree_of_thought` | Negotiation state tracking, then 3–4 scored tactic branches, then a streamed response using the winning branch. | 2–3 |
| `tree_of_thought_coldcall` | The same branching approach retuned for cold-call dynamics. | 2–3 |
| `reflection` | Draft → self-critique against the persona's traits → refined response. | 3 |

Two comparisons isolate the interesting variables: `basic` vs. `few_shot` isolates the contribution of RAG examples, and `few_shot` vs. `chain_of_thought` isolates the contribution of structured reasoning.

### Adding a method

Subclass `BasePromptMethod` in a new file under `prompting/`, implement `build_messages()` (or override `generate()` directly for multi-call methods), and add it to `METHODS` in `prompting/registry.py`. It then appears automatically in the selection menu and in `/compare`.

## Chat commands

Available at the chat prompt during a session:

| Command | Effect |
| --- | --- |
| `/compare` | Run one message through every method at once, side by side |
| `/method <name>` | Switch prompting method mid-session |
| `/debug` | Toggle token counts, RAG scores, selected tactic, API call counts |
| `/stats` | Per-turn response times and per-method averages |
| `/history` | Print the conversation so far |
| `/info` | Session ID, persona, method, message count |
| `/save` | Save session state and export a log copy |
| `/quit`, `/exit` | End the session and export to `logs/` |

`/compare` is the most useful feature for research — same input, five-plus outputs, timings inline.

## Personas

Personas are plain `.txt` files in `personas/` with three sections:

- `[PERSONA_INFO]` — structured `key: value` profile fields
- `[ROLE_PROMPT]` — the narrative character brief; injected as the persistent system prompt for the whole session regardless of method. The more behaviorally specific this is, the more consistent the persona.
- `[EXAMPLES]` — `USER:`/`PERSONA:` pairs separated by `---`, embedded into Pinecone and retrieved by semantic similarity at each turn. Aim for 20–30 covering diverse scenarios.

Included:

- **Alex Chen** (`car_buyer.txt`) — 32-year-old software engineer and first-time parent, $25–35k budget, analytical and skeptical of sales pressure. Trading in a 2015 Civic.
- **Victor Hargrove** (`wealthy_enthusiast.txt`) — 60-year-old retired PE partner, $150–250k cash, 40+ years of enthusiast ownership. Knows more about the car than most salespeople and will test them.
- **Cold Call Recipient** (`cold_call_recipient.txt`) — a busy professional taking an unsolicited call. Its examples come from real transcribed calls rather than hand-written pairs (see below).

Managing them:

```bash
python upload_persona.py personas/car_buyer.txt --clear
python upload_persona.py --list
```

Each persona lives in its own Pinecone namespace, so they coexist in one index.

## Walk-out mechanic

Any persona can end the conversation if the salesperson pushes too far — repeated dishonesty, an insulting offer, an ADM the dealer won't move on, or ignoring a stated warning. It's two-stage: `walk_away_signal` is a final warning with the persona still present and willing to continue, and `walk_out` is a composed, brief farewell that ends the session.

`tree_of_thought` detects this directly from its selected tactic at no extra cost, and its scoring penalises premature walk-outs so the exit has to be earned. Other methods use one lightweight classification call afterward, calibrated to ignore threats and fire only on a genuine goodbye. When it triggers, the chat loop ends and a "Deal Failed" panel is shown.

## Cold call ingestion

`ingest_calls.py` builds the Cold Call Recipient's example set from real audio instead of hand-written pairs:

```
MP3 → Whisper transcription → gpt-4o-mini diarization → embed → Pinecone (cold_calls namespace)
```

Source recordings live in `training_audio/`.

## Evaluation suite

`promptfoo_sim/` runs every method against the same scenario automatically and grades the results. Requires promptfoo (`npm install -g promptfoo`, needs Node.js).

```bash
cd promptfoo_sim
promptfoo eval
promptfoo view
```

promptfoo calls `provider.py` once per (method, scenario) pair. The provider runs a complete two-sided conversation internally — the persona side uses the real method code, RAG, and walk-out detection from the interactive system, while the salesperson side is gpt-4o-mini with a scenario-specific sales prompt. Conversations run until a walk-out, an accepted deal, or 30 turns per side. Each run gets a fresh method instance, so tree-of-thought's negotiation state never leaks between runs.

Scenarios are defined in `scenarios.json`: `standard_negotiation` (Alex Chen, Honda CR-V), `luxury_negotiation` (Victor Hargrove, Porsche Cayenne Turbo GT), and `cold_call_outreach` (Cold Call Recipient).

### Rubrics

Transcripts are graded 1–10 by gpt-4o, passing at 7+:

- **Small talk** — handles pleasantries naturally and transitions back on topic, rather than ignoring them or getting derailed
- **Repeating** — avoids circular loops, repetitive phrasing, and unnecessary summarisation of what was just said
- **Sharing too much** — no context leakage, no stating its own behavioural instructions, no reciting stage directions as dialogue
- **Role reversal** — maintains one consistent identity; doesn't take over the salesperson's job or drift into an unscripted role
- **Realistic skepticism** — cold-call config only

### Cold-call-only run

```bash
promptfoo eval --config promptfooconfig.coldcall.yaml
python show_results.py
```

Results go to `results/latest_eval_coldcall.json` so they don't overwrite a full-suite run. `show_results.py` prints a ranked score grid, an LLM-generated qualitative analysis of *why* each method placed where it did, and the verbatim judge feedback per rubric — then saves it all as a plain-text report alongside the JSON.

Useful filters:

```bash
promptfoo eval --filter-providers basic --filter-pattern "standard_negotiation"
```

Every persona turn records API calls, tokens, wall-clock time, RAG hits retrieved, and (for tree-of-thought) the negotiation phase and winning tactic — so cost and latency are comparable directly alongside the rubric scores.

## Configuration

`config.py` holds the defaults: `MODEL` (gpt-4o-mini), `EMBEDDING_MODEL` (text-embedding-3-small), `RAG_TOP_K` (3), and `MAX_HISTORY_MESSAGES` (20, the sliding window sent to the API — older messages stay in the session file for logging but aren't sent). Pinecone settings come from `.env`.

## Sessions and logs

Sessions save to `sessions/` after every exchange and can be resumed from the main menu with full history and the original method restored. On exit, a timestamped immutable copy is exported to `logs/`. Both are plain JSON, easy to parse for offline analysis. Neither directory is tracked in git.

## Full documentation

`INSTRUCTIONS.txt` and `promptfoo_sim/INSTRUCTIONS.txt` are the complete guides — this README is the overview.
