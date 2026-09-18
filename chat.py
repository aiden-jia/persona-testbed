"""
Persona Prompting Testbed — main chat interface.

Commands during chat:
  /help            Show this list
  /quit  /exit     End the session and export log
  /history         Print conversation so far
  /method <name>   Switch prompting method mid-session
  /compare         Compare all methods on the next message you type
  /info            Show current session details
  /debug           Toggle debug metadata display
  /save            Manually save session and export a log copy
  /stats           Show per-prompt response times and session average
"""

import sys
import time
from pathlib import Path

# ------------------------------------------------------------------ #
# Rich is required — fail fast with a helpful message.
# ------------------------------------------------------------------ #
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.text import Text
    from rich.rule import Rule
except ImportError:
    print("Install dependencies first:  pip install -r requirements.txt")
    sys.exit(1)

from rag.retriever import retrieve_examples, retrieve_cold_call_examples, retrieve_persona_info
from memory.conversation import ConversationMemory
from prompting.registry import METHODS, get_method, list_methods
from upload_persona import parse_persona_file
import config

console = Console()


# ------------------------------------------------------------------ #
# UI helpers
# ------------------------------------------------------------------ #

def print_header():
    console.print(
        Panel.fit(
            "[bold blue]Persona Prompting Testbed[/bold blue]\n"
            "[dim]Compare LLM prompting strategies for persona simulation[/dim]",
            border_style="blue",
        )
    )


def print_method_table():
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", width=3, style="dim")
    table.add_column("Method", style="cyan", width=18)
    table.add_column("RAG", width=5)
    table.add_column("Description")
    for i, m in enumerate(list_methods(), 1):
        table.add_row(
            str(i),
            m["name"],
            "[green]yes[/green]" if m["uses_rag"] else "[dim]no[/dim]",
            m["description"],
        )
    console.print(table)


def print_session_table(sessions: list[dict]):
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", width=3, style="dim")
    table.add_column("ID", width=10, style="cyan")
    table.add_column("Persona", width=16)
    table.add_column("Method", width=16)
    table.add_column("Msgs", justify="right", width=5)
    table.add_column("Last Updated")
    for i, s in enumerate(sessions, 1):
        table.add_row(
            str(i),
            s["session_id"],
            s["persona_name"],
            s["prompting_method"],
            str(s["message_count"]),
            s["updated_at"],
        )
    console.print(table)


# ------------------------------------------------------------------ #
# Persona loading
# ------------------------------------------------------------------ #

def load_persona_info(persona_name: str) -> dict:
    """Try Pinecone first, fall back to local file parse."""
    try:
        info = retrieve_persona_info(persona_name)
        if info:
            return info
    except Exception as e:
        console.print(f"[yellow]Pinecone unavailable ({e}). Falling back to local file.[/yellow]")

    # Local fallback
    slug = persona_name.lower().replace(" ", "_")
    candidates = [
        config.PERSONAS_DIR / f"{slug}.txt",
        config.PERSONAS_DIR / f"{persona_name}.txt",
    ]
    for path in candidates:
        if path.exists():
            info, _ = parse_persona_file(path)
            console.print(
                "[yellow]Warning: using local file — RAG retrieval disabled. "
                "Run upload_persona.py to enable it.[/yellow]"
            )
            return info

    console.print(f"[yellow]No persona data found for '{persona_name}'.[/yellow]")
    return {"name": persona_name}


def select_persona() -> tuple[str, dict]:
    persona_files = sorted(config.PERSONAS_DIR.glob("*.txt"))
    if not persona_files:
        console.print("[red]No persona files found in personas/.[/red]")
        console.print("Add a .txt file then run: python upload_persona.py personas/your_file.txt")
        sys.exit(1)

    console.print("\n[bold]Available Personas:[/bold]")
    for i, f in enumerate(persona_files, 1):
        label = f.stem.replace("_", " ").title()
        console.print(f"  [dim]{i}.[/dim] {label}")

    while True:
        raw = Prompt.ask("Select persona (number or name)", default="1")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(persona_files):
                name = persona_files[idx].stem.replace("_", " ").title()
                break
        except ValueError:
            name = raw.title()
            break

    console.print(f"[dim]Loading persona '{name}'...[/dim]")
    info = load_persona_info(name)
    actual_name = info.get("name", name)
    console.print(f"[green]Persona loaded:[/green] {actual_name}")
    return actual_name, info


def select_method() -> str:
    console.print("\n[bold]Prompting Methods:[/bold]")
    print_method_table()
    names = list(METHODS.keys())
    while True:
        raw = Prompt.ask("Select method (number or name)", default="2")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(names):
                return names[idx]
        except ValueError:
            if raw in names:
                return raw
        console.print(f"[red]Invalid. Options: {names}[/red]")


# ------------------------------------------------------------------ #
# RAG retrieval
# ------------------------------------------------------------------ #

def get_rag_examples(
    user_input: str,
    memory: ConversationMemory,
    method_name: str,
    persona_name: str,
    persona_info: dict = None,
) -> list[dict]:
    """Return RAG examples if the chosen method uses them, else []."""
    if not METHODS[method_name].uses_rag:
        return []
    try:
        if (persona_info or {}).get("rag_namespace") == "cold_calls":
            return retrieve_cold_call_examples(user_input)
        query = user_input
        if len(memory) > 0:
            query = memory.get_rag_context() + "\n" + user_input
        return retrieve_examples(query, persona_name)
    except Exception as e:
        console.print(f"[yellow]RAG error: {e}[/yellow]")
        return []


# ------------------------------------------------------------------ #
# Compare mode
# ------------------------------------------------------------------ #

def print_timing_table(response_times: list[tuple[int, str, float]], title: str = "Response Times"):
    """Render a Rich table of per-turn timings plus per-method averages."""
    if not response_times:
        console.print("[dim]No timing data yet.[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2), title=title)
    table.add_column("Turn", width=5, justify="right", style="dim")
    table.add_column("Method", width=18, style="cyan")
    table.add_column("Time (s)", width=9, justify="right")

    method_totals: dict[str, list[float]] = {}
    for turn, method, elapsed in response_times:
        table.add_row(str(turn), method, f"{elapsed:.2f}")
        method_totals.setdefault(method, []).append(elapsed)

    console.print(table)

    avg_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2), title="Averages by Method")
    avg_table.add_column("Method", width=18, style="cyan")
    avg_table.add_column("Turns", width=6, justify="right")
    avg_table.add_column("Avg (s)", width=8, justify="right")
    avg_table.add_column("Total (s)", width=9, justify="right")

    all_times = [t for _, _, t in response_times]
    for mname, times in sorted(method_totals.items()):
        avg_table.add_row(mname, str(len(times)), f"{sum(times)/len(times):.2f}", f"{sum(times):.2f}")

    overall_avg = sum(all_times) / len(all_times)
    avg_table.add_row(
        "[bold]Overall[/bold]", str(len(all_times)),
        f"[bold]{overall_avg:.2f}[/bold]", f"[bold]{sum(all_times):.2f}[/bold]"
    )
    console.print(avg_table)


def run_compare(
    user_input: str,
    persona_info: dict,
    memory: ConversationMemory,
):
    persona_name = persona_info.get("name", "Persona")
    history = memory.get_api_history()
    console.print(f"\n[bold yellow]Running all {len(METHODS)} methods...[/bold yellow]")

    results: dict[str, dict] = {}
    compare_times: list[tuple[int, str, float]] = []
    for mname, method in METHODS.items():
        rag = get_rag_examples(user_input, memory, mname, persona_name, persona_info)
        console.print(f"  [dim]{mname}...[/dim]", end=" ")
        t0 = time.perf_counter()
        try:
            resp, meta = method.generate(
                persona_info=persona_info,
                rag_examples=rag,
                conversation_history=history,
                user_input=user_input,
            )
            elapsed = time.perf_counter() - t0
            results[mname] = {"response": resp, "meta": meta, "elapsed": elapsed}
            compare_times.append((1, mname, elapsed))
            calls = meta.get("api_calls", 1)
            console.print(f"[green]done[/green] ({calls} call{'s' if calls > 1 else ''}, {elapsed:.2f}s)")
        except Exception as e:
            results[mname] = {"response": f"[ERROR] {e}", "meta": {}, "elapsed": 0.0}
            console.print(f"[red]failed[/red]")

    console.print()
    for mname, res in results.items():
        console.print(
            Panel(
                res["response"],
                title=f"[cyan bold]{mname}[/cyan bold]",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if compare_times:
        console.print()
        print_timing_table(compare_times, title="Compare — Response Times")

    return results


# ------------------------------------------------------------------ #
# Main chat loop
# ------------------------------------------------------------------ #

def run_chat(memory: ConversationMemory, persona_info: dict):
    persona_name = persona_info.get("name", "Persona")
    debug = False
    response_times: list[tuple[int, str, float]] = []  # (turn, method, elapsed_s)
    turn = 0

    console.print(Rule())
    console.print(
        f"[bold green]Session:[/bold green] {memory.session_id}  "
        f"[bold]Persona:[/bold] [cyan]{persona_name}[/cyan]  "
        f"[bold]Method:[/bold] [cyan]{memory.prompting_method}[/cyan]"
    )
    if len(memory) > 0:
        console.print(f"[dim]Resuming — {len(memory)} messages already in session.[/dim]")
    console.print("[dim]Type /help for commands. Ctrl+C or /quit to exit.[/dim]")
    console.print()

    method = get_method(memory.prompting_method)

    while True:
        try:
            user_input = Prompt.ask("[bold]Salesperson[/bold]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        # ---- Commands ------------------------------------------------ #
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                break

            elif cmd == "/help":
                console.print(Panel(
                    "/quit, /exit    End session and export log\n"
                    "/history        Print conversation history\n"
                    "/method <name>  Switch prompting method\n"
                    "/compare        Compare all methods on next input\n"
                    "/info           Show session info\n"
                    "/debug          Toggle token/metadata display\n"
                    "/save           Save session + export log now\n"
                    "/stats          Show per-prompt response times and average",
                    title="Commands", border_style="dim", padding=(0, 2)
                ))

            elif cmd == "/history":
                if not memory.messages:
                    console.print("[dim]No messages yet.[/dim]")
                else:
                    console.print()
                    for msg in memory.messages:
                        if msg["role"] == "user":
                            label = "[bold blue]Salesperson[/bold blue]"
                        else:
                            label = f"[bold green]{persona_name}[/bold green]"
                        console.print(f"{label}: {msg['content']}\n")

            elif cmd == "/method":
                if arg in METHODS:
                    memory.prompting_method = arg
                    method = get_method(arg)
                    memory.save()
                    console.print(f"[green]Switched to:[/green] {arg}")
                else:
                    console.print(f"[red]Unknown method. Options: {list(METHODS.keys())}[/red]")

            elif cmd == "/compare":
                msg = Prompt.ask("[bold]Message to compare[/bold]").strip()
                if msg:
                    run_compare(msg, persona_info, memory)

            elif cmd == "/info":
                console.print(Panel(
                    f"Session ID : {memory.session_id}\n"
                    f"Persona    : {memory.persona_name}\n"
                    f"Method     : {memory.prompting_method}\n"
                    f"Messages   : {len(memory)}\n"
                    f"Created    : {memory.created_at[:19]}\n"
                    f"Updated    : {memory.updated_at[:19]}",
                    title="Session Info", border_style="dim", padding=(0, 2)
                ))

            elif cmd == "/debug":
                debug = not debug
                console.print(f"[green]Debug mode:[/green] {'ON' if debug else 'OFF'}")

            elif cmd == "/save":
                memory.save()
                log = memory.export_log()
                console.print(f"[green]Saved.[/green] Log: [dim]{log}[/dim]")

            elif cmd == "/stats":
                print_timing_table(response_times, title="Session Response Times")

            else:
                console.print(f"[red]Unknown command '{cmd}'. Try /help.[/red]")
            continue

        # ---- Normal message ------------------------------------------ #
        turn += 1
        history = memory.get_api_history()  # snapshot before adding new message
        memory.add_message("user", user_input)

        rag_examples = get_rag_examples(
            user_input, memory, memory.prompting_method, persona_name, persona_info
        )
        if debug and rag_examples:
            scores = [f"{e['score']:.2f}" for e in rag_examples]
            console.print(f"[dim]RAG: {len(rag_examples)} examples, scores: {scores}[/dim]")

        t0 = time.perf_counter()
        if hasattr(method, "generate_streamed"):
            # Prep calls (analyze + branch/select) run under the spinner;
            # the final response streams in real-time after the spinner exits.
            with console.status(f"[dim]{persona_name} is thinking...[/dim]", spinner="dots"):
                try:
                    stream, meta = method.generate_streamed(
                        persona_info=persona_info,
                        rag_examples=rag_examples,
                        conversation_history=history,
                        user_input=user_input,
                    )
                except Exception as e:
                    console.print(f"[red]Generation error: {e}[/red]")
                    memory.messages.pop()
                    continue
            console.print(f"\n[bold green]{persona_name}:[/bold green] ", end="")
            chunks: list[str] = []
            try:
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                        chunks.append(delta)
            except Exception as e:
                console.print(f"\n[red]Streaming error: {e}[/red]")
                memory.messages.pop()
                continue
            elapsed = time.perf_counter() - t0
            sys.stdout.write("\n\n")
            sys.stdout.flush()
            response = "".join(chunks)
            if "walked_out" not in meta:
                meta["walked_out"] = method.detect_walkout(response, persona_info)
        else:
            with console.status(f"[dim]{persona_name} is thinking...[/dim]", spinner="dots"):
                try:
                    response, meta = method.generate(
                        persona_info=persona_info,
                        rag_examples=rag_examples,
                        conversation_history=history,
                        user_input=user_input,
                    )
                    # ToT sets "walked_out" explicitly; other methods need a detection call.
                    if "walked_out" not in meta:
                        meta["walked_out"] = method.detect_walkout(response, persona_info)
                except Exception as e:
                    console.print(f"[red]Generation error: {e}[/red]")
                    memory.messages.pop()  # undo the user message we added
                    continue
            elapsed = time.perf_counter() - t0
            console.print(f"\n[bold green]{persona_name}:[/bold green] {response}\n")

        response_times.append((turn, memory.prompting_method, elapsed))

        if debug:
            debug_lines = [
                f"{k}: {v}"
                for k, v in meta.items()
                if k not in {"draft", "critique", "selected_branch"}
            ]
            console.print(f"[dim]{'  |  '.join(debug_lines)}[/dim]\n")

        memory.add_message("assistant", response)
        memory.save()

        if meta.get("walked_out"):
            console.print(
                Panel(
                    f"[bold]{persona_name} has walked out of the negotiation.[/bold]\n"
                    "[dim]The deal could not be closed.[/dim]",
                    title="[bold red]Deal Failed[/bold red]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            break

    # ---- Session end ------------------------------------------------- #
    console.print("\n[bold]Session ended.[/bold]")
    if response_times:
        console.print()
        print_timing_table(response_times, title="Session Response Times")
    try:
        log = memory.export_log()
        console.print(f"[dim]Log saved → {log}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Could not export log: {e}[/yellow]")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    print_header()

    console.print("\n[bold]What would you like to do?[/bold]")
    console.print("  [dim]1.[/dim] New conversation")
    console.print("  [dim]2.[/dim] Continue existing session")
    console.print("  [dim]3.[/dim] Compare all methods on a single message")
    console.print("  [dim]4.[/dim] Exit")

    choice = Prompt.ask("Choice", choices=["1", "2", "3", "4"], default="1")

    if choice == "4":
        return

    # ---- Continue session -------------------------------------------- #
    if choice == "2":
        sessions = ConversationMemory.list_sessions()
        if not sessions:
            console.print("[yellow]No sessions found. Starting a new one.[/yellow]")
            choice = "1"
        else:
            console.print("\n[bold]Existing Sessions:[/bold]")
            print_session_table(sessions)
            raw = Prompt.ask("Select session number", default="1")
            try:
                session = sessions[int(raw) - 1]
                memory = ConversationMemory.load(session["session_id"])
                persona_info = load_persona_info(memory.persona_name)
                run_chat(memory, persona_info)
                return
            except (ValueError, IndexError):
                console.print("[red]Invalid selection.[/red]")
                return
            except FileNotFoundError as e:
                console.print(f"[red]{e}[/red]")
                return

    # ---- Compare mode ------------------------------------------------ #
    if choice == "3":
        persona_name, persona_info = select_persona()
        dummy = ConversationMemory(persona_name=persona_name, prompting_method="basic")
        msg = Prompt.ask("\n[bold]Enter message to compare across all methods[/bold]").strip()
        if msg:
            run_compare(msg, persona_info, dummy)
        return

    # ---- New conversation -------------------------------------------- #
    persona_name, persona_info = select_persona()
    method_name = select_method()

    memory = ConversationMemory(
        persona_name=persona_name,
        prompting_method=method_name,
    )
    memory.save()

    run_chat(memory, persona_info)


if __name__ == "__main__":
    main()
