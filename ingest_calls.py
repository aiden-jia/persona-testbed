#!/usr/bin/env python3
"""
Ingests cold call MP3s into Pinecone as exchange-pair RAG examples.

Pipeline: MP3 -> Whisper transcription -> GPT-4o-mini diarization -> embed -> Pinecone
Namespace: "cold_calls" within the existing persona-rag index.
"""

import json
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

import config
from rag.pinecone_client import get_client

TRAINING_DIR = Path(__file__).parent / "training_audio"
NAMESPACE = "cold_calls"
console = Console()


def transcribe(client: OpenAI, audio_path: Path) -> str:
    with open(audio_path, "rb") as f:
        return client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )


def extract_turns(client: OpenAI, transcript: str) -> list[dict]:
    """Skip the KLICKAUD provider intro and label every turn as CALLER or RECIPIENT."""
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {
                "role": "system",
                "content": "You analyze cold call transcripts. Return only valid JSON — no prose, no markdown.",
            },
            {
                "role": "user",
                "content": f"""This is a cold call transcript. At the very beginning there is an automated
announcement from the audio provider (KLICKAUD) — ignore it entirely.

After the intro there are exactly two speakers:
- CALLER: the salesperson who placed the call
- RECIPIENT: the person who received it

Label every turn and return JSON in this exact shape:
{{
  "turns": [
    {{"speaker": "CALLER", "text": "..."}},
    {{"speaker": "RECIPIENT", "text": "..."}}
  ]
}}

TRANSCRIPT:
{transcript}""",
            },
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("turns", [])


def build_exchange_pairs(turns: list[dict], call_id: str) -> list[dict]:
    """
    For each recipient turn, find the preceding caller turn and a short
    window of prior context (last 3 turns before the exchange).
    """
    pairs = []
    for i, turn in enumerate(turns):
        if turn["speaker"] != "RECIPIENT":
            continue

        # find the immediately preceding caller turn
        caller_turn = None
        for j in range(i - 1, -1, -1):
            if turns[j]["speaker"] == "CALLER":
                caller_turn = turns[j]["text"]
                break
        if not caller_turn:
            continue

        recipient_text = turn["text"].strip()
        if len(recipient_text.split()) < 3:  # skip filler responses
            continue

        prior = [f"{t['speaker']}: {t['text']}" for t in turns[max(0, i - 4) : i - 1]]

        pairs.append(
            {
                "caller_turn": caller_turn,
                "recipient_turn": recipient_text,
                "prior_context": prior,
                "call_id": call_id,
            }
        )
    return pairs


def embed(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def ingest_all():
    mp3_files = sorted(TRAINING_DIR.glob("*.mp3"))
    if not mp3_files:
        console.print("[red]No MP3 files found in training_audio/[/red]")
        sys.exit(1)

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    pc = get_client()
    total_pairs = 0

    for mp3 in mp3_files:
        call_id = mp3.stem
        console.print(f"\n[bold cyan]Processing:[/bold cyan] {mp3.name}")

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
            t = p.add_task("Transcribing with Whisper...")
            transcript = transcribe(client, mp3)
            p.update(t, description=f"Transcribed ({len(transcript.split())} words)")
            p.stop_task(t)

            t = p.add_task("Diarizing speakers with GPT...")
            turns = extract_turns(client, transcript)
            p.update(t, description=f"Found {len(turns)} turns")
            p.stop_task(t)

        pairs = build_exchange_pairs(turns, call_id)
        if not pairs:
            console.print("  [yellow]No usable exchange pairs — skipping.[/yellow]")
            continue

        console.print(f"  Embedding and upserting {len(pairs)} exchange pairs...")
        vectors = []
        for i, pair in enumerate(pairs):
            vec = embed(client, pair["caller_turn"])
            vectors.append(
                {
                    "id": f"{call_id}_{i:03d}",
                    "values": vec,
                    "metadata": {
                        "caller_turn": pair["caller_turn"],
                        "recipient_turn": pair["recipient_turn"],
                        "prior_context": json.dumps(pair["prior_context"]),
                        "call_id": call_id,
                    },
                }
            )

        pc.upsert(vectors, namespace=NAMESPACE)
        total_pairs += len(vectors)
        console.print(f"  [green]Done — {len(vectors)} pairs upserted.[/green]")

    console.print(
        f"\n[bold green]Ingestion complete: {total_pairs} exchange pairs across {len(mp3_files)} calls.[/bold green]"
    )


if __name__ == "__main__":
    ingest_all()
