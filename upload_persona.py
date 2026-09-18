"""
Upload a persona file to Pinecone.

Usage:
    python upload_persona.py personas/car_buyer.txt
    python upload_persona.py personas/car_buyer.txt --clear
    python upload_persona.py --list
"""
import argparse
import sys
from pathlib import Path
from openai import OpenAI
from rag.pinecone_client import get_client
from rag.retriever import embed_batch
import config


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_persona_file(filepath: Path) -> tuple[dict, list[dict]]:
    """
    Returns (persona_info_dict, examples_list).

    persona_info_dict includes a "role_prompt" key if a [ROLE_PROMPT] section
    is present — this is the rich narrative character brief used as the
    persistent system prompt throughout the entire conversation.

    File format:
        [PERSONA_INFO]   — structured key: value pairs
        [ROLE_PROMPT]    — free-form narrative (personality, tone, goals, style)
        [EXAMPLES]       — Q&A pairs separated by ---
    Lines starting with # are comments and are ignored.
    """
    text = filepath.read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    current = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)

    # Parse PERSONA_INFO
    persona_info: dict[str, str] = {}
    for line in sections.get("PERSONA_INFO", []):
        if ":" in line:
            key, _, value = line.partition(":")
            k, v = key.strip(), value.strip()
            if k and v:
                persona_info[k] = v

    # Parse ROLE_PROMPT — store as a single string under "role_prompt"
    role_prompt_lines = sections.get("ROLE_PROMPT", [])
    role_prompt = "\n".join(role_prompt_lines).strip()
    if role_prompt:
        persona_info["role_prompt"] = role_prompt

    # Parse EXAMPLES (split on ---)
    examples: list[dict] = []
    raw_examples = "\n".join(sections.get("EXAMPLES", []))
    for block in raw_examples.split("---"):
        block = block.strip()
        if not block:
            continue
        user_msg = ""
        persona_lines: list[str] = []
        in_persona = False
        for line in block.splitlines():
            if line.startswith("USER:"):
                user_msg = line[5:].strip()
                in_persona = False
            elif line.startswith("PERSONA:"):
                persona_lines.append(line[8:].strip())
                in_persona = True
            elif in_persona and line.strip():
                persona_lines.append(line.strip())
        if user_msg and persona_lines:
            examples.append(
                {
                    "user_msg": user_msg,
                    "persona_response": " ".join(persona_lines),
                }
            )

    return persona_info, examples


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #

def upload_persona(filepath: str, clear: bool = False):
    path = Path(filepath)
    if not path.exists():
        print(f"Error: file not found — {path}")
        sys.exit(1)

    print(f"Parsing {path.name}...")
    persona_info, examples = parse_persona_file(path)

    if not persona_info.get("name"):
        print("Error: [PERSONA_INFO] must contain a 'name' field.")
        sys.exit(1)

    persona_name: str = persona_info["name"]
    namespace: str = persona_name.lower().replace(" ", "_")

    print(f"  Persona   : {persona_name}")
    print(f"  Examples  : {len(examples)}")
    print(f"  Namespace : {namespace}")

    pc = get_client()

    if clear:
        print("Clearing existing namespace...")
        pc.delete_namespace(namespace)

    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    vectors: list[dict] = []

    # Persona info vector
    info_text = " | ".join(f"{k}: {v}" for k, v in persona_info.items())
    info_emb = openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL, input=[info_text]
    ).data[0].embedding

    # Ensure all metadata values are strings (Pinecone requirement)
    safe_info = {k: str(v) for k, v in persona_info.items()}

    vectors.append(
        {
            "id": f"{namespace}__info",
            "values": info_emb,
            "metadata": {
                **safe_info,
                "type": "persona_info",
                "persona_name": persona_name,
            },
        }
    )

    # Example vectors
    if examples:
        print(f"Embedding {len(examples)} examples...")
        texts = [
            f"Salesperson: {ex['user_msg']}\nPersona: {ex['persona_response']}"
            for ex in examples
        ]
        embeddings = embed_batch(texts)
        for i, (ex, emb) in enumerate(zip(examples, embeddings)):
            vectors.append(
                {
                    "id": f"{namespace}__ex_{i:03d}",
                    "values": emb,
                    "metadata": {
                        "type": "example",
                        "persona_name": persona_name,
                        "user_msg": ex["user_msg"],
                        "persona_response": ex["persona_response"],
                    },
                }
            )

    print(f"Uploading {len(vectors)} vectors...")
    pc.upsert(vectors, namespace=namespace)
    print(f"Done. '{persona_name}' is ready ({len(examples)} examples in Pinecone).")


# --------------------------------------------------------------------------- #
# List personas
# --------------------------------------------------------------------------- #

def list_personas():
    pc = get_client()
    stats = pc.stats()
    namespaces = stats.get("namespaces", {})
    if not namespaces:
        print("No personas found in Pinecone index.")
        return
    print(f"Personas in '{config.PINECONE_INDEX_NAME}':")
    for ns, info in namespaces.items():
        print(f"  {ns}  ({info.get('vector_count', '?')} vectors)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manage persona data in Pinecone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python upload_persona.py personas/car_buyer.txt\n"
            "  python upload_persona.py personas/car_buyer.txt --clear\n"
            "  python upload_persona.py --list"
        ),
    )
    parser.add_argument("filepath", nargs="?", help="Path to persona .txt file")
    parser.add_argument(
        "--clear", action="store_true",
        help="Delete existing vectors for this persona before uploading"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all personas currently in Pinecone"
    )
    args = parser.parse_args()

    if args.list:
        list_personas()
    elif args.filepath:
        upload_persona(args.filepath, clear=args.clear)
    else:
        parser.print_help()
