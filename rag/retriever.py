import json

from openai import OpenAI
import config
from rag.pinecone_client import get_client

_openai: OpenAI = None


def _get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai


def embed(text: str) -> list[float]:
    response = _get_openai().embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=[text],
    )
    return response.data[0].embedding


def embed_batch(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    client = _get_openai()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        embeddings.extend(d.embedding for d in resp.data)
    return embeddings


def _namespace(persona_name: str) -> str:
    return persona_name.lower().replace(" ", "_")


def retrieve_examples(
    query_text: str,
    persona_name: str,
    top_k: int = None,
) -> list[dict]:
    """Return top-k example Q&A pairs relevant to query_text."""
    if top_k is None:
        top_k = config.RAG_TOP_K

    vector = embed(query_text)
    matches = get_client().query(
        vector=vector,
        namespace=_namespace(persona_name),
        top_k=top_k,
        filter={"type": {"$eq": "example"}},
    )

    return [
        {
            "score": m.score,
            "user_msg": m.metadata.get("user_msg", ""),
            "persona_response": m.metadata.get("persona_response", ""),
        }
        for m in matches
    ]


def retrieve_cold_call_examples(
    query_text: str,
    top_k: int = None,
) -> list[dict]:
    """Return top-k real cold-call exchange pairs relevant to query_text."""
    if top_k is None:
        top_k = config.RAG_TOP_K

    vector = embed(query_text)
    matches = get_client().query(
        vector=vector,
        namespace="cold_calls",
        top_k=top_k,
    )

    return [
        {
            "score": m.score,
            "caller_turn": m.metadata.get("caller_turn", ""),
            "recipient_turn": m.metadata.get("recipient_turn", ""),
            "prior_context": json.loads(m.metadata.get("prior_context", "[]")),
            "call_id": m.metadata.get("call_id", ""),
        }
        for m in matches
    ]


def retrieve_persona_info(persona_name: str) -> dict:
    """Return the stored persona profile metadata dict."""
    vector = embed("persona background profile information")
    matches = get_client().query(
        vector=vector,
        namespace=_namespace(persona_name),
        top_k=1,
        filter={"type": {"$eq": "persona_info"}},
    )
    if matches:
        meta = dict(matches[0].metadata)
        meta.pop("type", None)
        return meta
    return {}
