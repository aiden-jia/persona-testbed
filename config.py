import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "persona-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
RAG_TOP_K = 3
MAX_HISTORY_MESSAGES = 20

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
SESSIONS_DIR = BASE_DIR / "sessions"
PERSONAS_DIR = BASE_DIR / "personas"

for _d in [LOGS_DIR, SESSIONS_DIR, PERSONAS_DIR]:
    _d.mkdir(exist_ok=True)
