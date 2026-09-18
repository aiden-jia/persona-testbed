from pinecone import Pinecone, ServerlessSpec
import config

_instance = None


def get_client() -> "PineconeClient":
    global _instance
    if _instance is None:
        _instance = PineconeClient()
    return _instance


class PineconeClient:
    def __init__(self):
        if not config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set in your .env file.")
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self._ensure_index()
        self.index = self.pc.Index(config.PINECONE_INDEX_NAME)

    def _ensure_index(self):
        existing = [idx.name for idx in self.pc.list_indexes()]
        if config.PINECONE_INDEX_NAME not in existing:
            print(f"Creating Pinecone index '{config.PINECONE_INDEX_NAME}'...")
            self.pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=config.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=config.PINECONE_CLOUD,
                    region=config.PINECONE_REGION,
                ),
            )
            print("Index created.")

    def upsert(self, vectors: list[dict], namespace: str):
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i : i + batch_size], namespace=namespace)

    def query(
        self,
        vector: list[float],
        namespace: str,
        top_k: int = 5,
        filter: dict = None,
    ) -> list:
        kwargs = dict(
            vector=vector,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True,
        )
        if filter:
            kwargs["filter"] = filter
        result = self.index.query(**kwargs)
        return result.matches

    def delete_namespace(self, namespace: str):
        self.index.delete(delete_all=True, namespace=namespace)

    def stats(self) -> dict:
        return self.index.describe_index_stats()
