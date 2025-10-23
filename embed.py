from typing import List
import numpy as np
from openai import OpenAI
from settings import settings

_client = OpenAI(api_key=settings.OPENAI_API_KEY)

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns an (n, d) float32 matrix of embeddings.
    """
    # OpenAI embeddings accept up to ~8192 tokens per text; our chunks are small.
    resp = _client.embeddings.create(
        model=settings.EMBED_MODEL,
        input=texts
    )
    vecs = [np.array(d.embedding, dtype="float32") for d in resp.data]
    return np.vstack(vecs)
