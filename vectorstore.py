import os, json
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from embed import embed_texts
from settings import settings

class VectorStore:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.index_path = os.path.join(index_dir, "faiss_ip.index")
        self.meta_path  = os.path.join(index_dir, "meta.jsonl")
        self.index = None
        self.dim = None
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load()

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norms

    def _save(self):
        faiss.write_index(self.index, self.index_path)

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        self.dim = self.index.d

    def add_texts(self, chunks: List[str], metadatas: List[Dict[str, Any]]):
        if not chunks:
            return
        vecs = self._normalize(embed_texts(chunks))
        if self.index is None:
            self.dim = vecs.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)
        with open(self.meta_path, "a", encoding="utf-8") as f:
            for m in metadatas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        self._save()

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        assert self.index is not None, "Index is empty. Ingest data first."
        q = self._normalize(embed_texts([query]))
        D, I = self.index.search(q, k)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # keep ids within meta length
        I = np.where(I < len(lines), I, -1)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = json.loads(lines[idx])
            results.append((float(score), meta))
        return results

    def stats(self) -> dict:
        chunks = 0
        files = set()
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    chunks += 1
                    try:
                        src = json.loads(line).get("source")
                        if src:
                            files.add(src)
                    except Exception:
                        pass
        return {"chunks": chunks, "files": len(files)}
