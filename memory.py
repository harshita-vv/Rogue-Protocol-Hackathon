"""Searchable local incident memory.

Prefers ChromaDB, then FAISS, when that library is already installed.
Otherwise uses scikit-learn TF-IDF. Nothing is installed from here.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class IncidentMemory:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []
        if self.path.is_file() and self.path.stat().st_size:
            self.records = json.loads(self.path.read_text())
        self.backend = "tfidf"
        self._collection = None
        self._faiss = None
        self._choose_backend()

    def _choose_backend(self):
        try:
            import chromadb
        except Exception:
            chromadb = None
        if chromadb is not None:
            client = chromadb.PersistentClient(path=str(self.path.parent / "chroma"))
            self._collection = client.get_or_create_collection("incidents")
            self.backend = "chroma"
            return
        try:
            import faiss
        except Exception:
            faiss = None
        if faiss is not None:
            self._faiss = faiss
            self.backend = "faiss"

    def _text(self, record):
        return " ".join(
            str(record.get(key, ""))
            for key in ("module", "message", "job", "explanation", "crew_id")
        )

    def add(self, record):
        log_id = str(record["log_id"])
        if any(str(item.get("log_id")) == log_id for item in self.records):
            return False
        self.records.append(record)
        self.path.write_text(json.dumps(self.records, indent=2))
        if self._collection is not None:
            self._collection.upsert(
                ids=[log_id],
                documents=[self._text(record)],
                metadatas=[
                    {
                        "module": str(record.get("module", "")),
                        "crew_id": str(record.get("crew_id", "")),
                    }
                ],
            )
        return True

    def search(self, query, limit=5):
        if not self.records or not str(query).strip():
            return []
        if self._collection is not None:
            found = self._collection.query(
                query_texts=[query],
                n_results=min(limit, len(self.records)),
            )
            ids = found.get("ids", [[]])[0]
            by_id = {str(item["log_id"]): item for item in self.records}
            return [by_id[item] for item in ids if item in by_id]
        corpus = [self._text(item) for item in self.records]
        matrix = normalize(TfidfVectorizer(stop_words="english").fit_transform(corpus + [query]))
        if self._faiss is not None:
            index = self._faiss.IndexFlatIP(matrix.shape[1])
            stored = matrix[:-1].toarray().astype("float32")
            probe = matrix[-1].toarray().astype("float32")
            index.add(stored)
            scores, positions = index.search(probe, min(limit, len(self.records)))
            return [
                self.records[int(position)]
                for position, score in zip(positions[0], scores[0])
                if position >= 0 and score > 0
            ]
        scores = (matrix[-1] @ matrix[:-1].T).toarray().ravel()
        order = np.argsort(scores)[::-1][:limit]
        return [self.records[int(index)] for index in order if scores[int(index)] > 0]
