from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: Dict[str, Any]


class RAGEngine:
    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str,
        llm_model_name: str = "google/flan-t5-base",
        device: int = -1,
    ) -> None:
        """
        - Embeddings: all-MiniLM-L6-v2 :contentReference[oaicite:19]{index=19}
        - Vector DB: ChromaDB :contentReference[oaicite:20]{index=20}
        """
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        self.client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # lightweight CPU-friendly generator model (replace if you have a better local model)
        # The assignment allows HF pipeline :contentReference[oaicite:21]{index=21}
        self.generator = pipeline(
            task="text2text-generation",
            model=llm_model_name,
            device=device,
        )

    def retrieve(self, question: str, k: int = 5) -> List[RetrievedChunk]:
        q_emb = self.embedder.encode([question], normalize_embeddings=True).tolist()[0]

        # Chroma returns distances; for cosine, smaller distance ~ more similar (depending on config).
        # We'll keep the returned distance as "score" to display; lower is better.
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        out = []
        for d, m, dist in zip(docs, metas, dists):
            out.append(RetrievedChunk(text=d, metadata=m, score=float(dist)))
        return out

    @staticmethod
    def build_prompt(context: str, question: str) -> str:
        # Matches the intent of the example template in the PDF :contentReference[oaicite:22]{index=22}
        return (
            "You are a financial analyst assistant for CrediTrust. Your task is to answer questions "
            "about customer complaints.\n"
            "Use ONLY the retrieved complaint excerpts in the Context. "
            "If the Context does not contain enough information to answer, say you don't have enough information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def generate(self, question: str, retrieved: List[RetrievedChunk], max_new_tokens: int = 256) -> str:
        context = "\n\n---\n\n".join(
            [f"[Source {i+1}] {c.text}" for i, c in enumerate(retrieved)]
        )
        prompt = self.build_prompt(context=context, question=question)

        gen = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        # HF returns list[dict] with "generated_text"
        return gen[0]["generated_text"].strip()

    def answer(self, question: str, k: int = 5) -> Tuple[str, List[RetrievedChunk]]:
        retrieved = self.retrieve(question, k=k)
        answer = self.generate(question, retrieved)
        return answer, retrieved


def format_sources_md(retrieved: List[RetrievedChunk], max_sources: int = 5) -> str:
    lines = []
    for i, ch in enumerate(retrieved[:max_sources]):
        meta = ch.metadata or {}
        header = (
            f"**Source {i+1}** "
            f"(complaint_id={meta.get('complaint_id')}, "
            f"product_category={meta.get('product_category')}, "
            f"chunk_index={meta.get('chunk_index')}/{meta.get('total_chunks')})"
        )
        lines.append(header)
        lines.append(ch.text)
        lines.append("")  # spacing
    return "\n".join(lines).strip()
