from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .chunking import chunk_text


FILTERED_PATH_DEFAULT = Path("data/filtered_complaints.csv")
SAMPLE_OUT_DEFAULT = Path("data/processed/sample_15k.csv")

CHROMA_DIR_DEFAULT = Path("vector_store/sample_chroma")
COLLECTION_NAME = "complaints_sample"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def stratified_sample(df: pd.DataFrame, label_col: str, n_total: int, seed: int = 42) -> pd.DataFrame:
    """Proportional stratified sampling across label_col."""
    if n_total <= 0:
        raise ValueError("n_total must be > 0")

    counts = df[label_col].value_counts()
    total = counts.sum()
    # target per class (rounded), ensure at least 1 if class exists
    targets = (counts / total * n_total).round().astype(int).to_dict()

    # fix rounding drift
    drift = n_total - sum(targets.values())
    if drift != 0:
        # add/subtract drift from largest class
        largest = counts.index[0]
        targets[largest] = max(1, targets[largest] + drift)

    parts = []
    for label, k in targets.items():
        group = df[df[label_col] == label]
        k = min(k, len(group))
        parts.append(group.sample(n=k, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def build_chroma_collection(persist_dir: Path, collection_name: str):
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=collection_name)


def main(
    filtered_path: Path = FILTERED_PATH_DEFAULT,
    persist_dir: Path = CHROMA_DIR_DEFAULT,
    sample_out: Path = SAMPLE_OUT_DEFAULT,
    n_total: int = 15000,  # pick 10k–15k per instructions :contentReference[oaicite:13]{index=13}
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> None:
    filtered_path = Path(filtered_path)
    if not filtered_path.exists():
        raise FileNotFoundError(
            f"{filtered_path} not found. Run Task 1 first to generate data/filtered_complaints.csv"
        )

    df = pd.read_csv(filtered_path)

    required_cols = {"complaint_id", "product_category", "clean_narrative"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Filtered dataset missing columns: {missing}. Found: {list(df.columns)}")

    df_sample = stratified_sample(df, label_col="product_category", n_total=n_total)
    sample_out.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(sample_out, index=False)
    print(f"Saved stratified sample to {sample_out} (rows={len(df_sample)})")

    # Embedding model
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Chroma collection
    collection = build_chroma_collection(persist_dir, COLLECTION_NAME)

    # Build docs + metadatas + ids
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict] = []

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Chunking"):
        complaint_id = str(row["complaint_id"])
        product_category = str(row["product_category"])
        product = str(row.get("product", ""))

        issue = str(row.get("issue", ""))
        sub_issue = str(row.get("sub_issue", ""))
        company = str(row.get("company", ""))
        state = str(row.get("state", ""))
        date_received = str(row.get("date_received", ""))

        chunks = chunk_text(str(row["clean_narrative"]), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for ch in chunks:
            doc_id = f"{complaint_id}_{ch.chunk_index}"
            ids.append(doc_id)
            documents.append(ch.text)
            metadatas.append(
                {
                    "complaint_id": complaint_id,
                    "product_category": product_category,
                    "product": product,
                    "issue": issue,
                    "sub_issue": sub_issue,
                    "company": company,
                    "state": state,
                    "date_received": date_received,
                    "chunk_index": ch.chunk_index,
                    "total_chunks": ch.total_chunks,
                }
            )

    # Embed + add (batched)
    batch_size = 256
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding+Indexing"):
        docs_batch = documents[i : i + batch_size]
        ids_batch = ids[i : i + batch_size]
        meta_batch = metadatas[i : i + batch_size]

        embeds = model.encode(docs_batch, show_progress_bar=False, normalize_embeddings=True).tolist()

        collection.add(
            ids=ids_batch,
            documents=docs_batch,
            metadatas=meta_batch,
            embeddings=embeds,
        )

    print(f"Persisted ChromaDB collection '{COLLECTION_NAME}' to {persist_dir}")


if __name__ == "__main__":
    main()
