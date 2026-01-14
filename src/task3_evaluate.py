from __future__ import annotations

from pathlib import Path

from .rag import RAGEngine


def main() -> None:
    # Point to your PREBUILT store folder (or sample store for testing)
    # The assignment says tasks 3–4 use the prebuilt full store :contentReference[oaicite:24]{index=24}
    chroma_dir = Path("vector_store/full_chroma")  # <-- put the prebuilt store here
    collection_name = "complaints_full"            # <-- use the prebuilt collection name if provided

    # If you only have your sample for now:
    # chroma_dir = Path("vector_store/sample_chroma")
    # collection_name = "complaints_sample"

    rag = RAGEngine(chroma_dir=chroma_dir, collection_name=collection_name)

    questions = [
        "Why are people unhappy with Credit Cards?",
        "What issues do customers report about money transfers?",
        "Are there complaints about savings account fees?",
        "What are the common personal loan complaint themes?",
        "What fraud-related problems appear across products?",
    ]

    print("| Question | Generated Answer | Retrieved Sources (1-2) | Quality Score (1-5) | Comments/Analysis |")
    print("|---|---|---|---:|---|")

    for q in questions:
        ans, sources = rag.answer(q, k=5)

        # show 1–2 sources in table as required :contentReference[oaicite:25]{index=25}
        s1 = sources[0].metadata if len(sources) > 0 else {}
        s2 = sources[1].metadata if len(sources) > 1 else {}

        sources_cell = (
            f"1) complaint_id={s1.get('complaint_id')}, product_category={s1.get('product_category')}<br>"
            f"2) complaint_id={s2.get('complaint_id')}, product_category={s2.get('product_category')}"
        )

        # You fill these manually in your report after reading outputs:
        quality_score = ""
        comments = ""

        # escape pipes
        safe_q = q.replace("|", "\\|")
        safe_ans = ans.replace("|", "\\|").replace("\n", "<br>")

        print(f"| {safe_q} | {safe_ans} | {sources_cell} | {quality_score} | {comments} |")


if __name__ == "__main__":
    main()
