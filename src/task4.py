from __future__ import annotations

from pathlib import Path
import gradio as gr

from src.rag import RAGEngine, format_sources_md


# Use the prebuilt full store for Tasks 3–4 :contentReference[oaicite:28]{index=28}
CHROMA_DIR = Path("vector_store/full_chroma")   # <-- put prebuilt store folder here
COLLECTION_NAME = "complaints_full"             # <-- set to the name you were given

# For local testing using your Task-2 sample store:
# CHROMA_DIR = Path("vector_store/sample_chroma")
# COLLECTION_NAME = "complaints_sample"


rag = RAGEngine(
    chroma_dir=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    llm_model_name="google/flan-t5-base",
    device=-1,  # CPU
)


def ask(question: str, chat_history):
    if not question or not question.strip():
        return chat_history, "", ""

    answer, retrieved = rag.answer(question, k=5)
    sources_md = format_sources_md(retrieved, max_sources=5)

    chat_history = chat_history + [(question, answer)]
    return chat_history, "", sources_md


with gr.Blocks(title="CrediTrust Complaint RAG Chatbot") as demo:
    gr.Markdown("# CrediTrust Complaint RAG Chatbot")
    gr.Markdown(
        "Ask questions about customer complaints. The answer is grounded in retrieved complaint excerpts."
    )

    chatbot = gr.Chatbot(label="Chat")
    sources_box = gr.Markdown(label="Sources (retrieved chunks)")

    with gr.Row():
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g., Why are people unhappy with Credit Cards?",
            scale=4,
        )
        submit = gr.Button("Ask", scale=1)
        clear = gr.Button("Clear", scale=1)

    submit.click(ask, inputs=[question, chatbot], outputs=[chatbot, question, sources_box])
    question.submit(ask, inputs=[question, chatbot], outputs=[chatbot, question, sources_box])

    def _clear():
        return [], "", ""

    clear.click(_clear, outputs=[chatbot, question, sources_box])

demo.launch()
