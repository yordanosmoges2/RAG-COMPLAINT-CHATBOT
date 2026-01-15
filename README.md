 RAG Complaint Chatbot

## Project Overview
This project implements a **Retrieval-Augmented Generation (RAG) chatbot** for analyzing and answering questions about **consumer complaints**.  
The system is designed to help internal stakeholders quickly understand complaint trends across financial products by grounding LLM-generated answers in real complaint narratives.

The project follows a **multi-step pipeline**:
1. Data exploration and preprocessing
2. Text chunking, embedding, and vector storage
3. RAG pipeline with retrieval and generation
4. Interactive chat interface using Gradio

---

## Dataset
The project uses the **CFPB Consumer Complaints Dataset**.

- Raw data is stored unchanged in `data/raw/`
- Only complaints with non-empty narratives are used
- Complaints are filtered to relevant product categories such as:
  - Credit Cards
  - Personal Loans
  - Savings Accounts
  - Money Transfers

---

## Project Structure
RAG-COMPLAINT-CHATBOT/
├── data/
│ ├── raw/
│ │ └── cfpb_complaints.csv
│ ├── processed/
│ │ ├── plots/
│ │ ├── sample_15k.csv
│ │ └── task3_qualitative_evaluation.csv
│ └── filtered_complaints.csv
│
├── notebooks/
│ ├── 01_task1_eda_preprocessing.ipynb
│ ├── 02_task2_chunk_embed_vectorstore.ipynb
│ ├── 03_task3_rag_pipeline_evaluation.ipynb
│ └── 04_task4_gradio_app.ipynb
│
├── src/
│ ├── chunking.py
│ ├── rag.py
│ ├── task1_preprocess.py
│ ├── task2_index_sample.py
│ └── task3_evaluate.py
│
├── vector_store/
│
├── app.py
├── requirements.txt
└── README.md

yaml
Copy code

---

## Task 1: EDA & Preprocessing
- Performed exploratory data analysis (EDA)
- Analyzed product distribution and narrative word counts
- Removed complaints with missing narratives
- Cleaned text (lowercasing, removing special characters, boilerplate text)
- Saved cleaned output as `data/filtered_complaints.csv`

**Notebook:** `01_task1_eda_preprocessing.ipynb`
Task 1 completed: EDA plots + cleaning + filtered dataset output.

---

## Task 2: Chunking, Embedding & Vector Store
- Created a **stratified sample (10,000–15,000 complaints)**
- Split narratives into overlapping chunks:
  - Chunk size: 500 characters
  - Overlap: 50 characters
- Generated embeddings using:
  - `sentence-transformers/all-MiniLM-L6-v2`
- Stored embeddings and metadata in a **ChromaDB persistent vector store**

**Notebook:** `02_task2_chunk_embed_vectorstore.ipynb`
Task 2 completed: stratified sample + chunking + ChromaDB embeddings

---

## Task 3: RAG Pipeline & Evaluation
- Implemented a retriever using vector similarity search (top-k = 5)
- Built a prompt template that restricts the LLM to retrieved context only
- Generated answers using a HuggingFace text-generation model
- Conducted qualitative evaluation with multiple test questions
- Saved results to `task3_qualitative_evaluation.csv`

**Notebook:** `03_task3_rag_pipeline_evaluation.ipynb`
Task 3 completed: RAG retriever, generator, and qualitative evaluation.


---

## Task 4: Interactive Chat Application
- Built an interactive **Gradio chat interface**
- Users can ask questions about complaints
- Answers are generated using the RAG pipeline
- Retrieved source chunks are displayed for transparency
- Includes a clear/reset option

**Notebook:** `04_task4_gradio_app.ipynb`  
**App entry point:** `app.py`
Task 4 completed: Gradio UI with answers and retrieved sources display.

---

## How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Run notebooks (recommended order)
Task 1 notebook

Task 2 notebook

Task 3 notebook

Task 4 notebook

3. Run the chat application
bash
Copy code
python app.py
Technologies Used
Python

Pandas, NumPy, Matplotlib

Sentence-Transformers

ChromaDB

HuggingFace Transformers

Gradio

Key Takeaways
RAG significantly improves trustworthiness by grounding answers in real data

Chunk size and overlap affect retrieval quality

Showing sources increases transparency for end users