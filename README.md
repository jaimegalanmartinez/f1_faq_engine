F1 Knowledge Engine 🏁🚥🏆🏎️💨

A production-style Retrieval-Augmented Generation (RAG) for Formula 1 race history. Ask natural language questions about winners, race results, drivers, and teams and get grounded answers from structured data. Focused specially in race results (1950-2024).

**Key Features**
- RAG Hybrid retrieval (vector + keyword) with structured filters (year, race, driver, constructor, circuit)
- Grounded answers with concise context
- Multilingual-friendly open source embeddings (BGE-M3 from Beijing Academy of AI) (https://huggingface.co/BAAI/bge-m3)
- Open source LLM for answering questions: katanemo/Arch-Router-1.5B:hf-inference (https://huggingface.co/katanemo/Arch-Router-1.5B)

**Tech Stack**
- Python 3.12.3
- Weaviate (vector database)
- SentenceTransformers (`BAAI/BGE-M3`) for embeddings
- Hugging Face Inference Router for LLM responses
- Pandas for data processing

**Data Sources**
- Kaggle “Formula 1 World Championship (1950–2024)” by Rohan Rao

**Limitations And Future Work**
- 2025 season results are not included yet (planned append step).
- Driver and constructor standings are out of scope for the current demo.
