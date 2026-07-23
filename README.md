# 🚀 CoreAssist

> **AI-Powered Retrieval-Augmented 5G Packet Core Engineering Assistant**

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![License](https://img.shields.io/badge/License-MIT-yellow)

</p>

### AI-Powered 5G Core Engineering Assistant

Built with Streamlit • PostgreSQL/pgvector • OpenAI GPT-4o-mini • BAAI/bge-reranker-base • Docker

**Developed by Rolando Bancod**

---


CoreAssist is a Retrieval-Augmented Generation (RAG) application that answers technical questions about the **3GPP TS 23.501 – System Architecture for the 5G Core (Release 19)**.

Instead of relying solely on a Large Language Model, CoreAssist retrieves relevant sections from the official 3GPP specification, reranks them using a cross-encoder model, and generates accurate, evidence-based responses using OpenAI GPT.

The project demonstrates an end-to-end AI engineering workflow that includes semantic retrieval, cross-encoder reranking, automated evaluation, operational monitoring, user feedback collection, and containerized deployment using Docker.

---


## 📚 Table of Contents

- [Features](#-features)
- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Application Workflow](#-application-workflow)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Evaluation](#-evaluation)
- [Monitoring](#-monitoring)
- [Running with Docker](#-running-with-docker)
- [Local Installation](#-local-installation)
- [Example Questions](#-example-questions)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---


## ✨ Features

- 🔍 Semantic search using PostgreSQL + pgvector
- 📚 Knowledge base built from 3GPP TS 23.501 Release 19
- 🤖 OpenAI GPT-4o-mini response generation
- 🧠 Query rewriting using conversation history
- 🎯 Cross-encoder reranking (BAAI/bge-reranker-base)
- 💬 Multi-turn conversational interface
- 📊 Real-time monitoring dashboard
- 👍 User feedback collection
- 📈 Retrieval and LLM evaluation framework
- 🐳 Dockerized deployment
- ⚡ CPU-optimized PyTorch runtime

---

# 📖 Overview

Large Language Models often struggle with specialized technical domains because they rely primarily on their pre-trained knowledge.

CoreAssist addresses this limitation using Retrieval-Augmented Generation (RAG). Relevant sections of the official 3GPP specification are retrieved from a vector database, reranked using a cross-encoder, and supplied as context to the language model before generating the final answer.

The project demonstrates how modern AI applications can combine information retrieval, LLMs, evaluation, monitoring, and user feedback into a production-style system.

---

# 🏗 System Architecture

```mermaid
flowchart TD
    A[User] --> B[Streamlit UI]

    B --> C[Conversation Memory]
    C --> D[Query Rewriter]

    D --> E[Embedding Model<br/>all-MiniLM-L6-v2]
    E --> F[(PostgreSQL + pgvector)]

    F --> G[Top-k Retrieved Chunks]
    G --> H[Cross-Encoder Reranker<br/>BAAI/bge-reranker-base]

    H --> I[Top Ranked Context]
    I --> J[OpenAI GPT-4o-mini]

    J --> K[Answer]
    K --> B

    J --> L[(Monitoring Database)]
    B --> L
    L --> M[Monitoring Dashboard]
```

---

# 🔄 Application Workflow

```text
User Question
      │
      ▼
Conversation Memory
      │
      ▼
Query Rewriter
      │
      ▼
Vector Search (pgvector)
      │
      ▼
Cross-Encoder Reranker
      │
      ▼
GPT-4o-mini
      │
      ▼
Final Answer
      │
      ├────────► Monitoring
      │
      └────────► PostgreSQL Metrics
```

---

# 🛠 Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.11 |
| UI | Streamlit |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | all-MiniLM-L6-v2 |
| Reranker | BAAI/bge-reranker-base |
| Vector Database | PostgreSQL + pgvector |
| Database Driver | psycopg |
| Monitoring | OpenTelemetry + PostgreSQL |
| Containerization | Docker & Docker Compose |
| Package Manager | uv |

---

# 📁 Project Structure

```text
coreassist-5g-rag/

├── app/
│   ├── CoreAssist.py
│   └── pages/
│
├── database/
│
├── ingestion/
│
├── monitoring/
│
├── rag/
│
├── retrieval/
│
├── evaluation/
│
├── scripts/
│
├── data/
│
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

# 📊 Evaluation

## Retrieval Performance

The retrieval pipeline was evaluated using a curated dataset of telecom-related questions against the 3GPP TS 23.501 knowledge base.

| Metric | Score |
|---------|------:|
| Hit@1 | 0.650 |
| Hit@3 | 0.900 |
| Hit@5 | 1.000 |
| MRR | 0.770 |

---

## LLM Evaluation

The generated answers were evaluated using an automated LLM judge against a curated evaluation dataset.

Overall answer quality score:

**4.08 / 5**

Evaluation criteria included:

- Correctness
- Completeness
- Relevance
- Grounding in retrieved context

---

# 📈 Monitoring

CoreAssist includes built-in operational monitoring.

Metrics collected include:

- Response latency
- Token usage
- Retrieved document count
- Context length
- Top reranker score
- User feedback
- Helpful response rate

The Streamlit monitoring dashboard visualizes these metrics in real time.

---

# 🐳 Running with Docker

### Build

```bash
docker compose build
```

### Start the application

```bash
docker compose up
```

Then open:

```
http://localhost:8501
```

---

## Prerequisites

- Python 3.11+
- Docker Desktop (required only for Docker deployment)
- OpenAI API key


# 🔑 Environment Variables

Create a `.env` file in the project root using the following template:

```text
OPENAI_API_KEY=your_openai_api_key

DATABASE_URL=postgresql://coreassist:coreassist@localhost:5432/coreassist

OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

RETRIEVAL_LIMIT=5
```

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `DATABASE_URL` | PostgreSQL connection string |
| `OPENAI_MODEL` | OpenAI model used for answer generation |
| `EMBEDDING_MODEL` | Sentence Transformer model used to generate embeddings |
| `RETRIEVAL_LIMIT` | Number of retrieved documents passed to the reranker |

> **Note:** Do not commit your `.env` file to GitHub. If you want to provide a template for other users, create a `.env.example` file containing placeholder values (without any secrets).

---

# ⚙ Local Installation

Clone the repository:

```bash
git clone https://github.com/robanc/coreassist-5g-rag.git
```

Install dependencies:

```bash
uv sync
```

Run the application:

```bash
streamlit run app/CoreAssist.py
```

---

# 💬 Example Questions

- What is the purpose of the AMF?
- How does SMF communicate with the UPF?
- Explain the N2 interface.
- What functions does the NRF provide?
- Describe the UE registration procedure.

---

# 📸 Screenshots

## Home Page

![Home Page](screenshots/home.png)

---

## Question Answering

![Question Answering](screenshots/question-answer.png)

---

## Monitoring Dashboard

![Monitoring Dashboard](screenshots/dashboard.png)

---

# 🚀 Future Improvements

- Hybrid BM25 + vector retrieval
- Support additional 3GPP specifications (e.g., TS 23.502)
- Source citations in generated responses
- Streaming responses
- Kubernetes deployment
- CI/CD with GitHub Actions

---

# 🙏 Acknowledgements

- 3GPP
- OpenAI
- PostgreSQL
- pgvector
- Streamlit
- Hugging Face
- LLM Zoomcamp (DataTalks.Club)

---

# 📄 License

This project is released under the MIT License.

