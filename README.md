# 📚 Research Paper Q&A Agent using LangGraph, RAG, and Groq

An Agentic AI-powered Research Paper Question Answering System built using **LangGraph**, **LangChain**, **ChromaDB**, **Sentence Transformers**, and **Groq LLM**.

This project demonstrates how modern AI agents can combine **memory**, **retrieval**, **reasoning**, **tool usage**, and **self-evaluation** to answer questions accurately from a research knowledge base.

---

## 🚀 Features

✅ Retrieval-Augmented Generation (RAG)

✅ Intelligent Query Routing

✅ Conversational Memory

✅ Semantic Search using Embeddings

✅ Vector Database Integration (ChromaDB)

✅ Tool-Augmented Agent (Web Search)

✅ Automatic Answer Evaluation

✅ Stateful Conversations with LangGraph

✅ Multi-Node Agent Workflow

---

## 🏗️ System Architecture

```text
User Question
      │
      ▼
 Memory Node
      │
      ▼
 Router Node
      │
 ┌────┼────┐
 │    │    │
 ▼    ▼    ▼
Retrieve Tool Memory
 │      │     │
 └──┬───┴─────┘
    ▼
Answer Generator
    ▼
Evaluation Node
    ▼
Save Memory
    ▼
 Final Response
```

---

## 🛠️ Tech Stack

| Component              | Technology            |
| ---------------------- | --------------------- |
| Programming Language   | Python                |
| Agent Framework        | LangGraph             |
| LLM Framework          | LangChain             |
| LLM Provider           | Groq                  |
| Model                  | Llama 3.1 8B Instant  |
| Embeddings             | Sentence Transformers |
| Vector Database        | ChromaDB              |
| Memory                 | MemorySaver           |
| Search Tool            | DuckDuckGo Search     |
| Environment Management | Python Dotenv         |

---

## 📂 Knowledge Base Topics

The agent contains research-focused documents on:

* Attention Mechanism in Transformers
* BERT
* GPT Series
* Retrieval-Augmented Generation (RAG)
* Diffusion Models
* Reinforcement Learning from Human Feedback (RLHF)
* Graph Neural Networks (GNNs)
* Contrastive Learning
* Neural Architecture Search (NAS)
* Federated Learning
* Vision Transformers (ViT)
* Mixture of Experts (MoE)

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/research-paper-qa-agent.git

cd research-paper-qa-agent
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Running the Project

Run:

```bash
python agent.py
```

Example:

```python
print(
    ask(
        "Explain Retrieval-Augmented Generation"
    )
)
```

---

## 🧠 Agent Workflow

### 1. Memory Node

Maintains conversation history and context.

### 2. Router Node

Determines whether the query requires:

* Retrieval
* Tool Usage
* Memory-Based Response

### 3. Retrieval Node

Uses embeddings and ChromaDB to fetch relevant document chunks.

### 4. Tool Node

Performs web searches when additional information is required.

### 5. Answer Node

Generates responses grounded in retrieved context.

### 6. Evaluation Node

Measures faithfulness and reduces hallucinations.

### 7. Save Node

Stores conversation history for future interactions.

---

## 📈 Key Agentic AI Concepts Demonstrated

* Retrieval-Augmented Generation (RAG)
* Memory Management
* State Management
* Semantic Search
* Tool Calling
* Multi-Step Reasoning
* Agent Evaluation
* Workflow Orchestration
* Autonomous Decision Making

---

## 📸 Sample Questions

```text
What is BERT?

Explain the Attention Mechanism.

How does RLHF work?

What are Mixture of Experts models?

Compare GPT and BERT.
```

---

## 🎯 Project Objectives

The primary goal of this project is to build a research assistant capable of:

* Understanding technical questions
* Retrieving relevant information
* Providing grounded responses
* Maintaining conversation context
* Evaluating answer quality

---

## 🔮 Future Improvements

* PDF Upload Support
* Dynamic Knowledge Base Ingestion
* Streamlit Web Interface
* FastAPI Deployment
* Advanced RAG Techniques
* Long-Term Memory
* Multi-Agent Collaboration
* Research Paper Summarization

---

## 📚 References

* LangGraph Documentation
* LangChain Documentation
* ChromaDB Documentation
* Sentence Transformers Documentation
* Groq API Documentation

---

## 👨‍💻 Author

**Enakshy Mondal**

B.Tech Computer Science

Agentic AI Capstone Project

---

## ⭐ Acknowledgements

This project was developed as part of the Agentic AI Capstone Program to demonstrate practical implementation of autonomous AI agents using modern LLM frameworks and Retrieval-Augmented Generation techniques.
