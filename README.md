# ⚡ RAG Tool Agent

A modular **Retrieval-Augmented Generation (RAG) chatbot** built with LangChain, Streamlit, and a tool-using AI agent. The agent first searches your documents for an answer, and only falls back to other tools or general knowledge if the document doesn't have it.

## Features

- 📄 **Document ingestion & chunking** — loads and splits source documents for retrieval
- 🧠 **Vector embeddings + vector database** — semantic search over your document content
- 🔍 **Retriever tool** — the agent queries the document store as its primary source of truth
- 🛠️ **Extra tools** — e.g. a `get_datetime` tool, easily extendable with more
- 🤖 **LLM agent with fallback** — primary model (Groq) with automatic fallback to Google Gemini if the primary call fails
- 💬 **Streamlit chat UI** — simple, interactive chat interface with message history

## Project Structure

| File | Purpose |
|---|---|
| `i_loader.py` | Loads source documents |
| `ii_chunking.py` | Splits documents into chunks |
| `iii_embedings.py` | Generates embeddings for the chunks |
| `iv_vectordb.py` | Builds/loads the vector database (`vector` retriever source) |
| `v_retrivering.py` | Retrieval logic/testing |
| `vi_tools.py` | Custom tool definitions |
| `vii_prompt.py` | Prompt/system message templates |
| `viii_agent.py` | Agent construction logic |
| `app.py` | Streamlit app — the chatbot UI and agent runtime |

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Safyan04/RAG-tool-agent.git
cd RAG-tool-agent
```

### 2. Install dependencies

```bash
pip install streamlit python-dotenv langchain langchain-core langchain-groq langchain-google-genai
```

> Add any other dependencies your `i_loader.py` → `iv_vectordb.py` pipeline needs (e.g. `langchain-community`, `chromadb`/`faiss-cpu`, `pypdf`, etc.)

### 3. Add your API keys

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

### 4. Run the app

```bash
streamlit run app.py
```

## Notes

- The primary LLM runs on **Groq** for speed, with automatic fallback to **Google Gemini** if the Groq call fails.
- If you see a `model_not_found` (404) error from Groq, check the [Groq model list](https://console.groq.com/docs/models) — some models move to Enterprise-only access over time. This project currently uses `openai/gpt-oss-20b`.
- The agent is instructed to always check the document retriever tool first before using other tools or its own general knowledge.

## License

MIT
