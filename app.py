import os
import re
import hashlib
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Modular RAG Agent", page_icon="⚡", layout="wide")
st.title("⚡ Modular RAG + Tools Assistant")

# Is se kam similarity wale chunks "relevant nahi" maane jayenge (0 se 1). Zaroorat ho to tune karein.
SIMILARITY_THRESHOLD = 0.65
NO_RESULT_TEXT = "NO_RELEVANT_RESULTS_IN_PDF"


# ---------------------------------------------------------------------------
# Agent builder (PDF ke hisab se banta hai)
# ---------------------------------------------------------------------------
def build_agent(pdf_bytes: bytes | None):
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    from langchain.agents import create_agent

    groq_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    # gemini-1.5-flash retire ho chuka hai, isliye naya model use kiya hai
    genai_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    llms = groq_llm.with_fallbacks([genai_llm])

    all_tools = []

    # ---- PDF tool (sirf tab jab PDF upload hui ho) ----
    if pdf_bytes:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.vectorstores import InMemoryVectorStore

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_bytes)
            tmp_path = f.name
        try:
            pages = PyPDFLoader(tmp_path).load()
        finally:
            os.remove(tmp_path)

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        ).split_documents(pages)

        # Agar aapke iv_vectordb wale embeddings use karne hain to yahan replace kar dein
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = InMemoryVectorStore(embeddings)
        vectorstore.add_documents(chunks)

        @tool(response_format="content_and_artifact")
        def document_search(query: str):
            """Search the user's uploaded PDF document for relevant information."""
            results = vectorstore.similarity_search_with_score(query, k=3)
            good = [(d, s) for d, s in results if s >= SIMILARITY_THRESHOLD]
            if not good:
                return NO_RESULT_TEXT, []
            content = "\n\n".join(d.page_content for d, _ in good)
            page_nums = sorted({int(d.metadata.get("page", 0)) + 1 for d, _ in good})
            return content, page_nums

        all_tools.append(document_search)

    # ---- Doosra tool ----
    @tool
    def get_datetime() -> str:
        """Get the current date and time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_tools.append(get_datetime)

    source_rule = (
        "\n\nVERY IMPORTANT: the LAST line of every reply must be exactly one of:\n"
        "SOURCE: pdf\n"
        "SOURCE: tool:<tool_name>   (e.g. SOURCE: tool:get_datetime)\n"
        "SOURCE: model\n"
        "Use 'pdf' ONLY if your answer actually uses text returned by document_search. "
        "Use 'tool:<name>' ONLY if your answer actually uses that tool's output. "
        "Use 'model' if you answered from your own knowledge. "
        "If both are used, join them with '+', e.g. SOURCE: pdf+tool:get_datetime.\n"
        "Reply in English or Roman Urdu (Latin letters). Never use Devanagari or Urdu script."
    )

    if pdf_bytes:
        system_prompt = (
            "You are an intelligent assistant.\n"
            "1. For questions about document content, call `document_search` first.\n"
            "2. Do NOT call `document_search` for questions about the current date or time; "
            "use `get_datetime` directly.\n"
            f"3. If `document_search` returns '{NO_RESULT_TEXT}', the PDF has no answer; "
            "then try other tools if useful.\n"
            "4. Use your own knowledge only as a last resort."
            + source_rule
        )
    else:
        system_prompt = (
            "You are an intelligent assistant. Use `get_datetime` for current date/time "
            "questions; otherwise answer from your own knowledge."
            + source_rule
        )

    return create_agent(model=llms, tools=all_tools, system_prompt=system_prompt)


# ---------------------------------------------------------------------------
# Source detection: kis ne jawab diya?
# ---------------------------------------------------------------------------
def to_text(content):
    """Gemini kabhi kabhi content list of blocks deta hai."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def detect_source(messages, output_text):
    """Return (clean_text, label). Model ke SOURCE tag ko tool calls se verify karta hai."""
    pdf_ok = False
    pdf_pages = []
    other_tools = []

    for m in messages:
        if getattr(m, "type", None) != "tool":
            continue
        name = getattr(m, "name", None)
        if name == "document_search":
            if NO_RESULT_TEXT not in str(m.content):
                pdf_ok = True
                if m.artifact:
                    pdf_pages.extend(m.artifact)
        elif name and name not in other_tools:
            other_tools.append(name)

    # SOURCE tag nikalo aur jawab se hata do
    clean_text = output_text
    claimed = None
    match = re.search(r"\n*\s*SOURCE:\s*([^\n]+)\s*$", output_text, re.IGNORECASE)
    if match:
        claimed = match.group(1).strip().lower()
        clean_text = output_text[: match.start()].rstrip()

    use_pdf = False
    used_tools = []
    model_only = False

    if claimed:
        tokens = [t.strip() for t in re.split(r"[+,|]", claimed)]
        use_pdf = "pdf" in tokens and pdf_ok
        used_tools = [
            t.split(":", 1)[1].strip()
            for t in tokens
            if t.startswith("tool:") and t.split(":", 1)[1].strip() in other_tools
        ]
        model_only = "model" in tokens and not use_pdf and not used_tools

    # Tag missing/invalid ho to tool calls se andaza lagao
    if not (use_pdf or used_tools or model_only):
        if other_tools:
            used_tools = other_tools
        elif pdf_ok:
            use_pdf = True

    parts = []
    if use_pdf:
        pages = sorted(set(pdf_pages))
        page_txt = f" (page {', '.join(map(str, pages))})" if pages else ""
        parts.append(f"📄 Answer from the PDF{page_txt}")
    if used_tools:
        parts.append(f"🛠️ Answer from tool: {', '.join(used_tools)}")
    if not parts:
        parts.append("🧠 Not from any tool. Generated by the model itself")

    return clean_text, " | ".join(parts)


# ---------------------------------------------------------------------------
# Sidebar: PDF upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Apni PDF upload karein")
    uploaded = st.file_uploader("PDF select karein", type=["pdf"])
    debug = st.checkbox("Debug mode")
    if st.button("🗑️ Chat clear karein"):
        st.session_state.messages = []
        st.rerun()

pdf_bytes = uploaded.getvalue() if uploaded else None
pdf_key = hashlib.md5(pdf_bytes).hexdigest() if pdf_bytes else "no_pdf"

# PDF badalne par agent dobara banao
if st.session_state.get("pdf_key") != pdf_key:
    st.session_state.pdf_key = pdf_key
    st.session_state.agent = None
    st.session_state.messages = []

if st.session_state.get("agent") is None:
    with st.spinner("PDF process ho rahi hai & Agent ban raha hai..."):
        try:
            st.session_state.agent = build_agent(pdf_bytes)
        except Exception as e:
            st.error(f"Error loading agent: {e}")
            st.stop()

if uploaded:
    st.success(f"✅ PDF ready: {uploaded.name}")
else:
    st.info("Abhi koi PDF upload nahi hui. Sidebar se PDF upload karein ya seedha sawal poochein.")

agent = st.session_state.agent

# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source"):
            st.caption(msg["source"])

if prompt := st.chat_input("Apna sawal poochein..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
                raw_text = to_text(result["messages"][-1].content)
                output_text, source = detect_source(result["messages"], raw_text)

                if debug:
                    with st.expander("Debug: messages"):
                        for m in result["messages"]:
                            st.write(m.type, getattr(m, "name", None), str(m.content)[:300])

                st.markdown(output_text)
                st.caption(source)

                st.session_state.messages.append(
                    {"role": "assistant", "content": output_text, "source": source}
                )
            except Exception as e:
                st.error(f"Execution Error: {e}")