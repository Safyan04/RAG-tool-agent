import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Modular RAG Agent", page_icon="⚡", layout="wide")
st.title("⚡ Modular RAG + Tools Assistant")

@st.cache_resource
def get_ready_agent():
    from iv_vectordb import vector
    retriever = vector.as_retriever(search_type='similarity', search_kwargs={'k': 2})
    
    from datetime import datetime
    from langchain_core.tools import tool, create_retriever_tool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq
    from langchain.agents import create_agent

    groq_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    genai_llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)
    llms = groq_llm.with_fallbacks([genai_llm])

    rag_tool = create_retriever_tool(
        retriever,
        'document_search',
        'Use retrieve information from document'
    )

    @tool
    def get_datetime() -> str:
        """current time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_tools = [rag_tool, get_datetime]

    return create_agent(
        model=llms,
        tools=all_tools,
        system_prompt=(
            "You are an intelligent assistant. Follow these rules strictly:\n"
            "1. Always check your primary data tool first to find the answer.\n"
            "2. If the relevant information is found there, use it to respond.\n"
            "3. If the answer is not available in your primary data tool, then and only then use the other available tools.\n"
            "4. Rely on your internal knowledge only as a final resort if all tools fail."
        )
    )

with st.spinner("Initializing Agent & Vector DB..."):
    try:
        agent = get_ready_agent()
        st.success("Agent Ready!")
    except Exception as e:
        st.error(f"Error loading agent: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Apna sawal poochein..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke({
                    "messages": [{"role": "user", "content": prompt}]
                })
                output_text = result["messages"][-1].content
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                st.error(f"Execution Error: {e}")