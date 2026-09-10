from langchain_core.tools.retriever import create_retriever_tool
from datetime import datetime
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from v_retrivering import retriver
from dotenv import load_dotenv

load_dotenv()

genai_llm = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')
groq_llm = ChatGroq(model="llama-3.1-8b-instant")

llms = groq_llm.with_fallbacks([genai_llm])

rag_tool = create_retriever_tool(
    retriver,
    'document_search',
    'Use retrive information from document'
)

@tool
def get_datetime() -> str:
    """current time"""
    return datetime.now().strftime("%y-%m-%d %H:%M:%S")

all_tools = [rag_tool,get_datetime]