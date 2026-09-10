from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2')