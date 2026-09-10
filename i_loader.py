from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\Safyan Ahmad\Desktop\hybrid_rag_project\y-Daily_English_Vocabulary_Phrases.pdf")

file = loader.load()