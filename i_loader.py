from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("y-Daily_English_Vocabulary_Phrases.pdf")

file = loader.load()