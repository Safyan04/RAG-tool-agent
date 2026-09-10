from i_loader import file
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=60
    )
chunk = loader.split_documents(file)