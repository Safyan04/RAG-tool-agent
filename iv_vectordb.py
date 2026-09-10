from langchain_community.vectorstores import Chroma
from i_loader import file
from iii_embedings import embeding

collection = 'ABC'
drct = './abcd'

vector = Chroma.from_documents(
    documents=file,
    embedding=embeding,
    collection_name=collection,
    persist_directory=drct
)