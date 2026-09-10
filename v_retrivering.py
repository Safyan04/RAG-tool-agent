from iv_vectordb import vector


retriver = vector.as_retriever(
    search_type='mmr',
    search_kwrgs={'k':3}
)
