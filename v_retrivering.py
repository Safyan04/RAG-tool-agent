from iv_vectordb import vector


retriver = vector.as_retriever(
    search_type='mmr',
    search_kwargs={'k':3}
)