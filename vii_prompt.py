from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Utilize available tools when necessary."
    ),
    (
        "human",
        "{input}"
    ),
    MessagesPlaceholder(
        "agent_scratchpad"
    ),
])