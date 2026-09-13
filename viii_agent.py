from langchain.agents import create_agent
from vi_tools import  all_tools, llms

agent = create_agent(
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

# while True:
#     query = input("\nYou: ")
#     if query.lower() == 'exit':
#         break
#     result = agent.invoke({
#     "messages": [
#         {
#             "role": "user",
#             "content": query
#         }
#     ]
# })
#     print(result["messages"][-1].content)