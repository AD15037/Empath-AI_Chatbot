from secret_key import azure_openai_key                     #API key to use Azure OpenAI
from secret_key import tavily_key
import os

os.environ["AZURE_OPENAI_API_KEY"] = azure_openai_key
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://21etc-m3br78jg-francecentral.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-35-turbo"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-08-01-preview"
os.environ["TAVILY_API_KEY"] = tavily_key

from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

workflow = StateGraph(state_schema=MessagesState)           # Define a new graph

def call_model(state: MessagesState):                       # Define the function that calls the model
    chain = prompt | model
    response = chain.invoke(state)
    return {"messages": response}

workflow.add_edge(START, "model")                           # Define the (single) node in the graph
workflow.add_node("model", call_model)

memory = MemorySaver()                                      # Add memory
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}          # Thread ID is configured here for a user

prompt = ChatPromptTemplate.from_messages(                  # Prompt Template
    [
        (
            "system",
            "You talk like a psychologist. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

query = "Hi! I'm Atharv."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

count = 10
while count < 10:
    query = input("You : ")
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    count += 1

