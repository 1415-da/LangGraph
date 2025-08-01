from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up environment
os.environ["LANGSMITH_PROJECT"] = "TestProject"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

# Initialize LLM
from langchain.chat_models import init_chat_model
llm = init_chat_model("groq:llama3-8b-8192")

# Define State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define tools
@tool
def add(a: float, b: float):
    """Add two numbers"""
    return a + b

@tool
def explain_machine_learning(description: str):
    """Explain machine learning concepts"""
    return f"Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. {description}"

@tool
def get_recent_ai_news(topic: str = "artificial intelligence"):
    """Get recent news about AI and machine learning"""
    return f"Here are some recent developments in {topic}: AI models are becoming more efficient, new breakthroughs in machine learning algorithms, and increased adoption in various industries."

# Set up tools and tool node
tools = [add, explain_machine_learning, get_recent_ai_news]
tool_node = ToolNode(tools)

# Bind tools to LLM
llm_with_tool = llm.bind_tools(tools)

def call_llm_model(state: State):
    return {"messages": [llm_with_tool.invoke(state['messages'])]}

# Build the graph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

# Node definition
def call_llm_model(state: State):
    return {"messages": [llm_with_tool.invoke(state['messages'])]}

# Graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", call_llm_model)
builder.add_node("tools", ToolNode(tools))

# Add Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
)
builder.add_edge("tools", "tool_calling_llm")

# Compile the graph
graph = builder.compile()

# Test the graph with proper input format
if __name__ == "__main__":
    try:
        # Fix the input format - messages should be a list of BaseMessage objects
        response = graph.invoke({"messages": [HumanMessage(content="What is machine learning")]})
        
        print("Response received successfully!")
        print("Last message content:")
        print(response['messages'][-1].content)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("This might be due to API key issues or network connectivity.") 