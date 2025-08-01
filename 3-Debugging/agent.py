from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"]="TestProject"

from langchain.chat_models import init_chat_model
llm=init_chat_model("groq:llama3-8b-8192")

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def make_tool_graph():
    ## Graph With tool Call
    from langchain_core.tools import tool

    @tool
    def add(a:float,b:float):
        """Add two numbers"""
        return a+b
    
    @tool
    def get_recent_ai_news(topic: str = "artificial intelligence"):
        """Get recent news about AI and machine learning"""
        return f"Here are some recent developments in {topic}: AI models are becoming more efficient, new breakthroughs in machine learning algorithms, and increased adoption in various industries."
    
    tools=[add, get_recent_ai_news]
    tool_node=ToolNode(tools)

    llm_with_tool=llm.bind_tools(tools)

    def call_llm_model(state:State):
        return {"messages":[llm_with_tool.invoke(state['messages'])]}
    

        ## Grpah
    builder=StateGraph(State)
    builder.add_node("tool_calling_llm",call_llm_model)
    builder.add_node("tools",ToolNode(tools))

    ## Add Edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition
    )
    builder.add_edge("tools","tool_calling_llm")

    ## compile the graph
    graph=builder.compile()
    return graph

tool_agent=make_tool_graph()