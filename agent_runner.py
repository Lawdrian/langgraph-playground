# agent_runner.py
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict
from graph_builder import GraphBuilder
from langgraph.graph import START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, Annotated
from langchain.messages import AnyMessage, ToolMessage, HumanMessage
import operator

# Import model factory and tools
from langchain.chat_models import init_chat_model
from tools import add, multiply, divide  # keep tools.py as-is

from agents import make_llm_call

# Prepare tools list and mapping (tools are the decorated tool objects)
TOOLS = [add, multiply, divide]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def make_tool_node(tools_by_name: Dict[str, Any]):
    """
    Return a node function `tool_node(state: dict) -> dict` that performs tool invocation
    for any tool calls the LLM produced.
    """
    def tool_node(state: dict) -> dict:
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}
    return tool_node


def should_continue(state: MessagesState):
    """
    Conditional used by the graph to decide whether to go to `tool_node` or END.
    Returns "tool_node" when the LLM created tool calls; otherwise returns END.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END


def _init_model():
    #return init_chat_model(
    #    model="gemini-2.5-flash-lite",
    #    model_provider="google_genai",
    #    temperature=0
    #)
    return init_chat_model(
        model="openai/gpt-oss-120b",
        model_provider="groq",
        temperature=0
    )


def build_and_compile_agent(checkpointer=None):
    """
    Build and compile the agent. If no checkpointer is provided, uses SqliteSaver.
    Returns (compiled_agent, builder).
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    model = _init_model()
    model_with_tools = model.bind_tools(TOOLS)

    llm_node = make_llm_call(model_with_tools)
    tool_node = make_tool_node(TOOLS_BY_NAME)

    builder = GraphBuilder(MessagesState, checkpointer=checkpointer)
    builder.add_node("llm_call", llm_node)
    builder.add_node("tool_node", tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")

    agent = builder.compile()
    return agent, builder