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
from tools import add, multiply, divide, check_balance

from agents import make_front_desk_agent, validate_transaction, human_approval_node, execute_transaction


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

# Prepare tools list and mapping (tools are the decorated tool objects)
MATH_TOOLS = [add, multiply, divide]
BANK_TOOLS = [check_balance]  # Only read-only tools for the LLM
TOOLS_BY_NAME = {t.name: t for t in MATH_TOOLS + BANK_TOOLS}


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    # Transaction state
    from_account: str
    to_account: str
    amount: float
    needs_approval: bool
    approved: bool
    # Bank database (stored in file)
    accounts_filepath: str  # Path to JSON file with account balances


def make_tool_node(tools_by_name: Dict[str, Any]):
    """
    Return a node function `tool_node(state: dict) -> dict` that performs tool invocation
    for any tool calls the LLM produced.
    """
    def tool_node(state: dict) -> dict:
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            args = tool_call["args"]
            
            # Inject accounts_filepath for banking tools
            if tool_call["name"] == "check_balance" and "accounts_filepath" not in args:
                args = {**args, "accounts_filepath": state.get("accounts_filepath")}
            
            observation = tool.invoke(args)
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}
    return tool_node


def should_continue(state: MessagesState):
    """Route based on tool calls or transfer intent"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if there are tool calls (getattr returns [] if no tool_calls attribute or empty list)
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:  # This checks if list exists AND is not empty
        print(f"LLM requested {len(tool_calls)} tool call(s), continuing to tool_node.")
        return "tool_node"
    
    # Check if LLM detected transfer intent and set transaction details
    if state.get("from_account") and state.get("to_account") and state.get("amount"):
        print(f"Transfer intent detected: €{state['amount']} from {state['from_account']} to {state['to_account']}")
        return "validate_transaction"
    
    print("No tool calls or transfer intent, ending agent run.")
    return END


def should_require_approval(state: dict) -> str:
    """Check if approval is needed based on amount in state"""
    needs_approval = state.get("needs_approval", False)
    
    if needs_approval:
        print(f"⏸️ Routing to human_approval (amount > €100)")
        return "human_approval"
    
    print(f"✓ Auto-approved, routing to execute_transaction")
    return "execute_transaction"


def build_and_compile_agent(checkpointer=None):
    """Build graph with HITL nodes"""
    if checkpointer is None:
        checkpointer = InMemorySaver()

    model = _init_model()
    model_with_tools = model.bind_tools(MATH_TOOLS + BANK_TOOLS)

    llm_node = make_front_desk_agent(model_with_tools)
    tool_node = make_tool_node(TOOLS_BY_NAME)

    builder = GraphBuilder(MessagesState, checkpointer=checkpointer)
    
    # Add nodes
    builder.add_node("llm_call", llm_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("validate_transaction", validate_transaction)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("execute_transaction", execute_transaction)
    
    # Flow: llm_call -> check_balance (tool) or validate_transaction (transfer) or END
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call", 
        should_continue, 
        ["tool_node", "validate_transaction", END]
    )
    builder.add_edge("tool_node", "llm_call")  # After tool execution, back to LLM
    
    # Transaction flow: validate -> (approve?) -> execute -> llm_call (to inform user)
    builder.add_conditional_edges(
        "validate_transaction",
        should_require_approval,
        ["human_approval", "execute_transaction"]
    )
    builder.add_edge("execute_transaction", "llm_call")  # LLM sends confirmation message
    
    # After human_approval, check if approved or rejected
    def after_human_approval(state: dict) -> str:
        if state.get("approved", False):
            return "execute_transaction"
        else:
            return END  # Rejected - end without executing
    
    builder.add_conditional_edges(
        "human_approval",
        after_human_approval,
        ["execute_transaction", END]
    )

    # Add interrupt BEFORE the human_approval node
    agent = builder.compile(interrupt_before=["human_approval"])
    return agent, builder