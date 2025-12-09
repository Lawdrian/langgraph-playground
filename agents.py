from langchain.messages import SystemMessage
from typing import Callable


def make_llm_call(model_with_tools) -> Callable[[dict], dict]:
    """
    Return a node function `llm_call(state: dict) -> dict` bound to the provided model_with_tools.
    The node returns a dict with updated "messages" and increments "llm_calls".
    LangGraph will automatically checkpoint the state after this node completes.
    """
    def llm_call(state: dict) -> dict:
        print("llm_call called!")
        system = SystemMessage(content="You are a helpful assistant tasked with arithmetic.")
        response = model_with_tools.invoke([system] + state["messages"])
        print(response)
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }
    return llm_call