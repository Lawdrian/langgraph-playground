"""
main.py — Streamlit UI and wiring
graph_builder.py — constructs and compiles the langgraph StateGraph
tools.py — tool definitions (pure functions, small wrappers)
agent_runner.py — functions to invoke compiled agent and return outputs
state_store.py — tiny wrapper around st.session_state
visualize.py — generates PNG bytes for st.image
config.py — env var parsing
requirements.txt, README.md, tests under tests/
"""

import asyncio
import uuid
import streamlit as st
from copy import deepcopy
from langchain.messages import HumanMessage
from agent_runner import build_and_compile_agent
from typing import Optional, Dict, Any

st.set_page_config(page_title="LangGraph Arithmetic Agent", layout="centered")


@st.cache_resource
def get_agent_and_builder():
    return build_and_compile_agent()


def load_latest_checkpoint(builder, thread_id):
    saver = getattr(builder, "_checkpointer", None)
    if not saver:
        return None

    try:
        checkpoints = saver.list_checkpoints(thread_id)
        if not checkpoints:
            return None
        latest = checkpoints[0]
        return latest.get("checkpoint", latest)
    except Exception:
        return None


def checkpoint_state_to_ui_history(state: Dict[str, Any]) -> list:
    """
    Convert the LangGraph state (which may contain LangChain message objects) into
    a simple list of text lines for display and persistence in Streamlit session state.
    """
    out = []
    msgs = state.get("messages", [])
    for m in msgs:
        # Try to extract a human-readable content
        content = getattr(m, "content", None) or str(m)
        out.append(content)
    return out


async def run_agent_with_events(agent, invoke_state, config):
    """
    Stream LangGraph events; capture latest state and any interruption checkpoint.
    """
    if not hasattr(agent, "astream_events"):
        return None, None

    latest_state = None
    pending_interrupt = None

    async for event in agent.astream_events(invoke_state, config=config, version="v1"):
        data = event.get("data") or {}
        # states can show up as 'value' or 'state' depending on stream_mode
        if "value" in data:
            latest_state = data["value"]
        elif "state" in data:
            latest_state = data["state"]

        checkpoint = data.get("checkpoint") or event.get("checkpoint")
        if checkpoint:
            pending_interrupt = {
                "checkpoint": checkpoint,
                "checkpoint_id": data.get("checkpoint_id") or event.get("checkpoint_id"),
                "run_id": (event.get("metadata") or {}).get("run_id") or event.get("run_id"),
            }

    return latest_state, pending_interrupt


def main():
    agent, builder = get_agent_and_builder()

    # Render graph image (optional)
    try:
        png = builder.render_graph(agent)
        st.image(png)
    except Exception:
        pass

    # Ensure a per-session thread id for checkpointer
    if "thread_id" not in st.session_state:
        print("Initialize thread_id")
        st.session_state.thread_id = str(uuid.uuid4())
    if "state" not in st.session_state:
        checkpoint = load_latest_checkpoint(builder, st.session_state.thread_id)
        base_state = checkpoint["state"] if isinstance(checkpoint, dict) and "state" in checkpoint else checkpoint
        st.session_state.state = base_state or {"messages": [], "llm_calls": 0}
    if "history" not in st.session_state:
        st.session_state.history = checkpoint_state_to_ui_history(st.session_state.state)
    if "pending_interrupt" not in st.session_state:
        st.session_state.pending_interrupt = None

    # Chat input (change prompt if we’re resuming)
    prompt = (
        "Enter a question about arithmetic (e.g. 'What is 2+2?')"
        if not st.session_state.pending_interrupt
        else "Provide the human response to resume the paused run"
    )
    user_input = st.chat_input(prompt)

    if user_input:
        # Decide which base state to use: fresh vs. interrupted checkpoint
        base_state = None
        if st.session_state.pending_interrupt:
            ckpt = st.session_state.pending_interrupt.get("checkpoint")
            base_state = ckpt.get("state") if isinstance(ckpt, dict) else ckpt

        invoke_state = deepcopy(base_state or st.session_state.state)
        invoke_state["messages"] = list(invoke_state.get("messages", []))
        invoke_state["messages"].append(HumanMessage(content=user_input))

        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        result = None
        interrupt = None

        # Try async streaming first to catch HITL pauses
        try:
            result, interrupt = asyncio.run(
                run_agent_with_events(agent, invoke_state, config)
            )
        except Exception:
            result, interrupt = None, None  # fall back

        # Fallback to the original synchronous invoke when no streaming result/interrupt
        if result is None and interrupt is None:
            try:
                result = agent.invoke(invoke_state, config=config)
            except Exception as e:
                st.error(f"Agent invocation failed: {e}")
                return

        # Handle interruption: stash checkpoint and show UI hint
        if interrupt:
            st.session_state.pending_interrupt = interrupt
            st.session_state.state = interrupt["checkpoint"].get(
                "state", interrupt["checkpoint"]
            )
            st.session_state.history = checkpoint_state_to_ui_history(
                st.session_state.state
            )
            st.info("Run paused for human input. Submit your response to resume.")
            return

        # Clear pending interrupt on success
        st.session_state.pending_interrupt = None

        # Persist final state (result or latest checkpoint)
        if result and isinstance(result, dict) and "messages" in result:
            st.session_state.state = result
        else:
            checkpoint = load_latest_checkpoint(builder, st.session_state.thread_id)
            if checkpoint:
                st.session_state.state = checkpoint["state"] if isinstance(checkpoint, dict) and "state" in checkpoint else checkpoint

        st.session_state.history = checkpoint_state_to_ui_history(st.session_state.state)

    # Show conversation history
    if st.session_state.history:
        st.header("Conversation")
        for i, msg in enumerate(st.session_state.history, start=1):
            st.markdown(f"**Turn {i}:** {msg}")


if __name__ == "__main__":
    main()