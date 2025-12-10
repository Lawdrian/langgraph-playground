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

import uuid
import os
import streamlit as st
from langchain.messages import HumanMessage
from agent_runner import build_and_compile_agent
from typing import Dict, Any

st.set_page_config(page_title="LangGraph Banking Agent", layout="centered")

# Path to accounts file
ACCOUNTS_FILE = os.path.join(os.path.dirname(__file__), "accounts.json")


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
    print("Num Messages to render:", len(msgs))
    for m in msgs:
        # Try to extract a human-readable content
        content = getattr(m, "content", "")
        if not content:
                tool_calls = getattr(m, "tool_calls", None)
                if tool_calls:
                    for tool_call in tool_calls:
                        content += f"[Tool Call: {tool_call.get('name', 'unknown')}]"
        if not content:
            content = str(m)

        role = getattr(m, "type", None) 
        if role:
            out.append(f"{role.upper()}: {content}")
        else:
            out.append(content)
    return out


def run_agent_sync(agent, message: str|None, config):
    """
    Run the agent synchronously and detect if it was interrupted for HITL.
    Returns (result_state, interrupt_info).
    """
    content = {"messages": [HumanMessage(message)]} if message else None
    result = agent.invoke(content, config)

    # Execute the agent
    
    # Check if the graph is paused/interrupted by examining the state snapshot
    thread_id = config["configurable"]["thread_id"]
    state_snapshot = agent.get_state({"configurable": {"thread_id": thread_id}})
    
    # If `next` has values, the graph is waiting at an interrupt point
    if state_snapshot.next:
        return result, {
            "interrupted": True,
            "next_nodes": state_snapshot.next,
            "state": state_snapshot.values
        }
    
    return result, None


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
        st.session_state.state = base_state or {
            "messages": [], 
            "llm_calls": 0,
            "accounts_filepath": ACCOUNTS_FILE
        }
    if "history" not in st.session_state:
        st.session_state.history = checkpoint_state_to_ui_history(st.session_state.state)
    if "pending_interrupt" not in st.session_state:
        st.session_state.pending_interrupt = None

    # Show conversation history FIRST (so it updates before the return)
    if st.session_state.history:
        st.header("Conversation")
        for i, msg in enumerate(st.session_state.history, start=1):
            st.markdown(f"**Turn {i}:** {msg}")

    # Handle pending approval UI
    if st.session_state.pending_interrupt:
        interrupt_state = st.session_state.pending_interrupt["state"]
        amount = interrupt_state.get("amount", 0)
        from_acc = interrupt_state.get("from_account", "?")
        to_acc = interrupt_state.get("to_account", "?")
        
        st.warning(f"⏸️ **Approval Required**")
        st.info(f"Transfer €{amount} from **{from_acc}** to **{to_acc}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Approve", key="approve_btn", type="primary"):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # Update state with approval
                agent.update_state(config, {"approved": True})
                
                # Resume execution (invoke with None to continue from interrupt)
                try:
                    with st.spinner("Executing transaction..."):
                        result, _ = run_agent_sync(agent, None, config)
                    
                    # Update state and history
                    st.session_state.state = result if result else agent.get_state(config).values
                    st.session_state.history = checkpoint_state_to_ui_history(st.session_state.state)
                    st.session_state.pending_interrupt = None
                    st.success("✅ Transaction approved and executed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Execution failed: {e}")
        
        with col2:
            if st.button("❌ Reject", key="reject_btn"):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # Update state with rejection and RESUME execution
                agent.update_state(config, {"approved": False})
                
                try:
                    with st.spinner("Canceling transaction..."):
                        # Resume execution - human_approval_node will handle rejection
                        result, _ = run_agent_sync(agent, None, config)
                    
                    # Update state and history
                    st.session_state.state = result if result else agent.get_state(config).values
                    st.session_state.history = checkpoint_state_to_ui_history(st.session_state.state)
                    st.session_state.pending_interrupt = None
                    st.error("❌ Transaction rejected.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Rejection failed: {e}")
        
        return  # Don't show chat input while waiting for approval

    # Chat input (only if not interrupted)
    prompt = "Enter a question about arithmetic or banking (e.g. 'Transfer €150 from checking to savings')"
    user_input = st.chat_input(prompt)

    if user_input:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        agent.update_state(config, {"accounts_filepath": ACCOUNTS_FILE})
        # Run agent
        try:
            with st.spinner("Agent thinking..."):
                result, interrupt = run_agent_sync(agent, message=user_input, config=config)
        except Exception as e:
            st.error(f"Agent invocation failed: {e}")
            return

        # Handle interruption: Update history BEFORE showing approval UI
        if interrupt:
            print("Interrupt detected in main.py")
            st.session_state.pending_interrupt = interrupt
            st.session_state.state = interrupt["state"]
            # Update history with all messages including the ones that triggered the interrupt
            st.session_state.history = checkpoint_state_to_ui_history(st.session_state.state)
            st.rerun()  # Rerun to show approval buttons with updated history
            return

        # No interrupt: update state normally
        if result and isinstance(result, dict) and "messages" in result:
            st.session_state.state = result
        else:
            checkpoint = load_latest_checkpoint(builder, st.session_state.thread_id)
            if checkpoint:
                st.session_state.state = checkpoint["state"] if isinstance(checkpoint, dict) and "state" in checkpoint else checkpoint

        st.session_state.history = checkpoint_state_to_ui_history(st.session_state.state)
        st.rerun()  # Rerun to show updated conversation


if __name__ == "__main__":
    main()