from langchain.messages import SystemMessage, AIMessage
from typing import Callable
import json
import os

from utils import load_accounts, save_accounts




def make_front_desk_agent(model_with_tools) -> Callable[[dict], dict]:
    """
    Return a node function `llm_call(state: dict) -> dict` bound to the provided model_with_tools.
    The node returns a dict with updated "messages" and increments "llm_calls".
    If transfer intent detected, also returns transaction details in state.
    """
    def front_desk_agent(state: dict) -> dict:
        print("llm_call called!")
        system = SystemMessage(content="""You are a helpful banking assistant. 
        
You can:
- Help with arithmetic
- Check account balances using the check_balance tool
- Initiate transfers (you'll gather details but human approval may be needed)

When a user wants to transfer money:
1. Ask for from_account, to_account, and amount if not provided
2. Once you have all details, respond with: "TRANSFER_REQUEST: {from_account} -> {to_account}: €{amount}"
   Example: "TRANSFER_REQUEST: checking -> savings: €150"

IMPORTANT: If you see a transfer completion message (e.g., ✅ Transfer completed) in one of the last requests, DO NOT create another TRANSFER_REQUEST. Instead, acknowledge it naturally and ask if they need anything else.
""")
        response = model_with_tools.invoke([system] + state["messages"])
        
        result = {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }
        
        # Parse transfer intent from response
        if hasattr(response, 'content') and "TRANSFER_REQUEST:" in response.content:
            import re
            # Parse: "TRANSFER_REQUEST: checking -> savings: €150"
            match = re.search(r'TRANSFER_REQUEST:\s*(\w+)\s*->\s*(\w+):\s*€(\d+(?:\.\d+)?)', response.content)
            if match:
                result["from_account"] = match.group(1)
                result["to_account"] = match.group(2)
                result["amount"] = float(match.group(3))
                print(f"✓ Parsed transfer: €{result['amount']} from {result['from_account']} to {result['to_account']}")
        
        return result
    return front_desk_agent



def validate_transaction(state: dict) -> dict:
    """
    Check if transaction amount is within automatic approval limit.
    """
    amount = state.get("amount", 0)
    if amount > 100:
        print(f"⚠️ Transaction €{amount} requires approval (> €100)")
        return {"needs_approval": True}
    else:
        print(f"✓ Transaction €{amount} auto-approved (<= €100)")
        return {"needs_approval": False}


def human_approval_node(state: dict) -> dict:
    """Validate that approval was given or handle rejection"""
    print("Human approval node called.")
    approved = state.get("approved", False)
    
    if not approved:
        print("❌ Transaction rejected by user")
        # Clear transaction state and add rejection message
        return {
            "messages": [AIMessage(content="❌ Transaction was rejected and will not be executed.")],
            "from_account": None,
            "to_account": None,
            "amount": None,
            "needs_approval": False,
            "approved": False
        }
    
    print("✅ Transaction approved by user")
    return {}



def execute_transaction(state: dict) -> dict:
    """
    Transfer money between accounts using file-based storage.
    Reads from state, loads file, performs transaction, saves file.
    """
    from_account = state["from_account"]
    to_account = state["to_account"]
    amount = state["amount"]
    filepath = state["accounts_filepath"]
    
    print(f"💸 Executing: Transfer €{amount} from {from_account} to {to_account}")
    
    # Load current balances from file
    account_balances = load_accounts(filepath)
    
    # Validate accounts exist
    if from_account not in account_balances:
        error_msg = f"Error: Account '{from_account}' not found"
        return {"messages": [AIMessage(content=error_msg)]}
    
    if to_account not in account_balances:
        error_msg = f"Error: Account '{to_account}' not found"
        return {"messages": [AIMessage(content=error_msg)]}
    
    # Check balance
    if account_balances[from_account] < amount:
        error_msg = f"Error: Insufficient funds in {from_account}. Current balance: €{account_balances[from_account]}"
        return {"messages": [AIMessage(content=error_msg)]}
    
    # Execute transaction
    account_balances[from_account] -= amount
    account_balances[to_account] += amount
    
    # Save updated balances back to file
    save_accounts(filepath, account_balances)
    
    success_msg = f"""✅ Transfer completed successfully!
- Transferred: €{amount}
- From: {from_account} (new balance: €{account_balances[from_account]})
- To: {to_account} (new balance: €{account_balances[to_account]})"""
    
    return {
        "messages": [AIMessage(content=success_msg)],
        "from_account": None,
        "to_account": None,
        "amount": None,
        "needs_approval": False,
        "approved": False
    }


