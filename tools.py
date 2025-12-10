from langchain.tools import tool
import json


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


@tool
def check_balance(accounts_filepath: str, account: str) -> str:
    """
    Check the balance of an account.
    
    Args:
        accounts_filepath: Path to JSON file containing account balances
        account: Account name (checking, savings, investment)
    
    Returns:
        Current balance
    """
    # Load balances from file
    with open(accounts_filepath, 'r') as f:
        account_balances = json.load(f)
    
    if account not in account_balances:
        return f"Error: Account '{account}' not found"
    
    return f"Balance in {account}: €{account_balances[account]}"
