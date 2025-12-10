import os
import json

def load_accounts(filepath: str) -> dict:
    """Load account balances from JSON file"""
    if not os.path.exists(filepath):
        # Initialize with default accounts if file doesn't exist
        default_accounts = {
            "checking": 500.0,
            "savings": 1000.0,
            "investment": 2000.0
        }
        save_accounts(filepath, default_accounts)
        return default_accounts
    
    with open(filepath, 'r') as f:
        return json.load(f)


def save_accounts(filepath: str, accounts: dict) -> None:
    """Save account balances to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(accounts, f, indent=2)