import requests
import json
from typing import List, Dict, Union

# --- Configuration ---
API_BASE_URL = "http://your-api-host:8000"  # Replace with your actual API endpoint

# --- API Interaction Functions ---
def call_triage_api(ticket_description: str) -> Union[str, None]:
    """
    Calls the triage API to predict the best support team for a new ticket.

    Args:
        ticket_description (str): The text content of the new support ticket.

    Returns:
        Union[str, None]: The predicted team name or None if the request fails.
    """
    endpoint = f"{API_BASE_URL}/predict_triage"
    payload = {"text": ticket_description}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json().get("predicted_team")
    except requests.exceptions.RequestException as e:
        print(f"Error calling triage API: {e}")
        return None

def call_solution_api(ticket_query: str) -> Union[List[str], None]:
    """
    Calls the solution recommendation API to get relevant knowledge base articles.

    Args:
        ticket_query (str): The query for the solution (e.g., the ticket description).

    Returns:
        Union[List[str], None]: A list of recommended solution descriptions or None if the request fails.
    """
    endpoint = f"{API_BASE_URL}/recommend_solution"
    payload = {"query": ticket_query}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("recommended_solutions")
    except requests.exceptions.RequestException as e:
        print(f"Error calling solution API: {e}")
        return None

def call_predictive_api(ticket_history: List[str]) -> Union[float, None]:
    """
    Calls the predictive API to get the escalation risk based on a ticket history sequence.

    Args:
        ticket_history (List[str]): A sequence of the last N ticket descriptions.

    Returns:
        Union[float, None]: The predicted escalation risk (0 to 1) or None if the request fails.
    """
    endpoint = f"{API_BASE_URL}/predict_escalation"
    payload = {"ticket_history": ticket_history}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("predicted_escalation_risk")
    except requests.exceptions.RequestException as e:
        print(f"Error calling predictive API: {e}")
        return None

# --- Example Usage ---
def demonstrate_integration():
    """
    Demonstrates how the APIs could be integrated into a simulated ticketing system workflow.
    """
    print("--- Simulating New Ticket Creation ---")
    new_ticket_text = "My account is locked and I can't log in. I've tried resetting my password multiple times, but it keeps saying 'Incorrect credentials'."
    
    # 1. Automated Triage and Routing
    predicted_team = call_triage_api(new_ticket_text)
    if predicted_team:
        print(f"Triage API suggests routing this ticket to: {predicted_team}")
        # In a real system, update the ticket's 'assigned_to' field via its API.
    
    print("\n--- Simulating Agent Dashboard View ---")
    # 2. Solution Recommendations for Agents
    recommended_solutions = call_solution_api(new_ticket_text)
    if recommended_solutions:
        print("DSSM API recommends the following solutions:")
        for i, solution in enumerate(recommended_solutions, 1):
            print(f"{i}. {solution}")
        # In a real system, display these recommendations prominently in the agent's UI.

    print("\n--- Simulating Proactive Monitoring ---")
    # 3. Proactive Monitoring and Alerts
    # Assume a ticket history is retrieved from the database
    example_ticket_history = [
        "First ticket with login issue.",
        "Second ticket still having trouble logging in.",
        "Third ticket with a similar login problem.",
        "Fourth ticket with error 404.",
        "Fifth ticket with login issue, not resolved."
    ]
    escalation_risk = call_predictive_api(example_ticket_history)
    if escalation_risk is not None:
        print(f"Predictive API indicates an escalation risk of: {escalation_risk:.2f}")
        if escalation_risk > 0.8:  # Example threshold
            print("ALERT: High escalation risk detected. Notifying L3 support...")
            # In a real system, trigger an alert or automatically create a high-priority ticket.

if __name__ == "__main__":
    demonstrate_integration()
