"""Simple API test script for the policy-aware order chatbot."""

import json

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_chat():
    """Test the chat endpoint."""
    print("\nTesting chat endpoint...")

    payload = {"message": "Hello, I need help with my order", "order_id": "1"}

    response = requests.post(
        f"{BASE_URL}/chat",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=10,
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_chat_without_order():
    """Test chat without order ID."""
    print("\nTesting chat without order ID...")

    payload = {"message": "I want to place a new order"}

    response = requests.post(
        f"{BASE_URL}/chat",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=10,
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_orders():
    """Test the orders endpoint."""
    print("\nTesting orders endpoint...")

    response = requests.get(f"{BASE_URL}/orders", timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


if __name__ == "__main__":
    print("Starting API tests...\n")

    tests = [
        ("Health Check", test_health),
        ("Chat with Order", test_chat),
        ("Chat without Order", test_chat_without_order),
        ("Orders List", test_orders),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"Error in {test_name}: {e}")
            results.append((test_name, "ERROR"))

        print("-" * 50)

    print("\nTest Results:")
    for test_name, result in results:
        print(f"{test_name}: {result}")
