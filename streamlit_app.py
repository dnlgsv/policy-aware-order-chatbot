"""
Streamlit UI for Policy-Aware Order Chatbot
Simple interface for demonstrating chatbot capabilities.
"""

import os
import uuid
from datetime import datetime

import requests
import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Policy-Aware Order Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def get_available_orders():
    """Fetch available orders from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/orders", timeout=5)
        if response.status_code == 200:
            data = response.json()
            orders = data.get("orders", [])
            # Ensure we return a list
            if isinstance(orders, list):
                return orders
            else:
                return []
        return []
    except requests.RequestException:
        return []
    except Exception:
        return []


def send_chat_message(message: str, history: list[dict], session_id: str):
    """Send a chat message to the API."""
    try:
        payload = {
            "message": message,
            "conversation_history": history,
            "session_id": session_id,
        }

        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except requests.RequestException as e:
        return {"error": f"Connection Error: {str(e)}"}


def check_api_health():
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


# Main App
def main():
    st.title("🤖 Policy-Aware Order Chatbot")

    # Check API status
    api_healthy = check_api_health()

    if not api_healthy:
        st.error("🚨 **API Server Not Available**")
        st.markdown("""
        The chatbot backend is not running. Please start it with:
        ```bash
        docker-compose up
        ```
        or
        ```bash
        uv run uvicorn main:app --reload
        ```
        """)
        return
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot can help you with:
        - **Order Cancellations** (policy-aware)
        - **Order Tracking**
        - **Policy Information**
        - **General Inquiries**
        """)

        st.header("📋 Available Test Orders")
        orders = get_available_orders()

        if orders and isinstance(orders, list) and len(orders) > 0:
            st.markdown("Sample orders for testing:")
            for order in orders[:5]:  # Show first 5 orders
                # Handle both dict and object formats
                if isinstance(order, dict):
                    order_date = datetime.fromisoformat(order["order_date"])
                    order_id = order["order_id"]
                    status = order["status"]
                    total_amount = order["total_amount"]
                else:
                    order_date = datetime.fromisoformat(order.order_date)
                    order_id = order.order_id
                    status = order.status
                    total_amount = order.total_amount

                days_ago = (datetime.now() - order_date).days

                st.markdown(f"""
                **{order_id}**
                - Status: `{status}`
                - Age: {days_ago} days
                - Total: ${total_amount:.2f}
                """)
        else:
            st.warning("No orders available")

        st.header("💡 Example Queries")
        st.markdown("""
        - "Cancel my order ORD-2025-1000"
        - "Where is order ORD-2025-1002?"
        - "What's your cancellation policy?"
        - "I need help with my order"
        """)

    # Initialize chat history and session ID
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show additional info if available
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]

                if metadata.get("intent"):
                    st.caption(f"🎯 Intent: {metadata['intent']}")

                if metadata.get("extracted_entities"):
                    st.caption(f"📋 Extracted: {metadata['extracted_entities']}")

                if metadata.get("requires_human_handoff"):
                    st.warning("⚠️ This request requires human review")

    # Chat input
    if prompt := st.chat_input("Ask your question here..."):
        api_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]

        # Add user message to chat history for display
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                response = send_chat_message(
                    prompt,
                    api_history,
                    st.session_state.session_id,
                )

            if "error" in response:
                st.error(f"Error: {response['error']}")
                assistant_message = "I apologize, but I'm experiencing technical difficulties. Please try again."
                metadata = {}
            else:
                assistant_message = response.get(
                    "response", "I apologize, but I couldn't process your request."
                )
                metadata = {
                    "intent": response.get("intent"),
                    "extracted_entities": response.get("extracted_entities"),
                    "requires_human_handoff": response.get("requires_human_handoff"),
                }

            st.markdown(assistant_message)

            # Show metadata
            if metadata.get("intent"):
                st.caption(f"🎯 Intent: {metadata['intent']}")

            if metadata.get("extracted_entities"):
                st.caption(f"📋 Extracted: {metadata['extracted_entities']}")

            if metadata.get("requires_human_handoff"):
                st.warning("⚠️ This request requires human review")

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_message, "metadata": metadata}
        )

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    with col2:
        if st.button("📊 View API Docs"):
            st.markdown(f"[Open API Documentation]({API_BASE_URL}/docs)")


if __name__ == "__main__":
    main()
