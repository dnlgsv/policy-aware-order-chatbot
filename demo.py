"""
Demo Script for Chatbot
Interactive demonstration of the chatbot capabilities.
"""

import asyncio
from datetime import datetime

from dotenv import load_dotenv

from chatbot.agents import PolicyAwareChatbot
from chatbot.order_service import order_service
from chatbot.policy_engine import PolicyEngine

# load environment variables
load_dotenv()


class ChatbotDemo:
    """Interactive demo for the chatbot system."""

    def __init__(self):
        self.chatbot = PolicyAwareChatbot()
        self.policy_engine = PolicyEngine()
        self.conversation_history = []

    def display_welcome(self):
        """Display welcome message and available orders."""
        print("=" * 60)
        print("POLICY-AWARE ORDER CHATBOT DEMO")
        print("=" * 60)
        print("Welcome! I'm your AI customer service assistant.")
        print("I can help you with:")
        print("- Order cancellations")
        print("- Order tracking")
        print("- Policy information")
        print("- General inquiries")
        print()

        # Show available test orders
        orders = order_service.list_all_orders()[:5]  # Show first 5 orders
        print("Sample Orders Available for Testing:")
        print("-" * 40)
        for order in orders:
            days_ago = (datetime.now() - datetime.fromisoformat(order.order_date)).days
            print(f"- {order.order_id} - {order.status} - {days_ago} days old")
        print()
        print(
            "Try asking: 'Cancel order ORD-2025-1000' or 'Track my order ORD-2025-1005'"
        )
        print("Type 'quit' to exit, 'help' for more commands")
        print("-" * 60)

    def display_help(self):
        """Display help information."""
        print("\nAVAILABLE COMMANDS:")
        print("- 'help' - Show this help message")
        print("- 'orders' - List all available test orders")
        print("- 'policies' - Show policy information")
        print("- 'clear' - Clear conversation history")
        print("- 'quit' - Exit the demo")
        print("\nEXAMPLE QUERIES:")
        print("- 'I want to cancel my order ORD-2025-1000'")
        print("- 'Where is my order ORD-2025-1005?'")
        print("- 'What is your cancellation policy?'")
        print("- 'I need help with my order'")
        print()

    def display_orders(self):
        """Display all available orders."""
        orders = order_service.list_all_orders()
        print(f"\nALL AVAILABLE ORDERS ({len(orders)} total):")
        print("-" * 60)
        for order in orders:
            days_ago = (datetime.now() - datetime.fromisoformat(order.order_date)).days
            print(f"Order ID: {order.order_id}")
            print(f"  Status: {order.status}")
            print(f"  Date: {days_ago} days ago")
            print(f"  Customer: {order.customer_email}")
            print(f"  Total: ${order.total_amount:.2f}")
            print()

    def display_policies(self):
        """Display policy information."""
        print("\nCOMPANY POLICIES:")
        print("-" * 30)

        cancellation_policy = self.policy_engine.get_policy_explanation("cancellation")
        tracking_policy = self.policy_engine.get_policy_explanation("tracking")

        print("CANCELLATION POLICY:")
        print(cancellation_policy)
        print("\nTRACKING POLICY:")
        print(tracking_policy)
        print()

    async def process_user_input(self, user_input: str) -> bool:
        """Process user input and return whether to continue."""
        user_input = user_input.strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nThank you for using the Policy-Aware Order Chatbot!")
            return False

        elif user_input.lower() == "help":
            self.display_help()
            return True

        elif user_input.lower() == "orders":
            self.display_orders()
            return True

        elif user_input.lower() == "policies":
            self.display_policies()
            return True

        elif user_input.lower() == "clear":
            self.conversation_history = []
            print("\n🧹 Conversation history cleared!")
            return True

        elif not user_input:
            print("Please enter a message or command.")
            return True

        else:
            # Process with chatbot
            try:
                print("\nProcessing your request...")

                result = self.chatbot.chat(
                    message=user_input,
                    conversation_history=self.conversation_history.copy(),
                )

                # Update conversation history
                self.conversation_history = result["conversation_history"]

                # Display response
                print("\nAssistant:")
                print("-" * 20)
                print(result["response"])

                # Display additional info if available
                if result.get("intent"):
                    print(f"\nDetected Intent: {result['intent']}")

                if result.get("extracted_entities"):
                    entities = result["extracted_entities"]
                    if entities:
                        print(f"Extracted Info: {entities}")

                if result.get("requires_human_handoff"):
                    print("This request requires human review.")

                print()

            except Exception as e:
                print(f"\nError processing request: {str(e)}")
                print("Please try again or contact support.")
                print()

        return True

    async def run_demo(self):
        """Run the interactive demo."""
        self.display_welcome()

        while True:
            try:
                user_input = input("You: ").strip()

                should_continue = await self.process_user_input(user_input)
                if not should_continue:
                    break

            except KeyboardInterrupt:
                print("\n\nDemo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {str(e)}")
                print("Please try again.")


async def run_automated_demo():
    """Run an automated demo with predefined scenarios."""
    print("=" * 60)
    print("AUTOMATED DEMO - POLICY-AWARE ORDER CHATBOT")
    print("=" * 60)

    chatbot = PolicyAwareChatbot()

    # Define test scenarios
    scenarios = [
        {
            "name": "Recent Order Cancellation",
            "message": "I want to cancel my order ORD-2025-1000",
            "description": "Testing cancellation of a recent order (should be allowed)",
        },
        {
            "name": "Old Order Cancellation",
            "message": "Please cancel order ORD-2025-1005",
            "description": "Testing cancellation of an old order (should be denied)",
        },
        {
            "name": "Order Tracking",
            "message": "Where is my order ORD-2025-1002?",
            "description": "Testing order tracking functionality",
        },
        {
            "name": "Policy Inquiry",
            "message": "What is your cancellation policy?",
            "description": "Testing general policy information request",
        },
        {
            "name": "Invalid Order",
            "message": "Track order INVALID-123",
            "description": "Testing handling of invalid order ID",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"User Input: '{scenario['message']}'")
        print("-" * 50)

        try:
            result = chatbot.chat(scenario["message"])

            print("Response:")
            print(result["response"])

            if result.get("intent"):
                print(f"\nIntent: {result['intent']}")

            if result.get("extracted_entities"):
                print(f"Entities: {result['extracted_entities']}")

            if result.get("requires_human_handoff"):
                print("Requires human handoff")

        except Exception as e:
            print(f"Error: {str(e)}")

        print("\n" + "=" * 60)

    print("\nAutomated demo completed!")


def main():
    """Main demo function."""
    print("Policy-Aware Order Chatbot Demo")
    print("Choose demo mode:")
    print("1. Interactive Demo")
    print("2. Automated Demo")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        demo = ChatbotDemo()
        asyncio.run(demo.run_demo())
    elif choice == "2":
        asyncio.run(run_automated_demo())
    else:
        print("Invalid choice. Running interactive demo...")
        demo = ChatbotDemo()
        asyncio.run(demo.run_demo())


if __name__ == "__main__":
    main()
