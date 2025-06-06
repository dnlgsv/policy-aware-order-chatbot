"""
Multi-Agent Chatbot System
LangGraph-based conversational AI with specialized agents for different order operations.
"""

import json
import os
from typing import Any, Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from .order_service import Order, order_service
from .policy_engine import PolicyDecision, PolicyEngine


class ChatState(BaseModel):
    """State management for the chatbot conversation."""

    messages: list[dict[str, str]] = []
    current_intent: str | None = None
    order_id: str | None = None
    customer_email: str | None = None
    extracted_entities: dict[str, Any] = {}
    policy_decision: PolicyDecision | None = None
    order_data: Order | None = None
    requires_human_handoff: bool = False


class IntentClassification(BaseModel):
    intent: Literal[
        "order_cancellation",
        "order_tracking",
        "general_inquiry",
        "complaint",
        "unknown",
    ]
    confidence: float
    extracted_entities: dict[str, str]


class ChatbotAgent:
    """Base class for specialized chatbot agents."""

    def __init__(self, model_name: str = "gpt-4.1-mini-2025-04-14"):
        self.llm = ChatOpenAI(
            model=model_name, temperature=0.1, api_key=os.getenv("OPENAI_API_KEY")
        )
        self.policy_engine = PolicyEngine()


class RouterAgent(ChatbotAgent):
    """Routes conversations to appropriate specialist agents."""

    def __init__(self, model_name: str = "gpt-4.1-mini-2025-04-14"):
        super().__init__(model_name)
        self.intent_parser = PydanticOutputParser(pydantic_object=IntentClassification)

    def classify_intent(
        self, message: str, conversation_history: list[dict[str, str]]
    ) -> IntentClassification:
        """Classify user intent and extract relevant entities."""

        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant that classifies customer service intents.

        Analyze the user's message and conversation history to determine their intent.

        Available intents:
        - order_cancellation: User wants to cancel an order
        - order_tracking: User wants to track an order or check order status
        - general_inquiry: General questions about policies, services, etc.
        - complaint: User is expressing dissatisfaction or filing a complaint
        - unknown: Intent cannot be determined

        Extract relevant entities:
        - order_id: Any order identification numbers
        - email: Customer email addresses
        - product_name: Specific products mentioned

        Conversation History:
        {history}

        Current Message: {message}

        {format_instructions}
        """)  # If I had real data, I would combine Few-Shot examples and Chain of Thought reasoning here

        history_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]]
        )

        chain = prompt | self.llm | self.intent_parser

        try:
            result = chain.invoke(
                {
                    "message": message,
                    "history": history_text,
                    "format_instructions": self.intent_parser.get_format_instructions(),
                }
            )
            return result
        except Exception:
            # Fallback classification
            return IntentClassification(
                intent="unknown", confidence=0.0, extracted_entities={}
            )


class CancellationAgent(ChatbotAgent):
    """Handles order cancellation requests with policy enforcement."""

    def process_cancellation_request(self, state: ChatState) -> dict[str, Any]:
        """Process order cancellation with policy checks."""

        if not state.order_id:
            return {
                "response": "I'd be happy to help you cancel your order. Could you please provide your order ID?",
                "requires_followup": True,
            }

        # get order information
        order = order_service.get_order(state.order_id)
        if not order:
            return {
                "response": f"I couldn't find an order with ID {state.order_id}. Please check the order ID and try again.",
                "requires_followup": False,
            }
        # Check cancellation policy
        policy_decision = self.policy_engine.evaluate_cancellation(order.model_dump())

        if not policy_decision.allowed:
            return {
                "response": f"I'm sorry, but this order cannot be cancelled. {policy_decision.reason}",
                "requires_followup": False,
                "policy_decision": policy_decision.model_dump(),
            }

        if policy_decision.requires_approval:
            return {
                "response": f"Your cancellation request has been noted. {policy_decision.reason}. A manager will review your request within 24 hours. You'll receive an email confirmation once processed.",
                "requires_followup": False,
                "requires_human_handoff": True,
                "policy_decision": policy_decision.model_dump(),
            }

        # process cancellation
        success = order_service.cancel_order(state.order_id)
        if success:
            refund_timeline = self.policy_engine.order_policy.get_refund_timeline(
                order.status
            )
            return {
                "response": f"Your order {state.order_id} has been successfully cancelled. You can expect your refund within {refund_timeline}. You'll receive a confirmation email shortly.",
                "requires_followup": False,
                "policy_decision": policy_decision.model_dump(),
            }
        else:
            return {
                "response": "I encountered an error while processing your cancellation. Please try again or contact customer support.",
                "requires_followup": False,
            }

    def generate_response(
        self, state: ChatState, cancellation_result: dict[str, Any]
    ) -> str:
        """Generate a natural, empathetic response."""

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful customer service agent. Generate a natural, empathetic response based on the cancellation processing result.

        Be conversational, understanding, and provide clear next steps when applicable.

        Order ID: {order_id}
        Processing Result: {result}

        Generate a response that:
        1. Acknowledges the customer's request
        2. Clearly explains the outcome
        3. Provides next steps if needed
        4. Maintains a helpful, professional tone
        """)

        chain = prompt | self.llm

        response = chain.invoke(
            {
                "order_id": state.order_id,
                "result": json.dumps(cancellation_result, indent=2),
            }
        )

        return response.content


class TrackingAgent(ChatbotAgent):
    """Handles order tracking and status inquiries."""

    def process_tracking_request(self, state: ChatState) -> dict[str, Any]:
        """Process order tracking request."""

        if not state.order_id:
            return {
                "response": "I can help you track your order. Please provide your order ID.",
                "requires_followup": True,
            }

        # Check tracking policy
        policy_decision = self.policy_engine.evaluate_tracking(state.order_id)
        if not policy_decision.allowed:
            return {
                "response": f"I'm unable to provide tracking information. {policy_decision.reason}",
                "requires_followup": False,
            }

        # Get tracking information
        tracking_info = order_service.get_tracking_info(state.order_id)
        if not tracking_info:
            return {
                "response": f"I couldn't find tracking information for order {state.order_id}. Please verify your order ID.",
                "requires_followup": False,
            }

        return {
            "response": "Here's your order tracking information:",
            "tracking_info": tracking_info,
            "requires_followup": False,
        }

    def generate_response(
        self, state: ChatState, tracking_result: dict[str, Any]
    ) -> str:
        """Generate a comprehensive tracking response."""

        if "tracking_info" not in tracking_result:
            return tracking_result["response"]

        tracking_info = tracking_result["tracking_info"]

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful customer service agent providing order tracking information.

        Create a clear, informative response that includes:
        1. Current order status
        2. Current location
        3. Estimated delivery date
        4. Recent tracking events

        Order ID: {order_id}
        Tracking Information: {tracking_info}

        Make the response conversational and easy to understand.
        """)

        chain = prompt | self.llm

        response = chain.invoke(
            {"order_id": state.order_id, "tracking_info": tracking_info.dict()}
        )

        return response.content


class PolicyAwareChatbot:
    """Main chatbot orchestrator using LangGraph."""

    def __init__(self, model_name: str = "gpt-4.1-mini-2025-04-14"):
        self.router = RouterAgent(model_name)
        self.cancellation_agent = CancellationAgent(model_name)
        self.tracking_agent = TrackingAgent(model_name)

        # conversation graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the conversation flow graph."""

        # create the graph
        workflow = StateGraph(dict)

        # add nodes
        workflow.add_node("router", self._route_conversation)
        workflow.add_node("cancellation", self._handle_cancellation)
        workflow.add_node("tracking", self._handle_tracking)
        workflow.add_node("general", self._handle_general)

        # set entry point
        workflow.set_entry_point("router")

        # add conditional edges
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "cancellation": "cancellation",
                "tracking": "tracking",
                "general": "general",
                "end": END,
            },
        )

        # All agents end the conversation
        workflow.add_edge("cancellation", END)
        workflow.add_edge("tracking", END)
        workflow.add_edge("general", END)

        return workflow.compile()

    def _route_conversation(self, state: dict) -> dict:
        """Route the conversation to appropriate agent."""
        current_message = state["messages"][-1]["content"]
        conversation_history = state["messages"][:-1]

        intent_result = self.router.classify_intent(
            current_message, conversation_history
        )

        state["current_intent"] = intent_result.intent
        state["extracted_entities"] = intent_result.extracted_entities

        # Extract order ID if present
        if "order_id" in intent_result.extracted_entities:
            state["order_id"] = intent_result.extracted_entities["order_id"]

        # Extract email if present
        if "email" in intent_result.extracted_entities:
            state["customer_email"] = intent_result.extracted_entities["email"]

        return state

    def _route_decision(self, state: dict) -> str:
        """Decide which agent should handle the conversation."""
        intent = state.get("current_intent", "unknown")

        if intent == "order_cancellation":
            return "cancellation"
        elif intent == "order_tracking":
            return "tracking"
        elif intent in ["general_inquiry", "complaint", "unknown"]:
            return "general"
        else:
            return "end"

    def _handle_cancellation(self, state: dict) -> dict:
        """Handle order cancellation requests."""
        chat_state = ChatState(**state)

        result = self.cancellation_agent.process_cancellation_request(chat_state)

        if result.get("requires_followup"):
            response = result["response"]
        else:
            response = self.cancellation_agent.generate_response(chat_state, result)

        state["messages"].append({"role": "assistant", "content": response})

        if result.get("requires_human_handoff"):
            state["requires_human_handoff"] = True

        return state

    def _handle_tracking(self, state: dict) -> dict:
        """Handle order tracking requests."""
        chat_state = ChatState(**state)

        result = self.tracking_agent.process_tracking_request(chat_state)

        if result.get("requires_followup"):
            response = result["response"]
        else:
            response = self.tracking_agent.generate_response(chat_state, result)

        state["messages"].append({"role": "assistant", "content": response})

        return state

    def _handle_general(self, state: dict) -> dict:
        """Handle general inquiries."""
        current_message = state["messages"][-1]["content"]

        # Generate a helpful general response
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful customer service agent. The user has made a general inquiry.

        Provide a helpful response and guide them to specific services if appropriate.
        Available services:
        - Order cancellation
        - Order tracking
        - Policy information

        User message: {message}

        Be helpful, concise, and guide them to the right service if needed.
        """)

        chain = prompt | self.cancellation_agent.llm
        response = chain.invoke({"message": current_message})

        state["messages"].append({"role": "assistant", "content": response.content})

        return state

    def chat(
        self, message: str, conversation_history: list[dict[str, str]] = None
    ) -> dict[str, Any]:
        """Main chat interface."""
        if conversation_history is None:
            conversation_history = []

        # add user message
        conversation_history.append({"role": "user", "content": message})

        # iinitialize state
        initial_state = {
            "messages": conversation_history,
            "current_intent": None,
            "order_id": None,
            "customer_email": None,
            "extracted_entities": {},
            "requires_human_handoff": False,
        }

        # run the graph
        final_state = self.graph.invoke(initial_state)

        # return the response
        return {
            "response": final_state["messages"][-1]["content"],
            "conversation_history": final_state["messages"],
            "intent": final_state.get("current_intent"),
            "extracted_entities": final_state.get("extracted_entities", {}),
            "requires_human_handoff": final_state.get("requires_human_handoff", False),
        }
