"""
Policy-Aware Order Chatbot
A sophisticated conversational AI system for customer service.
"""

from .agents import PolicyAwareChatbot
from .order_service import OrderService, order_service
from .policy_engine import OrderPolicy, PolicyEngine

__version__ = "1.0.0"
__all__ = [
    "PolicyAwareChatbot",
    "PolicyEngine",
    "OrderPolicy",
    "OrderService",
    "order_service",
]
