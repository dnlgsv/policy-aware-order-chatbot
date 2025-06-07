"""
Policy Engine for Order Management
Centralized business rule enforcement for order operations.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class OrderStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class PolicyDecision(BaseModel):
    allowed: bool
    reason: str
    requires_approval: bool = False
    approval_type: str | None = None


class OrderPolicy:
    """Core policy engine for order management decisions."""

    CANCELLATION_WINDOW_DAYS = 10
    MANAGER_APPROVAL_STATUSES = {OrderStatus.SHIPPED}

    @classmethod
    def can_cancel_order(
        cls, order_date: datetime, order_status: OrderStatus
    ) -> PolicyDecision:
        """
        Determine if an order can be cancelled based on business policies.

        Policy Rules:
        1. Orders placed less than 10 days ago are eligible for cancellation
        2. Orders with "shipped" status require manager approval
        3. Delivered orders cannot be cancelled (refund process required)
        """
        current_time = datetime.now()
        days_since_order = (current_time - order_date).days

        # Map order status to one of the predefined statuses
        if order_status == OrderStatus.CANCELLED:
            return PolicyDecision(allowed=False, reason="Order is already cancelled")

        if order_status == OrderStatus.DELIVERED:
            return PolicyDecision(
                allowed=False,
                reason="Delivered orders cannot be cancelled. Please contact support for refund options.",
            )

        if days_since_order >= cls.CANCELLATION_WINDOW_DAYS:
            return PolicyDecision(
                allowed=False,
                reason=f"Orders can only be cancelled within {cls.CANCELLATION_WINDOW_DAYS} days of placement. This order was placed {days_since_order} days ago.",
            )

        if order_status in cls.MANAGER_APPROVAL_STATUSES:
            return PolicyDecision(
                allowed=True,
                reason="Cancellation requires manager approval due to shipping status",
                requires_approval=True,
                approval_type="manager",
            )

        # Else: order can be cancelled
        return PolicyDecision(allowed=True, reason="Order is eligible for cancellation")

    @classmethod
    def can_track_order(cls, order_id: str) -> PolicyDecision:
        """
        Determine if order tracking is available.

        Policy Rules:
        1. All orders with valid IDs can be tracked
        2. Tracking is available for all statuses except cancelled
        """
        if not order_id or len(order_id) < 5:
            return PolicyDecision(
                allowed=False,
                reason="Invalid order ID provided. Order IDs must be at least 5 characters.",
            )

        return PolicyDecision(allowed=True, reason="Tracking information is available")

    @classmethod
    def get_refund_timeline(cls, order_status: OrderStatus) -> str:
        """Get expected refund timeline based on order status."""
        if order_status in [OrderStatus.PENDING, OrderStatus.PROCESSING]:
            return "5-7 business days"
        elif order_status == OrderStatus.SHIPPED:
            return "7-10 business days after item return"
        else:
            return "Contact customer service for refund timeline"


class PolicyEngine:
    """Main policy engine interface."""

    def __init__(self):
        self.order_policy = OrderPolicy()

    def evaluate_cancellation(self, order_data: dict[str, Any] | Any) -> PolicyDecision:
        """Evaluate if an order cancellation is allowed."""
        # handle both dict and Pydantic model inputs
        if hasattr(order_data, "order_date"):
            # Pydantic model
            order_date_str = order_data.order_date
            order_status = OrderStatus(order_data.status)
        else:
            # Dictionary
            order_date_str = order_data["order_date"]
            order_status = OrderStatus(order_data["status"])

        # convert string to datetime
        order_date = datetime.fromisoformat(order_date_str)

        return self.order_policy.can_cancel_order(order_date, order_status)

    def evaluate_tracking(self, order_id: str) -> PolicyDecision:
        """Evaluate if order tracking is allowed."""
        return self.order_policy.can_track_order(order_id)

    def get_policy_explanation(self, policy_type: str) -> str:
        """Get human-readable policy explanations."""
        explanations = {
            "cancellation": f"""
            Order Cancellation Policy:
            • Orders can be cancelled within {OrderPolicy.CANCELLATION_WINDOW_DAYS} days of placement
            • Shipped orders require manager approval for cancellation
            • Delivered orders cannot be cancelled (refund process applies)
            • Refunds are processed within 5-7 business days for eligible orders
            """,
            "tracking": """
            Order Tracking Policy:
            • All valid orders can be tracked using order ID
            • Tracking includes current status, location, and estimated delivery
            • Historical tracking is available for completed orders
            """,
        }
        return explanations.get(policy_type, "Policy information not available")
