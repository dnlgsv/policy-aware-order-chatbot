"""
Mock Order Service
Simulates an order management system with realistic data and operations.
"""
# ruff: noqa: S311

import random
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel

from .policy_engine import OrderStatus


class Order(BaseModel):
    order_id: str
    customer_email: str
    order_date: str  # ISO format
    status: str
    items: list[dict[str, Any]]
    total_amount: float
    shipping_address: dict[str, str]
    estimated_delivery: str | None = None
    tracking_number: str | None = None


class TrackingInfo(BaseModel):
    order_id: str
    status: str
    current_location: str
    estimated_delivery: str
    tracking_events: list[dict[str, str]]


class OrderService:
    """Mock order service with realistic looking order data."""

    def __init__(self):
        self._orders = self._generate_sample_orders()

    def _generate_sample_orders(self) -> dict[str, Order]:
        """Generate realistic looking sample orders for testing."""
        sample_orders = {}

        # Recent orders (eligible for cancellation)
        for i in range(5):
            order_id = f"ORD-2025-{1000 + i}"
            order_date = datetime.now() - timedelta(days=random.randint(1, 8))

            order = Order(
                order_id=order_id,
                customer_email=f"customer{i}@example.com",
                order_date=order_date.isoformat(),
                status=random.choice(
                    [OrderStatus.PENDING.value, OrderStatus.PROCESSING.value]
                ),
                items=[
                    {
                        "name": f"Product {i + 1}",
                        "quantity": random.randint(1, 3),
                        "price": round(random.uniform(20, 200), 2),
                    }
                ],
                total_amount=round(random.uniform(50, 500), 2),
                shipping_address={
                    "street": f"{100 + i} Main St",
                    "city": "Berlin",
                    "country": "Germany",
                    "postal_code": f"1011{i}",
                },
            )
            sample_orders[order_id] = order

        # Older orders (not eligible for cancellation)
        for i in range(5, 10):
            order_id = f"ORD-2025-{1000 + i}"
            order_date = datetime.now() - timedelta(days=random.randint(15, 30))

            order = Order(
                order_id=order_id,
                customer_email=f"customer{i}@example.com",
                order_date=order_date.isoformat(),
                status=random.choice(
                    [OrderStatus.SHIPPED.value, OrderStatus.DELIVERED.value]
                ),
                items=[
                    {
                        "name": f"Product {i + 1}",
                        "quantity": random.randint(1, 2),
                        "price": round(random.uniform(30, 150), 2),
                    }
                ],
                total_amount=round(random.uniform(50, 300), 2),
                shipping_address={
                    "street": f"{100 + i} Oak Ave",
                    "city": "Dublin",
                    "country": "Ireland",
                    "postal_code": f"D0{i}",
                },
                tracking_number=f"TRK{1000 + i}",
                estimated_delivery=(order_date + timedelta(days=5)).isoformat(),
            )
            sample_orders[order_id] = order

        return sample_orders

    def get_order(self, order_id: str) -> Order | None:
        """Retrieve order by ID."""
        return self._orders.get(order_id)

    def get_orders_by_email(self, email: str) -> list[Order]:
        """Retrieve all orders for a customer email."""
        return [
            order for order in self._orders.values() if order.customer_email == email
        ]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order if it exists."""
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED.value
            return True
        return False

    def get_tracking_info(self, order_id: str) -> TrackingInfo | None:
        """Get detailed tracking information for an order."""
        order = self.get_order(order_id)
        if not order:
            return None

        # Generate realistic tracking events
        tracking_events = []
        order_date = datetime.fromisoformat(order.order_date)

        if order.status in [
            OrderStatus.PROCESSING.value,
            OrderStatus.SHIPPED.value,
            OrderStatus.DELIVERED.value,
        ]:
            tracking_events.append(
                {
                    "timestamp": order_date.isoformat(),
                    "status": "Order confirmed",
                    "location": "Fulfillment Center",
                }
            )

            if order.status in [OrderStatus.SHIPPED.value, OrderStatus.DELIVERED.value]:
                ship_date = order_date + timedelta(days=1)
                tracking_events.append(
                    {
                        "timestamp": ship_date.isoformat(),
                        "status": "Shipped",
                        "location": "Berlin Distribution Center",
                    }
                )

                if order.status == OrderStatus.DELIVERED.value:
                    delivery_date = ship_date + timedelta(days=3)
                    tracking_events.append(
                        {
                            "timestamp": delivery_date.isoformat(),
                            "status": "Delivered",
                            "location": order.shipping_address["city"],
                        }
                    )

        current_location = "Fulfillment Center"
        if order.status == OrderStatus.SHIPPED.value:
            current_location = "In Transit"
        elif order.status == OrderStatus.DELIVERED.value:
            current_location = order.shipping_address["city"]

        estimated_delivery = (
            order.estimated_delivery or (datetime.now() + timedelta(days=5)).isoformat()
        )

        return TrackingInfo(
            order_id=order_id,
            status=order.status,
            current_location=current_location,
            estimated_delivery=estimated_delivery,
            tracking_events=tracking_events,
        )

    def list_all_orders(self) -> list[Order]:
        """List all orders (for admin/testing purposes)."""
        return list(self._orders.values())


order_service = OrderService()
