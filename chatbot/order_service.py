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

        # define specific orders that match test case expectations
        test_order_configs = [
            # ORD-2025-1000: Recent pending order (cancellable)
            {
                "id": "ORD-2025-1000",
                "days_ago": 2,
                "status": OrderStatus.PENDING.value,
                "customer": "customer0@example.com",
            },
            # ORD-2025-1001: Recent processing order (cancellable)
            {
                "id": "ORD-2025-1001",
                "days_ago": 5,
                "status": OrderStatus.PROCESSING.value,
                "customer": "customer1@example.com",
            },
            # ORD-2025-1002: Recent pending order (cancellable)
            {
                "id": "ORD-2025-1002",
                "days_ago": 3,
                "status": OrderStatus.PENDING.value,
                "customer": "customer2@example.com",
            },
            # ORD-2025-1003: Recent processing order (cancellable)
            {
                "id": "ORD-2025-1003",
                "days_ago": 7,
                "status": OrderStatus.PROCESSING.value,
                "customer": "customer3@example.com",
            },
            # ORD-2025-1004: Recent pending order (cancellable)
            {
                "id": "ORD-2025-1004",
                "days_ago": 1,
                "status": OrderStatus.PENDING.value,
                "customer": "customer4@example.com",
            },
            # ORD-2025-1005: Old order (not cancellable - too old)
            {
                "id": "ORD-2025-1005",
                "days_ago": 16,
                "status": OrderStatus.DELIVERED.value,
                "customer": "customer5@example.com",
            },
            # ORD-2025-1006: Shipped order within cancellation window (requires approval)
            {
                "id": "ORD-2025-1006",
                "days_ago": 8,
                "status": OrderStatus.SHIPPED.value,
                "customer": "customer6@example.com",
            },
            # ORD-2025-1007: Old shipped order (not cancellable - too old)
            {
                "id": "ORD-2025-1007",
                "days_ago": 20,
                "status": OrderStatus.DELIVERED.value,
                "customer": "customer7@example.com",
            },
            # ORD-2025-1008: Recent shipped order (requires approval)
            {
                "id": "ORD-2025-1008",
                "days_ago": 6,
                "status": OrderStatus.SHIPPED.value,
                "customer": "customer8@example.com",
            },
            # ORD-2025-1009: Old delivered order (not cancellable)
            {
                "id": "ORD-2025-1009",
                "days_ago": 25,
                "status": OrderStatus.DELIVERED.value,
                "customer": "customer9@example.com",
            },
        ]

        for i, config in enumerate(test_order_configs):
            order_date = datetime.now() - timedelta(days=config["days_ago"])

            order = Order(
                order_id=config["id"],
                customer_email=config["customer"],
                order_date=order_date.isoformat(),
                status=config["status"],
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
                    "city": "Berlin" if i < 5 else "Dublin",
                    "country": "Germany" if i < 5 else "Ireland",
                    "postal_code": f"1011{i}" if i < 5 else f"D0{i}",
                },
                tracking_number=f"TRK{1000 + i}"
                if config["status"]
                in [OrderStatus.SHIPPED.value, OrderStatus.DELIVERED.value]
                else None,
                estimated_delivery=(order_date + timedelta(days=5)).isoformat()
                if config["status"]
                in [OrderStatus.SHIPPED.value, OrderStatus.DELIVERED.value]
                else None,
            )
            sample_orders[config["id"]] = order

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
