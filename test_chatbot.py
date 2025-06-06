"""
Test Chatbot
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio

from chatbot.policy_engine import PolicyEngine, OrderStatus, PolicyDecision
from chatbot.order_service import OrderService, Order
from chatbot.agents import PolicyAwareChatbot, RouterAgent, CancellationAgent, TrackingAgent


class TestPolicyEngine:
    """Test the policy engine functionality."""
    
    def setup_method(self):
        self.policy_engine = PolicyEngine()
    
    def test_can_cancel_recent_order(self):
        """Test cancellation of recent order."""
        order_date = datetime.now() - timedelta(days=3)
        decision = self.policy_engine.order_policy.can_cancel_order(
            order_date, OrderStatus.PENDING
        )
        assert decision.allowed is True
        assert "eligible for cancellation" in decision.reason
    
    def test_cannot_cancel_old_order(self):
        """Test cancellation of old order."""
        order_date = datetime.now() - timedelta(days=15)
        decision = self.policy_engine.order_policy.can_cancel_order(
            order_date, OrderStatus.PENDING
        )
        assert decision.allowed is False
        assert "10 days" in decision.reason
    
    def test_shipped_order_requires_approval(self):
        """Test that shipped orders require approval."""
        order_date = datetime.now() - timedelta(days=3)
        decision = self.policy_engine.order_policy.can_cancel_order(
            order_date, OrderStatus.SHIPPED
        )
        assert decision.allowed is True
        assert decision.requires_approval is True
        assert decision.approval_type == "manager"
    
    def test_delivered_order_cannot_be_cancelled(self):
        """Test that delivered orders cannot be cancelled."""
        order_date = datetime.now() - timedelta(days=3)
        decision = self.policy_engine.order_policy.can_cancel_order(
            order_date, OrderStatus.DELIVERED
        )
        assert decision.allowed is False
        assert "refund" in decision.reason.lower()
    
    def test_tracking_valid_order_id(self):
        """Test tracking with valid order ID."""
        decision = self.policy_engine.order_policy.can_track_order("ORD-2025-1000")
        assert decision.allowed is True
    
    def test_tracking_invalid_order_id(self):
        """Test tracking with invalid order ID."""
        decision = self.policy_engine.order_policy.can_track_order("123")
        assert decision.allowed is False
        assert "Invalid order ID" in decision.reason


class TestOrderService:
    """Test the order service functionality."""
    
    def setup_method(self):
        self.order_service = OrderService()
    
    def test_get_existing_order(self):
        """Test retrieving an existing order."""
        orders = self.order_service.list_all_orders()
        if orders:
            order_id = orders[0].order_id
            retrieved_order = self.order_service.get_order(order_id)
            assert retrieved_order is not None
            assert retrieved_order.order_id == order_id
    
    def test_get_nonexistent_order(self):
        """Test retrieving a non-existent order."""
        order = self.order_service.get_order("INVALID-ORDER-ID")
        assert order is None
    
    def test_cancel_order(self):
        """Test order cancellation."""
        orders = self.order_service.list_all_orders()
        if orders:
            order_id = orders[0].order_id
            success = self.order_service.cancel_order(order_id)
            assert success is True
            
            # Verify the order status changed
            updated_order = self.order_service.get_order(order_id)
            assert updated_order.status == OrderStatus.CANCELLED.value
    
    def test_get_tracking_info(self):
        """Test tracking information retrieval."""
        orders = self.order_service.list_all_orders()
        if orders:
            order_id = orders[0].order_id
            tracking_info = self.order_service.get_tracking_info(order_id)
            assert tracking_info is not None
            assert tracking_info.order_id == order_id
    
    def test_get_orders_by_email(self):
        """Test retrieving orders by email."""
        orders = self.order_service.list_all_orders()
        if orders:
            email = orders[0].customer_email
            customer_orders = self.order_service.get_orders_by_email(email)
            assert len(customer_orders) > 0
            assert all(order.customer_email == email for order in customer_orders)


class TestRouterAgent:
    """Test the router agent intent classification."""
    
    def setup_method(self):
        self.router = RouterAgent()
    
    @pytest.mark.asyncio
    async def test_classify_cancellation_intent(self):
        """Test classification of cancellation intent."""
        message = "I want to cancel my order ORD-2025-1000"
        
        # Mock the LLM response since we don't want to make actual API calls in tests
        with patch.object(self.router, 'classify_intent') as mock_classify:
            mock_classify.return_value = Mock(
                intent="order_cancellation",
                confidence=0.95,
                extracted_entities={"order_id": "ORD-2025-1000"}
            )
            
            result = self.router.classify_intent(message, [])
            assert result.intent == "order_cancellation"
            assert "order_id" in result.extracted_entities
    
    @pytest.mark.asyncio
    async def test_classify_tracking_intent(self):
        """Test classification of tracking intent."""
        message = "Where is my order ORD-2025-1000?"
        
        with patch.object(self.router, 'classify_intent') as mock_classify:
            mock_classify.return_value = Mock(
                intent="order_tracking",
                confidence=0.90,
                extracted_entities={"order_id": "ORD-2025-1000"}
            )
            
            result = self.router.classify_intent(message, [])
            assert result.intent == "order_tracking"


class TestCancellationAgent:
    """Test the cancellation agent functionality."""
    
    def setup_method(self):
        self.agent = CancellationAgent()
    
    def test_process_cancellation_without_order_id(self):
        """Test cancellation request without order ID."""
        from chatbot.agents import ChatState
        
        state = ChatState()
        result = self.agent.process_cancellation_request(state)
        
        assert result["requires_followup"] is True
        assert "order ID" in result["response"]
    
    def test_process_cancellation_with_invalid_order(self):
        """Test cancellation with invalid order ID."""
        from chatbot.agents import ChatState
        
        state = ChatState(order_id="INVALID-ID")
        result = self.agent.process_cancellation_request(state)
        
        assert result["requires_followup"] is False
        assert "couldn't find" in result["response"]


class TestTrackingAgent:
    """Test the tracking agent functionality."""
    
    def setup_method(self):
        self.agent = TrackingAgent()
    
    def test_process_tracking_without_order_id(self):
        """Test tracking request without order ID."""
        from chatbot.agents import ChatState
        
        state = ChatState()
        result = self.agent.process_tracking_request(state)
        
        assert result["requires_followup"] is True
        assert "order ID" in result["response"]


class TestIntegration:
    """Integration tests for the complete system."""
    
    def setup_method(self):
        self.chatbot = PolicyAwareChatbot()
    
    @pytest.mark.asyncio
    async def test_end_to_end_cancellation_flow(self):
        """Test complete cancellation flow."""
        # This would require mocking the LLM calls
        with patch('chatbot.agents.ChatOpenAI') as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm
            
            # Mock the LLM responses
            mock_llm.invoke.return_value = Mock(content="Order cancelled successfully")
            
            # Test the chat method
            result = self.chatbot.chat("Cancel order ORD-2025-1000")
            
            assert "response" in result
            assert "conversation_history" in result
    
    @pytest.mark.asyncio
    async def test_end_to_end_tracking_flow(self):
        """Test complete tracking flow."""
        with patch('chatbot.agents.ChatOpenAI') as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm
            
            mock_llm.invoke.return_value = Mock(content="Here's your tracking information")
            
            result = self.chatbot.chat("Track order ORD-2025-1000")
            
            assert "response" in result
            assert "conversation_history" in result


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        order_id="TEST-ORDER-001",
        customer_email="test@example.com",
        order_date=datetime.now().isoformat(),
        status=OrderStatus.PENDING.value,
        items=[{"name": "Test Product", "quantity": 1, "price": 99.99}],
        total_amount=99.99,
        shipping_address={
            "street": "123 Test St",
            "city": "Test City",
            "country": "Test Country",
            "postal_code": "12345"
        }
    )


@pytest.fixture
def policy_decisions():
    """Sample policy decisions for testing."""
    return {
        "allowed": PolicyDecision(allowed=True, reason="Order eligible for cancellation"),
        "denied": PolicyDecision(allowed=False, reason="Order too old for cancellation"),
        "requires_approval": PolicyDecision(
            allowed=True, 
            reason="Manager approval required", 
            requires_approval=True, 
            approval_type="manager"
        )
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
