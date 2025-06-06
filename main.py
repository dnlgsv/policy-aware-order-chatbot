"""
Policy-Aware Order Chatbot API
FastAPI application providing conversational AI for order management.
"""

import logging
from datetime import datetime
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chatbot.agents import PolicyAwareChatbot
from chatbot.order_service import order_service
from chatbot.policy_engine import PolicyEngine

# load environment variables
load_dotenv()

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Policy-Aware Order Chatbot",
    description="Conversational AI for customer service with policy enforcement",
    version="1.0.0",
)


# initialize chatbot
chatbot = PolicyAwareChatbot()
policy_engine = PolicyEngine()


# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict[str, str]] | None = []
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    conversation_history: list[dict[str, str]]
    intent: str | None
    extracted_entities: dict[str, Any]
    requires_human_handoff: bool
    session_id: str | None


class OrderCancellationRequest(BaseModel):
    reason: str | None = None


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Policy-Aware Order Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "orders": "/orders/{order_id}",
            "cancel": "/orders/{order_id}/cancel",
            "tracking": "/orders/{order_id}/tracking",
            "health": "/health",
        },
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chatbot conversation endpoint."""
    try:
        logger.info(f"Chat request: {request.message}")

        result = chatbot.chat(
            message=request.message, conversation_history=request.conversation_history
        )

        response = ChatResponse(
            response=result["response"],
            conversation_history=result["conversation_history"],
            intent=result["intent"],
            extracted_entities=result["extracted_entities"],
            requires_human_handoff=result["requires_human_handoff"],
            session_id=request.session_id,
        )

        logger.info(
            f"Chat response: {result['intent']} - {result['response'][:100]}..."
        )
        return response

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Chat processing failed: {str(e)}"
        ) from e


@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    """Get order information by ID."""
    order = order_service.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    return order


@app.post("/orders/{order_id}/cancel")
async def cancel_order(order_id: str, request: OrderCancellationRequest = None):
    """Cancel an order with policy enforcement."""
    # Get order
    order = order_service.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    # Check policy
    policy_decision = policy_engine.evaluate_cancellation(order.dict())

    if not policy_decision.allowed:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Cancellation not allowed",
                "reason": policy_decision.reason,
                "policy_decision": policy_decision.dict(),
            },
        )

    if policy_decision.requires_approval:
        return {
            "message": "Cancellation request submitted for approval",
            "reason": policy_decision.reason,
            "approval_required": True,
            "estimated_processing_time": "24 hours",
        }

    # process cancellation
    success = order_service.cancel_order(order_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cancel order")

    return {
        "message": "Order cancelled successfully",
        "order_id": order_id,
        "refund_timeline": policy_engine.order_policy.get_refund_timeline(order.status),
    }


@app.get("/orders/{order_id}/tracking")
async def get_tracking(order_id: str):
    """Get order tracking information."""
    # check policy
    policy_decision = policy_engine.evaluate_tracking(order_id)
    if not policy_decision.allowed:
        raise HTTPException(status_code=400, detail=policy_decision.reason)

    tracking_info = order_service.get_tracking_info(order_id)
    if not tracking_info:
        raise HTTPException(status_code=404, detail="Tracking information not found")

    return tracking_info


@app.get("/orders")
async def list_orders(email: str | None = None):
    """List orders (optionally filtered by email)."""
    if email:
        orders = order_service.get_orders_by_email(email)
    else:
        orders = order_service.list_all_orders()

    return {"orders": orders, "count": len(orders)}


@app.get("/policies/{policy_type}")
async def get_policy_info(policy_type: str):
    """Get policy information."""
    policy_explanation = policy_engine.get_policy_explanation(policy_type)
    if "not available" in policy_explanation:
        raise HTTPException(status_code=404, detail="Policy type not found")

    return {"policy_type": policy_type, "explanation": policy_explanation}


@app.get("/health")
async def health_check():
    """Health check endpoint for container monitoring."""
    try:
        orders_count = len(order_service.list_all_orders())

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "policy-aware-order-chatbot",
            "version": "1.0.0",
            "checks": {
                "order_service": "ok",
                "orders_loaded": orders_count,
                "policy_engine": "ok",
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            },
        ) from e


def main():
    """Run the application."""
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")  # noqa: S104


if __name__ == "__main__":
    main()
