# Solution Design: Policy-Aware Order Chatbot

## 1. Overview

This document outlines the solution design for the Policy-Aware Order Chatbot. The chatbot is designed to handle customer inquiries regarding order cancellations and tracking while strictly adhering to predefined company policies. It leverages Large Language Models (LLMs) through the LangChain and LangGraph frameworks for conversational abilities and complex task execution. The system is exposed via a FastAPI-based API and includes a mock order management backend.

## 2. Core Requirements Addressed

-   **Generative Chatbot**: Utilizes LLMs to understand user intent, extract information, and generate natural, context-aware responses.
-   **API Connectivity**: Integrates with an `OrderService` (mocked) to fetch order details, process cancellations, and retrieve tracking information.
-   **Policy Adherence**: Implements a `PolicyEngine` to evaluate customer requests against company policies (e.g., 10-day cancellation window).
-   **Specific Use Cases**: Handles:
    -   Order Cancellation: Evaluates eligibility and processes cancellations.
    -   Order Tracking: Provides status and tracking details.
-   **Experimentation & Evaluation**: A dedicated framework (`evaluation.py`) is designed to assess chatbot performance systematically.

## 3. System Architecture

The system comprises several key components:

**3.1. FastAPI Application (`main.py`)**

-   Serves as the main entry point for user interactions.
-   Exposes a `/chat` endpoint to receive user messages and return chatbot responses.
-   Provides direct API endpoints for order operations (`/orders/{order_id}`, `/orders/{order_id}/cancel`, etc.) which internally use the `OrderService` and `PolicyEngine`.
-   Handles request/response validation using Pydantic models.

**3.2. Chatbot Core (`chatbot/` directory)**

**3.2.1. `PolicyAwareChatbot` (`chatbot/agents.py`)**
-   The central orchestrator, built using **LangGraph**.
-   Manages the conversational flow as a state machine.
-   Integrates various specialized agents (nodes in the graph) to handle different aspects of the conversation.

**3.2.2. LangGraph State (`ChatState` in `chatbot/agents.py`)**
-   A Pydantic model representing the state of the conversation at any point. It includes:
    -   `messages`: History of the conversation.
    -   `current_intent`: The user's classified intent.
    -   `order_id`, `customer_email`: Extracted entities.
    -   `policy_decision`: Outcome of a policy check.
    -   `order_data`: Fetched order details.
    -   `requires_human_handoff`: Flag for escalation.

**3.2.3. LangGraph Nodes & Edges (Conceptualized in `PolicyAwareChatbot._build_graph()`):**

*   **Node 1: Intent Classification & Entity Extraction (`RouterAgent.classify_intent`)**
    -   Input: User message, conversation history.
    -   Action: Employs an LLM with a `ChatPromptTemplate` and `PydanticOutputParser` (`IntentClassification`) to determine user intent (e.g., `order_cancellation`, `order_tracking`, `general_inquiry`) and extract entities like `order_id`.
    -   Output: Updates `ChatState` with `current_intent` and `extracted_entities`.

*   **Conditional Edge: Route based on Intent**
    -   Directs the flow to the appropriate specialized agent/node based on `current_intent`.

*   **Node 2a: Order Cancellation Handling (`CancellationAgent`)**
    -   Triggered for `order_cancellation` intent.
    -   Action: Verifies presence of `order_id` (prompts if missing). Fetches order details via `order_service.get_order()`. Evaluates cancellation eligibility using `policy_engine.evaluate_cancellation()`. If policy permits, processes cancellation via `order_service.cancel_order()`. Generates an LLM-based response reflecting the outcome.
    -   Output: Updates `ChatState` with `response`, `policy_decision`, `order_data`, and `requires_human_handoff` status.

*   **Node 2b: Order Tracking Handling (`TrackingAgent`)**
    -   Triggered for `order_tracking` intent.
    -   Action: Verifies presence of `order_id` (prompts if missing). Evaluates tracking eligibility using `policy_engine.evaluate_tracking()`. If permitted, fetches tracking details via `order_service.get_tracking_info()`. Generates an LLM-based response with tracking information or an explanation if unavailable.
    -   Output: Updates `ChatState` with `response` and `policy_decision`.

*   **Node 2c: General Inquiry Handling**
    -   Triggered for `general_inquiry` or `unknown` intents.
    -   Action: Leverages an LLM to provide a helpful response. May consult `policy_engine.get_policy_explanation()` for policy-related questions.
    -   Output: Updates `ChatState` with `response`.

*   **Node 3: Final Response Formatting**
    -   Ensures the `ChatState.messages` (conversation history) is updated with the latest user message and AI response.
    -   The final `response` and other relevant fields from `ChatState` are returned to the FastAPI layer.

**3.2.4. `PolicyEngine` (`chatbot/policy_engine.py`)**
-   Encapsulates all business rules.
-   `OrderStatus` (Enum) and `PolicyDecision` (Pydantic model) define clear states and outcomes.
-   `evaluate_cancellation()`: Checks order date (10-day window), status (shipped orders require approval, delivered cannot be cancelled).
-   `evaluate_tracking()`: Checks validity of order ID for tracking.
-   `get_policy_explanation()`: Provides text for policies.

**3.2.5. `OrderService` (`chatbot/order_service.py`)**
-   A mock service simulating an external order management system.
-   `Order` and `TrackingInfo` (Pydantic models) define data structures.
-   Generates sample orders with varying dates and statuses to facilitate testing of policy adherence.
-   Provides methods: `get_order()`, `cancel_order()`, `get_tracking_info()`, `list_all_orders()`.

**3.3. LLM Integration (LangChain)**

-   `ChatOpenAI` is used as the LLM provider (configurable, e.g., `gpt-4.1-mini-2025-04-14`).
-   `ChatPromptTemplate` is used for structuring inputs to the LLM for consistent and reliable outputs.
-   `PydanticOutputParser` is used to parse LLM outputs into structured Pydantic objects (e.g., `IntentClassification`, `ResponseQualityScore` in evaluation).

## 4. Data Models (Pydantic)

Key Pydantic models are used throughout the application for data validation, structured state management, and clear API contracts:

-   `ChatState`: For LangGraph state.
-   `IntentClassification`: For structured intent output.
-   `PolicyDecision`: For policy evaluation results.
-   `Order`, `TrackingInfo`: For order data.
-   `ChatRequest`, `ChatResponse`: For API endpoint I/O.
-   `ResponseQualityScore`: For LLM-based evaluation output.

## 5. Experimentation and Evaluation (`evaluation.py`)

-   **Objective**: To assess chatbot performance in decision-making, policy adherence, and response generation.
-   **Methodology**:
    -   A suite of predefined `TestCase` (dataclass) instances covering various scenarios (cancellations allowed/denied/approval, tracking success/failure, general queries, edge cases).
    -   `ChatbotEvaluator` class orchestrates the evaluation.
    -   Each test case is run through `chatbot.chat()`.
-   **Metrics**:
    -   **Intent Classification Accuracy**: Compares predicted vs. expected intent. *Rationale: Fundamental for understanding the user's goal and initiating the correct workflow. Metrics like precision, recall, and F1-score per intent class provide deeper insights into classification performance, especially for imbalanced datasets or critical misclassifications.*
    -   **Entity Extraction Accuracy**: (`_calculate_entity_accuracy`) Compares extracted vs. expected entities. *Rationale: Critical for obtaining data (e.g., `order_id`) required for API interactions and policy evaluations. Precision, recall, and F1-score per entity type offer a detailed view of extraction quality.*
    -   **Policy Compliance Rate**: (`_evaluate_policy_compliance`) Assesses if chatbot actions (e.g., cancellation, approval request, denial) align with policy-defined outcomes. *Rationale: Directly measures adherence to core business rules, a primary project requirement.*
    -   **Response Quality Score**: (`_evaluate_response_quality`) Employs an LLM-as-a-judge to score responses (1-5) on helpfulness, correctness, clarity, tone, and actionability, using a structured Pydantic model for output. *Rationale: Offers a nuanced evaluation of the conversational experience, assessing qualitative aspects crucial for user satisfaction.*
    -   **Average Execution Time**: Time taken per chat interaction. *Rationale: Important for user experience; responses should be timely to maintain engagement.*
    -   **Error Rate**: Percentage of test cases resulting in an error. *Rationale: Indicates the overall robustness and reliability of the system.*
    -   *Note on MRR/NDCG*: Metrics like Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG) are powerful for evaluating ranked lists. They are less directly applicable here as the primary evaluation focuses on the correctness of a single classified intent, extracted entities, and a single generated response. However, they could become relevant if the system were adapted to, for example, produce a ranked list of potential intents or candidate responses.
-   **Reporting**: Generates a summary report and saves detailed results to `evaluation_results/`.

## 6. Tooling

-   **`uv`**: For Python package management, ensuring reproducible environments via `pyproject.toml` and `uv.lock`.
-   **`python-dotenv`**: To manage environment variables (like `OPENAI_API_KEY`).

## 7. Design Rationale for Key Choices

-   **LangGraph**: Chosen for its ability to build robust, stateful, and complex agentic applications. It allows for clear definition of conversational flows with conditional logic, making it ideal for handling multi-turn interactions and decision-making processes.
-   **Pydantic**: Used extensively for data validation, settings, and defining clear structures for state, API contracts, and LLM outputs. This improves reliability and maintainability.
-   **LLM Choice (e.g., `gpt-4.1-mini` or similar instruction-following models)**: The `gpt-4.1-mini` model (or comparable instruction-tuned models) was selected for its balance of strong natural language understanding capabilities, speed, and cost-effectiveness. These characteristics make it suitable for tasks like intent classification, entity extraction, and response generation in a conversational AI context. Its instruction-following capabilities are also beneficial for integrating with the structured outputs required by Pydantic models when using tools like Instructor.
-   **LLM-as-a-Judge for Response Quality**: Provides a more nuanced and human-like assessment of response quality compared to simple heuristic checks, aligning with advanced ML practices for evaluating generative models.
-   **Separation of Concerns**: The `PolicyEngine`, `OrderService`, and various `Agent` classes are designed to be modular and independent, facilitating easier testing, maintenance, and future extensions.
-   **FastAPI**: Provides a modern, high-performance framework for building the API layer, with automatic data validation and documentation.

### Arbitrary Policy Definitions:
Beyond the core 10-day cancellation window specified in the challenge, additional policies were defined to reflect common e-commerce scenarios and demonstrate robust policy handling:
-   **Shipped Order Cancellations**: Orders that have already been shipped require manager approval. This simulates a more complex cancellation process for items already in transit, showcasing the chatbot's ability to handle conditional logic and escalate or inform about next steps.
-   **Delivered Order Cancellations**: Orders that have been delivered cannot be cancelled, as the transaction is typically considered complete. This adds another layer to the policy logic.
-   **Tracking Policy**: Basic validation of order ID for tracking requests ensures the system handles invalid or non-existent orders gracefully.
These policies were chosen to create a more realistic and challenging environment for the chatbot, testing its capacity to navigate multiple conditions and provide accurate, policy-compliant responses.

## 8. Future Enhancements (Considerations)

-   Integration with a real database for `OrderService`.
-   More sophisticated state management for multi-turn entity gathering (e.g., if an order ID is not provided initially).
-   Full implementation of human handoff capabilities.
-   Advanced RAG for knowledge-intensive general inquiries.

A RAG system would:
- Retrieve relevant information from a knowledge source (e.g., company - FAQs, policy documents, product catalogs).
- Augment the LLM's prompt with this retrieved information.
- Generate a more informed and accurate answer based on this augmented context.
- This would make the chatbot more versatile and capable of answering a wider range of general questions without needing to explicitly train the LLM on every piece of information or hardcode extensive Q&A pairs. It helps keep the chatbot's knowledge current by simply updating the knowledge base RAG draws from.
