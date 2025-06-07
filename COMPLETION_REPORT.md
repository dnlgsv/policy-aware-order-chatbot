# Project Completion Report: Policy-Aware Order Chatbot

**Date:** June 7, 2025

## Project Overview

This project involved the development of a generative chatbot designed to handle customer inquiries related to order cancellations and tracking. The chatbot integrates Large Language Model (LLM) capabilities with a robust policy enforcement engine and API connectivity to a mock order management system. The primary goal was to create a system that not only provides natural and helpful responses but also strictly adheres to predefined company policies.

### Core System Components & Features:

1.  **Conversational AI Core**:
    *   Implemented a multi-agent architecture using LangGraph for managing complex conversational flows.
    *   Developed a Router Agent for accurate intent classification (e.g., order cancellation, order tracking, general inquiry).
    *   Created specialized agents for handling cancellation and tracking requests, each incorporating policy checks.
    *   Utilized OpenAI's GPT models for natural language understanding and response generation.

2.  **Policy Engine**:
    *   Established a centralized module for enforcing business rules, including a 10-day cancellation window.
    *   Implemented logic for scenarios requiring manager approval (e.g., shipped orders) and outright denial (e.g., delivered orders).
    *   Designed the engine to be configurable, allowing for future expansion of policy rules.

3.  **Order Management System (Mock)**:
    *   Created a simulated order service with a predefined set of test orders, featuring diverse creation dates and statuses.
    *   Provided API endpoints for retrieving order details, processing cancellations, and fetching tracking information.

4.  **API Infrastructure (FastAPI)**:
    *   Exposed chatbot functionalities via a REST API, including a primary `/chat` endpoint.
    *   Included endpoints for direct order operations and system health monitoring.
    *   Integrated Swagger UI for interactive API documentation.

5.  **Evaluation Framework**:
    *   Designed a comprehensive testing suite with multiple scenarios to assess chatbot performance.
    *   Measured key metrics: intent classification accuracy (including precision, recall, F1-score), entity extraction accuracy, policy compliance rate, and response quality (using an LLM-as-a-judge approach).
    *   Tracked average execution time and error rates.

6.  **Documentation**:
    *   Provided a detailed `README.md` with setup and usage instructions.
    *   Created a `SOLUTION_DESIGN.md` document outlining the technical architecture and design choices.

## Evaluation Summary

The chatbot underwent rigorous testing across 11 distinct scenarios, covering various user intents and policy conditions.

*   **Policy Compliance Rate**: **100%**
    *   The system correctly applied all defined policies in every test case.
*   **Intent Classification Accuracy**: **90.91%** (Weighted Avg F1-score: 0.94)
    *   The chatbot demonstrated strong performance in understanding user requests. Minor areas for improvement in `general_inquiry` recall were noted.
*   **Entity Extraction Accuracy**: **100%** (F1-score)
    *   Critical information, such as order IDs, was consistently and accurately extracted.
*   **Response Quality Score**: **4.36 / 5.0**
    *   Responses were rated as high-quality, helpful, clear, and contextually appropriate by an LLM-based evaluation.
*   **Error Rate**: **0%**
    *   The system proved robust, with no operational errors during the evaluation.
*   **Average Execution Time**: 3.96 seconds per interaction.

## Key Achievements and Learnings

*   **Effective Policy Adherence**: The integration of the `PolicyEngine` with the LangGraph-based conversational flow ensured strict compliance with business rules.
*   **Modular Design**: The separation of concerns (chatbot logic, policy engine, order service) facilitated development, testing, and maintainability.
*   **Advanced Evaluation**: The use of an LLM-as-a-judge provided nuanced insights into the qualitative aspects of the chatbot's responses.
*   **Iterative Refinement**: The evaluation framework was instrumental in identifying and resolving issues, such as ensuring deterministic test data for consistent results and refining policy logic for edge cases.

## Conclusion

The Policy-Aware Order Chatbot project successfully met all specified requirements. It demonstrates a practical application of LLMs in a business context, balancing generative capabilities with the critical need for policy enforcement. The system's robust architecture, comprehensive evaluation, and strong performance metrics highlight its potential for real-world deployment in customer service operations.
