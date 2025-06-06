"""
Evaluation Framework
Comprehensive testing and metrics for chatbot performance assessment.
"""

import asyncio
import json
import logging  # Add logging import
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score

from chatbot.agents import PolicyAwareChatbot
from chatbot.policy_engine import PolicyEngine

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)  # Initialize logger


class ResponseQualityScore(BaseModel):
    """Structured output for LLM-based response quality evaluation."""

    score: float = Field(
        ...,
        description="Numerical score from 1 (poor) to 5 (excellent) for the response quality.",
    )
    justification: str = Field(
        ..., description="Brief justification for the assigned score."
    )
    critique: str | None = Field(
        default=None, description="Specific aspects the chatbot could improve."
    )


@dataclass
class TestCase:
    """Individual test case for chatbot evaluation."""

    id: str
    user_message: str
    expected_intent: str
    expected_entities: dict[str, str]
    expected_policy_outcome: str  # "allowed", "denied", "requires_approval"
    context: dict[str, Any]
    description: str


@dataclass
class EvaluationResult:
    """Results from a single test case evaluation."""

    test_case_id: str
    predicted_intent: str
    extracted_entities: dict[str, Any]
    policy_compliance: bool
    response_quality_score: float
    execution_time: float
    error: str = None


class ChatbotEvaluator:
    """Comprehensive evaluation system for the chatbot."""

    def __init__(self):
        self.chatbot = PolicyAwareChatbot()
        self.policy_engine = PolicyEngine()
        self.test_cases = self._create_test_cases()

        # LLM for response quality evaluation (I keep OPENAI_API_KEY in .env)
        self.quality_llm = ChatOpenAI(
            model="gpt-4.1-mini-2025-04-14",
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.quality_parser = PydanticOutputParser(pydantic_object=ResponseQualityScore)

    def _create_test_cases(self) -> list[TestCase]:
        """Create comprehensive test cases covering various scenarios."""
        test_cases = [
            # Order Cancellation - Allowed Cases
            TestCase(
                id="cancel_01",
                user_message="I want to cancel my order ORD-2025-1000",
                expected_intent="order_cancellation",
                expected_entities={"order_id": "ORD-2025-1000"},
                expected_policy_outcome="allowed",
                context={"order_exists": True, "order_age_days": 3},
                description="Recent order cancellation request",
            ),
            TestCase(
                id="cancel_02",
                user_message="Can you please help me cancel order ORD-2025-1001? I changed my mind.",
                expected_intent="order_cancellation",
                expected_entities={"order_id": "ORD-2025-1001"},
                expected_policy_outcome="allowed",
                context={"order_exists": True, "order_age_days": 5},
                description="Polite cancellation request with reason",
            ),
            # Order Cancellation - Denied Cases
            TestCase(
                id="cancel_03",
                user_message="I need to cancel ORD-2025-1005, it's been too long",
                expected_intent="order_cancellation",
                expected_entities={"order_id": "ORD-2025-1005"},
                expected_policy_outcome="denied",
                context={"order_exists": True, "order_age_days": 15},
                description="Order too old for cancellation",
            ),
            # Order Cancellation - Requires Approval
            TestCase(
                id="cancel_04",
                user_message="Cancel my order ORD-2025-1006 please, even though it shipped",
                expected_intent="order_cancellation",
                expected_entities={"order_id": "ORD-2025-1006"},
                expected_policy_outcome="requires_approval",
                context={"order_exists": True, "status": "shipped"},
                description="Shipped order cancellation requiring approval",
            ),
            # Order Tracking Cases
            TestCase(
                id="track_01",
                user_message="Where is my order ORD-2025-1000?",
                expected_intent="order_tracking",
                expected_entities={"order_id": "ORD-2025-1000"},
                expected_policy_outcome="allowed",
                context={"order_exists": True},
                description="Simple order tracking request",
            ),
            TestCase(
                id="track_02",
                user_message="Can you track order ORD-2025-1002 for me?",
                expected_intent="order_tracking",
                expected_entities={"order_id": "ORD-2025-1002"},
                expected_policy_outcome="allowed",
                context={"order_exists": True},
                description="Polite tracking request",
            ),
            TestCase(
                id="track_03",
                user_message="What's the status of ORD-2025-9999?",
                expected_intent="order_tracking",
                expected_entities={"order_id": "ORD-2025-9999"},
                expected_policy_outcome="denied",
                context={"order_exists": False},
                description="Tracking request for non-existent order",
            ),
            # General Inquiries
            TestCase(
                id="general_01",
                user_message="What is your cancellation policy?",
                expected_intent="general_inquiry",
                expected_entities={},
                expected_policy_outcome="allowed",
                context={},
                description="Policy information request",
            ),
            TestCase(
                id="general_02",
                user_message="Hello, I need help with my order",
                expected_intent="general_inquiry",
                expected_entities={},
                expected_policy_outcome="allowed",
                context={},
                description="General help request",
            ),
            # Edge Cases
            TestCase(
                id="edge_01",
                user_message="Cancel order ABC123",
                expected_intent="order_cancellation",
                expected_entities={"order_id": "ABC123"},
                expected_policy_outcome="denied",
                context={"order_exists": False},
                description="Cancellation request with invalid order ID",
            ),
            TestCase(
                id="edge_02",
                user_message="I'm very angry about my order! This is terrible service!",
                expected_intent="complaint",
                expected_entities={},
                expected_policy_outcome="allowed",
                context={},
                description="Complaint without specific order information",
            ),
        ]

        return test_cases

    async def evaluate_single_case(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case."""
        start_time = datetime.now()

        try:
            # get chatbot response
            result = self.chatbot.chat(test_case.user_message)

            execution_time = (datetime.now() - start_time).total_seconds()

            # evaluate intent classification
            predicted_intent = result.get("intent", "unknown")

            # evaluate entity extraction
            extracted_entities = result.get("extracted_entities", {})
            self._calculate_entity_accuracy(
                test_case.expected_entities, extracted_entities
            )  # Call the method without assigning to unused variable

            # evaluate policy compliance
            policy_compliance = self._evaluate_policy_compliance(test_case, result)

            # evaluate response quality
            response_quality = await self._evaluate_response_quality(
                test_case.user_message, result["response"], test_case.context
            )

            return EvaluationResult(
                test_case_id=test_case.id,
                predicted_intent=predicted_intent,
                extracted_entities=extracted_entities,
                policy_compliance=policy_compliance,
                response_quality_score=response_quality,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return EvaluationResult(
                test_case_id=test_case.id,
                predicted_intent="error",
                extracted_entities={},
                policy_compliance=False,
                response_quality_score=0.0,
                execution_time=execution_time,
                error=str(e),
            )

    def _calculate_entity_accuracy(
        self, expected: dict[str, str], predicted: dict[str, Any]
    ) -> float:
        """Calculate entity extraction accuracy."""
        if not expected:
            return 1.0 if not predicted else 0.5

        correct = 0
        total = len(expected)

        for key, expected_value in expected.items():
            if key in predicted and str(predicted[key]) == expected_value:
                correct += 1

        return correct / total if total > 0 else 1.0

    def _evaluate_policy_compliance(
        self, test_case: TestCase, result: dict[str, Any]
    ) -> bool:
        """Evaluate if the chatbot correctly followed policies."""
        policy_decision_data = result.get("policy_decision")
        requires_handoff = result.get("requires_human_handoff", False)
        response_text = result.get("response", "").lower()

        if policy_decision_data:
            if isinstance(policy_decision_data, dict):
                actual_allowed = policy_decision_data.get("allowed", False)
                actual_requires_approval = policy_decision_data.get(
                    "requires_approval", False
                )
            else:
                actual_allowed = policy_decision_data.allowed
                actual_requires_approval = policy_decision_data.requires_approval

            if test_case.expected_policy_outcome == "allowed":
                return actual_allowed and not actual_requires_approval
            elif test_case.expected_policy_outcome == "denied":
                return not actual_allowed
            elif test_case.expected_policy_outcome == "requires_approval":
                return actual_allowed and actual_requires_approval and requires_handoff
            elif test_case.expected_policy_outcome == "n/a":
                return True
            else:
                return False
        else:
            if test_case.expected_policy_outcome == "allowed":
                return (
                    (
                        "cancelled" in response_text
                        or "tracking information" in response_text
                    )
                    and "approval" not in response_text
                    and "sorry" not in response_text
                    and "unable" not in response_text
                )
            elif test_case.expected_policy_outcome == "denied":
                return (
                    "sorry" in response_text
                    or "unable" in response_text
                    or "cannot" in response_text
                    or "not found" in response_text
                )
            elif test_case.expected_policy_outcome == "requires_approval":
                return "approval" in response_text and requires_handoff
            elif test_case.expected_policy_outcome == "n/a":
                return True

        return False

    async def _evaluate_response_quality(
        self,
        user_message: str,
        chatbot_response: str,
        test_case_context: dict[str, Any],
    ) -> float:
        """Evaluate response quality using an LLM-as-a-judge approach."""

        prompt_template = ChatPromptTemplate.from_template("""
        You are an impartial AI quality assurance expert evaluating a customer service chatbot.
        Assess the chatbot's response based on the user's message and the context of the test case.

        User's Message: "{user_message}"
        Chatbot's Response: "{chatbot_response}"
        Test Case Context: {test_case_context}
        (The context might include things like whether an order exists, its age, or its status, which are crucial for judging the response's appropriateness if the chatbot was supposed to use this information).

        Evaluation Criteria:
        1.  Helpfulness & Relevance: Does the response directly address the user's query or problem? Is it on-topic?
        2.  Correctness & Accuracy: Is the information provided accurate given the context? Does it correctly reflect policies (if applicable and knowable from context)?
        3.  Clarity & Conciseness: Is the response easy to understand, free of jargon, and to the point?
        4.  Tone & Empathy: Is the tone professional, polite, and appropriately empathetic?
        5.  Actionability: If the user needs to do something, are the instructions clear? If the chatbot is performing an action, is this communicated clearly?

        Considering all these criteria, provide a holistic quality score from 1 to 5, where:
        1 = Poor: Largely unhelpful, incorrect, unclear, or inappropriate tone.
        2 = Fair: Some issues in one or more areas; partially helpful but needs significant improvement.
        3 = Good: Generally helpful and correct, but with minor issues in clarity, tone, or completeness.
        4 = Very Good: Helpful, correct, clear, and good tone; minor suggestions for improvement might exist.
        5 = Excellent: Outstanding response that meets all criteria effectively and professionally.

        {format_instructions}
        """)

        chain = prompt_template | self.quality_llm | self.quality_parser

        try:
            evaluation = await chain.ainvoke(
                {
                    "user_message": user_message,
                    "chatbot_response": chatbot_response,
                    "test_case_context": json.dumps(
                        test_case_context
                    ),  # Ensure context is a string for the prompt
                    "format_instructions": self.quality_parser.get_format_instructions(),
                }
            )
            # Log the justification and critique for manual review if needed
            logger.info(
                f"Response Quality Eval: Score={evaluation.score}, Justification='{evaluation.justification}', Critique='{evaluation.critique}'"
            )
            return evaluation.score
        except Exception as e:
            logger.error(f"Error during response quality evaluation: {str(e)}")
            return 0.0  # Return a default low score in case of error

    async def run_evaluation(self) -> dict[str, Any]:
        """Run complete evaluation suite."""
        print("Starting chatbot evaluation...")

        results = []
        for i, test_case in enumerate(self.test_cases):
            print(
                f"Evaluating test case {i + 1}/{len(self.test_cases)}: {test_case.id}"
            )
            result = await self.evaluate_single_case(test_case)
            results.append(result)

        # calculate metrics
        metrics = self._calculate_metrics(results)

        # generate report
        report = self._generate_report(results, metrics)

        return {"results": results, "metrics": metrics, "report": report}

    def _calculate_metrics(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Calculate aggregate metrics from evaluation results."""
        total_cases = len(results)
        successful_cases = [r for r in results if r.error is None]

        # intent classification accuracy
        test_intents = [tc.expected_intent for tc in self.test_cases]
        predicted_intents = [r.predicted_intent for r in results]
        intent_accuracy = accuracy_score(test_intents, predicted_intents)

        # policy compliance rate
        policy_compliance_rate = (
            sum(r.policy_compliance for r in successful_cases) / len(successful_cases)
            if successful_cases
            else 0
        )

        # average response quality
        avg_response_quality = (
            np.mean([r.response_quality_score for r in successful_cases])
            if successful_cases
            else 0
        )

        # average execution time
        avg_execution_time = np.mean([r.execution_time for r in results])

        # error rate
        error_rate = len([r for r in results if r.error is not None]) / total_cases

        return {
            "total_test_cases": total_cases,
            "successful_cases": len(successful_cases),
            "intent_accuracy": intent_accuracy,
            "policy_compliance_rate": policy_compliance_rate,
            "average_response_quality": avg_response_quality,
            "average_execution_time": avg_execution_time,
            "error_rate": error_rate,
        }

    def _generate_report(
        self, results: list[EvaluationResult], metrics: dict[str, Any]
    ) -> str:
        """Generate human-readable evaluation report."""
        report = f"""
# Chatbot Evaluation Report
Generated: {datetime.now().isoformat()}

## Summary Metrics
- Total Test Cases: {metrics["total_test_cases"]}
- Successful Cases: {metrics["successful_cases"]}
- Intent Classification Accuracy: {metrics["intent_accuracy"]:.2%}
- Policy Compliance Rate: {metrics["policy_compliance_rate"]:.2%}
- Average Response Quality: {metrics["average_response_quality"]:.2f}/1.0
- Average Execution Time: {metrics["average_execution_time"]:.3f}s
- Error Rate: {metrics["error_rate"]:.2%}

## Key Insights

### Strengths
"""

        if metrics["intent_accuracy"] > 0.8:
            report += "- Strong intent classification performance\n"
        if metrics["policy_compliance_rate"] > 0.8:
            report += "- Excellent policy adherence\n"
        if metrics["average_response_quality"] > 0.6:
            report += "- High-quality conversational responses\n"
        if metrics["average_execution_time"] < 2.0:
            report += "- Fast response times\n"

        report += "\n### Areas for Improvement\n"

        if metrics["intent_accuracy"] < 0.7:
            report += "- Intent classification needs improvement\n"
        if metrics["policy_compliance_rate"] < 0.7:
            report += "- Policy enforcement could be more consistent\n"
        if metrics["average_response_quality"] < 0.5:
            report += "- Response quality could be enhanced\n"
        if metrics["error_rate"] > 0.1:
            report += "- Error handling needs attention\n"

        return report

    def save_results(
        self, evaluation_data: dict[str, Any], output_dir: str = "evaluation_results"
    ):
        """Save evaluation results to files."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # save detailed results as JSON
        results_file = os.path.join(
            output_dir,
            f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(results_file, "w") as f:
            # convert results to serializable format
            serializable_data = {
                "metrics": evaluation_data["metrics"],
                "report": evaluation_data["report"],
                "results": [
                    {
                        "test_case_id": r.test_case_id,
                        "predicted_intent": r.predicted_intent,
                        "extracted_entities": r.extracted_entities,
                        "policy_compliance": r.policy_compliance,
                        "response_quality_score": r.response_quality_score,
                        "execution_time": r.execution_time,
                        "error": r.error,
                    }
                    for r in evaluation_data["results"]
                ],
            }
            json.dump(serializable_data, f, indent=2)

        report_file = os.path.join(
            output_dir,
            f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        )
        with open(report_file, "w") as f:
            f.write(evaluation_data["report"])

        print(f"Evaluation results saved to {output_dir}")


async def main():
    """Run the evaluation."""
    evaluator = ChatbotEvaluator()
    evaluation_data = await evaluator.run_evaluation()

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print(evaluation_data["report"])

    # save results
    evaluator.save_results(evaluation_data)

    return evaluation_data


if __name__ == "__main__":
    asyncio.run(main())
