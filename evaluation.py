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
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

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
    entity_metrics: dict[str, float]
    policy_compliance: bool
    response_quality_score: float
    execution_time: float
    error: str | None = None
    policy_decision_details: dict[str, Any] | None = None
    actual_response: str | None = None


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
        raw_chatbot_result = {}
        self.chatbot = PolicyAwareChatbot()  # Re-initialize chatbot for isolation

        try:
            # get chatbot response
            raw_chatbot_result = self.chatbot.chat(
                test_case.user_message, session_id=f"eval-{test_case.id}"
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # evaluate intent classification
            predicted_intent = raw_chatbot_result.get("intent", "unknown")

            # evaluate entity extraction
            extracted_entities = raw_chatbot_result.get("extracted_entities", {})
            # store the detailed entity metrics
            entity_metrics = self._calculate_entity_metrics(
                test_case.expected_entities, extracted_entities
            )

            # evaluate policy compliance
            policy_compliance = self._evaluate_policy_compliance(
                test_case, raw_chatbot_result
            )

            # evaluate response quality
            chatbot_response_text = raw_chatbot_result.get("response", "")
            response_quality = await self._evaluate_response_quality(
                test_case.user_message, chatbot_response_text, test_case.context
            )

            # get policy decision details for logging
            policy_decision_output = raw_chatbot_result.get("policy_decision")
            if hasattr(policy_decision_output, "model_dump"):
                policy_details_for_log = policy_decision_output.model_dump()
            elif isinstance(policy_decision_output, dict):
                policy_details_for_log = policy_decision_output
            else:
                policy_details_for_log = (
                    str(policy_decision_output)
                    if policy_decision_output is not None
                    else None
                )

            return EvaluationResult(
                test_case_id=test_case.id,
                predicted_intent=predicted_intent,
                extracted_entities=extracted_entities,
                entity_metrics=entity_metrics,
                policy_compliance=policy_compliance,
                response_quality_score=response_quality,
                execution_time=execution_time,
                policy_decision_details=policy_details_for_log,  # log details
                actual_response=chatbot_response_text,  # log actual response
            )

        except Exception as e:
            logger.error(
                f"Error evaluating test case {test_case.id}: {e}", exc_info=True
            )  # Added exc_info for better debugging
            execution_time = (datetime.now() - start_time).total_seconds()

            # get response text even in case of partial failure before exception
            chatbot_response_text_on_error = (
                raw_chatbot_result.get("response", "") if raw_chatbot_result else ""
            )
            policy_decision_on_error = (
                raw_chatbot_result.get("policy_decision")
                if raw_chatbot_result
                else None
            )
            if hasattr(policy_decision_on_error, "model_dump"):
                policy_details_on_error = policy_decision_on_error.model_dump()
            elif isinstance(policy_decision_on_error, dict):
                policy_details_on_error = policy_decision_on_error
            else:
                policy_details_on_error = (
                    str(policy_decision_on_error)
                    if policy_decision_on_error is not None
                    else None
                )

            return EvaluationResult(
                test_case_id=test_case.id,
                predicted_intent="error",
                extracted_entities=raw_chatbot_result.get("extracted_entities", {})
                if raw_chatbot_result
                else {},
                entity_metrics={
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "correct": 0,
                    "total_expected": 0,
                    "total_predicted": 0,
                },
                policy_compliance=False,
                response_quality_score=0.0,
                execution_time=execution_time,
                error=str(e),
                policy_decision_details=policy_details_on_error,  # log details even on error if available
                actual_response=chatbot_response_text_on_error,  # log response even on error if available
            )

    def _calculate_entity_metrics(
        self, expected: dict[str, str], predicted: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate precision, recall, and F1 for entity extraction."""
        if not expected and not predicted:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "correct": 0,
                "total_expected": 0,
                "total_predicted": 0,
            }
        if not expected:  # all predicted are false positives if any
            return {
                "precision": 0.0,
                "recall": 1.0,
                "f1": 0.0,
                "correct": 0,
                "total_expected": 0,
                "total_predicted": len(predicted),
            }
        if not predicted:  # all expected are false negatives
            return {
                "precision": 1.0,
                "recall": 0.0,
                "f1": 0.0,
                "correct": 0,
                "total_expected": len(expected),
                "total_predicted": 0,
            }

        expected_set = set(expected.items())
        predicted_set = set(predicted.items())

        correct_predictions = len(expected_set.intersection(predicted_set))

        precision = correct_predictions / len(predicted_set) if predicted_set else 1.0
        recall = correct_predictions / len(expected_set) if expected_set else 1.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct": correct_predictions,
            "total_expected": len(expected_set),
            "total_predicted": len(predicted_set),
        }

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
                return not actual_allowed and not actual_requires_approval
            elif test_case.expected_policy_outcome == "requires_approval":
                return actual_allowed and actual_requires_approval and requires_handoff
            elif test_case.expected_policy_outcome == "n/a":
                return True
            else:
                logger.warning(
                    f"Unknown expected_policy_outcome '{test_case.expected_policy_outcome}' for TC {test_case.id} with policy_decision_data."
                )
                return False
        else:  # fallback logic: policy_decision_data is None
            logger.info(
                f"TC {test_case.id}: No policy_decision_data found, using fallback logic."
            )
            if test_case.expected_policy_outcome == "n/a":
                return True

            if test_case.expected_policy_outcome == "allowed":
                if test_case.expected_intent in ["general_inquiry", "complaint"]:
                    return not (
                        "sorry" in response_text
                        or "unable" in response_text
                        or "cannot" in response_text
                        or "denied" in response_text
                        or "not allowed" in response_text
                        or "approval" in response_text
                    )
                else:
                    action_success_indicated = False
                    if test_case.expected_intent == "order_cancellation":
                        action_success_indicated = (
                            "cancelled" in response_text
                            or "has been cancelled" in response_text
                        )
                    elif test_case.expected_intent == "order_tracking":
                        action_success_indicated = (
                            "tracking information" in response_text
                            or "status is" in response_text
                            or "shipping details" in response_text
                            or "here is the information" in response_text
                        )

                    return (
                        action_success_indicated
                        and "approval" not in response_text
                        and not (
                            "sorry" in response_text
                            or "unable" in response_text
                            or "cannot" in response_text
                            or "denied" in response_text
                        )
                    )

            elif test_case.expected_policy_outcome == "denied":
                denial_indicated = (
                    "sorry" in response_text
                    or "unable" in response_text
                    or "cannot" in response_text
                    or "not found" in response_text  # for non-existent orders
                    or "denied" in response_text
                    or "not allowed" in response_text
                    or "policy prevents" in response_text
                )
                return denial_indicated and "approval" not in response_text

            elif test_case.expected_policy_outcome == "requires_approval":
                approval_text_present = (
                    "approval" in response_text
                    or "needs approval" in response_text
                    or "manager approval" in response_text
                    or "requires authorization" in response_text
                )
                return approval_text_present and requires_handoff
            else:
                logger.warning(
                    f"Unknown expected_policy_outcome '{test_case.expected_policy_outcome}' for TC {test_case.id} without policy_decision_data."
                )
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

        # intent classification etrics
        test_intents = [
            tc.expected_intent
            for tc in self.test_cases
            if tc.id in [r.test_case_id for r in successful_cases]
        ]
        predicted_intents = [r.predicted_intent for r in successful_cases]

        intent_accuracy = (
            accuracy_score(test_intents, predicted_intents) if successful_cases else 0
        )

        # using zero_division=0 to handle cases where a class might not be predicted or not present in true labels for a small test set
        intent_precision, intent_recall, intent_f1, _ = precision_recall_fscore_support(
            test_intents, predicted_intents, average="weighted", zero_division=0
        )

        intent_classification_report_str = (
            classification_report(test_intents, predicted_intents, zero_division=0)
            if successful_cases
            else "No successful cases to report for intent classification."
        )

        # entity extraction metrics
        all_entity_metrics = [
            r.entity_metrics for r in successful_cases if hasattr(r, "entity_metrics")
        ]
        if all_entity_metrics:
            avg_entity_precision = np.mean([m["precision"] for m in all_entity_metrics])
            avg_entity_recall = np.mean([m["recall"] for m in all_entity_metrics])
            avg_entity_f1 = np.mean([m["f1"] for m in all_entity_metrics])
            total_correct_entities = sum(m["correct"] for m in all_entity_metrics)
            total_expected_entities = sum(
                m["total_expected"] for m in all_entity_metrics
            )
            total_predicted_entities = sum(
                m["total_predicted"] for m in all_entity_metrics
            )
            overall_entity_precision = (
                total_correct_entities / total_predicted_entities
                if total_predicted_entities > 0
                else 1.0
            )
            overall_entity_recall = (
                total_correct_entities / total_expected_entities
                if total_expected_entities > 0
                else 1.0
            )
            overall_entity_f1 = (
                2
                * (overall_entity_precision * overall_entity_recall)
                / (overall_entity_precision + overall_entity_recall)
                if (overall_entity_precision + overall_entity_recall) > 0
                else 0.0
            )
        else:
            avg_entity_precision = 0
            avg_entity_recall = 0
            avg_entity_f1 = 0
            overall_entity_precision = 0
            overall_entity_recall = 0
            overall_entity_f1 = 0

        entity_metrics_summary = {
            "average_precision": avg_entity_precision,
            "average_recall": avg_entity_recall,
            "average_f1": avg_entity_f1,
            "overall_precision": overall_entity_precision,
            "overall_recall": overall_entity_recall,
            "overall_f1": overall_entity_f1,
        }

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
            "intent_precision_weighted": intent_precision,
            "intent_recall_weighted": intent_recall,
            "intent_f1_weighted": intent_f1,
            "intent_classification_report": intent_classification_report_str,
            "entity_extraction_metrics": entity_metrics_summary,
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
- Intent Precision (Weighted Avg): {metrics["intent_precision_weighted"]:.2f}
- Intent Recall (Weighted Avg): {metrics["intent_recall_weighted"]:.2f}
- Intent F1-score (Weighted Avg): {metrics["intent_f1_weighted"]:.2f}
- Entity Extraction F1 (Overall): {metrics["entity_extraction_metrics"]["overall_f1"]:.2f}
  - Entity Precision (Overall): {metrics["entity_extraction_metrics"]["overall_precision"]:.2f}
  - Entity Recall (Overall): {metrics["entity_extraction_metrics"]["overall_recall"]:.2f}
- Policy Compliance Rate: {metrics["policy_compliance_rate"]:.2%}
- Average Response Quality: {metrics["average_response_quality"]:.2f}/5.0
- Average Execution Time: {metrics["average_execution_time"]:.3f}s
- Error Rate: {metrics["error_rate"]:.2%}

## Intent Classification Detailed Report
```
{metrics["intent_classification_report"]}
```

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
                        "entity_metrics": r.entity_metrics,
                        "policy_compliance": r.policy_compliance,
                        "response_quality_score": r.response_quality_score,
                        "execution_time": r.execution_time,
                        "error": r.error,
                        "policy_decision_details": r.policy_decision_details,
                        "actual_response": r.actual_response,
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
