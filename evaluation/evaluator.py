"""
Core AI interviewer quality evaluator.

Uses NVIDIA Nemotron to evaluate an interview transcript
against the discrete rubric defined in rubric.py.
"""

import json
import re
from typing import Any

from evaluation.models import evaluate_with_nvidia
from evaluation.rubric import ALLOWED_SCORES, RUBRIC


# ============================================================
# JSON extraction
# ============================================================

def _candidate_json_objects(
    response: str,
) -> list[dict[str, Any]]:
    """
    Scan the response for every syntactically valid JSON
    object, using the JSON parser itself (raw_decode) so
    braces inside string values or surrounding prose can
    never corrupt the extraction.
    """

    decoder = json.JSONDecoder()

    candidates: list[dict[str, Any]] = []

    index = response.find("{")

    while index != -1:

        try:
            parsed, end = decoder.raw_decode(
                response,
                index,
            )

        except json.JSONDecodeError:
            index = response.find("{", index + 1)
            continue

        if isinstance(parsed, dict):
            candidates.append(parsed)

        index = response.find("{", end)

    return candidates


def extract_json(response: str) -> dict[str, Any]:
    """
    Extract a JSON object from the model response.

    The model is instructed to return JSON only, but this
    function also handles accidental markdown code fences,
    reasoning (<think>) blocks, and prose around the JSON.
    """

    response = response.strip()

    # Remove reasoning blocks emitted by reasoning models.
    # Also handle a reasoning prefix whose opening tag was
    # not included in the returned content.
    response = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    response = re.sub(
        r"^.*?</think>",
        "",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove markdown code fences if present.
    response = re.sub(
        r"^```(?:json)?\s*",
        "",
        response,
        flags=re.IGNORECASE,
    )

    response = re.sub(
        r"\s*```$",
        "",
        response,
    )

    response = response.strip()

    try:
        parsed = json.loads(response)

    except json.JSONDecodeError as error:

        candidates = _candidate_json_objects(
            response
        )

        if not candidates:
            raise ValueError(
                "Model did not return valid JSON."
            ) from error

        # Prefer the evaluation payload; otherwise take the
        # last complete object in the response, which for a
        # reasoning model is the final answer.
        parsed = next(
            (
                candidate
                for candidate in reversed(candidates)
                if "criteria" in candidate
            ),
            candidates[-1],
        )

    if not isinstance(parsed, dict):
        raise ValueError(
            "Evaluation response must be a JSON object."
        )

    return parsed


# ============================================================
# Score validation
# ============================================================

def validate_score(
    criterion: str,
    score: Any,
) -> float:
    """
    Validate that a criterion score is one of:

        0.00
        0.25
        0.50
        0.75
        1.00
    """

    if isinstance(score, bool):
        raise ValueError(
            f"{criterion}: score must be numeric."
        )

    try:
        numeric_score = float(score)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{criterion}: invalid score {score!r}."
        ) from error

    if numeric_score not in ALLOWED_SCORES:
        raise ValueError(
            f"{criterion}: score {numeric_score} is invalid. "
            f"Allowed scores: {sorted(ALLOWED_SCORES)}"
        )

    return numeric_score


# ============================================================
# Evaluation validation
# ============================================================

def validate_evaluation(
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate the complete evaluator response.
    """

    if "criteria" not in evaluation:
        raise ValueError(
            "Evaluation is missing 'criteria'."
        )

    criteria = evaluation["criteria"]

    if not isinstance(criteria, dict):
        raise ValueError(
            "'criteria' must be a JSON object."
        )

    validated_criteria: dict[str, dict[str, Any]] = {}

    for criterion in RUBRIC:

        if criterion == "overall_interview_quality":
            continue

        if criterion not in criteria:
            raise ValueError(
                f"Missing criterion: {criterion}"
            )

        result = criteria[criterion]

        if not isinstance(result, dict):
            raise ValueError(
                f"{criterion} must be an object."
            )

        if "score" not in result:
            raise ValueError(
                f"{criterion} is missing 'score'."
            )

        if "reasoning" not in result:
            raise ValueError(
                f"{criterion} is missing 'reasoning'."
            )

        score = validate_score(
            criterion,
            result["score"],
        )

        reasoning = result["reasoning"]

        if not isinstance(reasoning, str):
            raise ValueError(
                f"{criterion}: reasoning must be a string."
            )

        if not reasoning.strip():
            raise ValueError(
                f"{criterion}: reasoning cannot be empty."
            )

        validated_criteria[criterion] = {
            "score": score,
            "reasoning": reasoning.strip(),
        }

    # --------------------------------------------------------
    # Calculate the overall score ourselves.
    #
    # Do NOT trust the LLM to calculate this.
    # --------------------------------------------------------

    scores = [
        item["score"]
        for item in validated_criteria.values()
    ]

    overall_score = sum(scores) / len(scores)

    return {
        "criteria": validated_criteria,
        "overall_score": overall_score,
    }


# ============================================================
# Structured output schema
# ============================================================

def build_response_schema() -> dict[str, Any]:
    """
    Build the response_format json_schema for the rubric.

    Enforced server-side by the NVIDIA endpoint, so the
    model cannot return malformed keys, invalid scores, or
    prose around the JSON. Python-side validation still runs
    on the result.
    """

    criterion_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "enum": sorted(ALLOWED_SCORES),
            },
            "reasoning": {
                "type": "string",
            },
        },
        "required": ["score", "reasoning"],
        "additionalProperties": False,
    }

    criteria_properties = {
        criterion: criterion_schema
        for criterion in RUBRIC
        if criterion != "overall_interview_quality"
    }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "InterviewEvaluation",
            "schema": {
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "object",
                        "properties": criteria_properties,
                        "required": list(
                            criteria_properties
                        ),
                        "additionalProperties": False,
                    },
                },
                "required": ["criteria"],
                "additionalProperties": False,
            },
        },
    }


# ============================================================
# Prompt construction
# ============================================================

def build_evaluation_prompt(
    interview_transcript: str,
) -> str:
    """
    Build the evaluation prompt sent to Nemotron.
    """

    rubric_text = []

    for criterion, definition in RUBRIC.items():

        # Overall score is calculated by Python.
        if criterion == "overall_interview_quality":
            continue

        rubric_text.append(
            f"\n### {criterion}\n"
            f"{definition['description']}\n"
        )

        for score, description in definition["scores"].items():
            rubric_text.append(
                f"- {score}: {description}\n"
            )

    rubric = "".join(rubric_text)

    return f"""
You are a strict evaluator for an AI technical interviewer.

Your task is to evaluate the QUALITY OF THE AI INTERVIEWER,
not the candidate.

Analyze the complete interview transcript below.

============================================================
SCORING RULE
============================================================

You MUST use ONLY these scores:

0.0
0.25
0.5
0.75
1.0

Never invent another score.

Score every criterion independently.

Do not give a score based on your general impression.
Use the written rubric for each criterion.

============================================================
RUBRIC
============================================================

{rubric}

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Do not use markdown.
Do not use ```json.
Do not include introductory text.
Do not include a conclusion outside the JSON.

The required structure is:

{{
  "criteria": {{
    "technical_correctness": {{
      "score": 0.0,
      "reasoning": "..."
    }},
    "relevance": {{
      "score": 0.0,
      "reasoning": "..."
    }},
    "follow_up_quality": {{
      "score": 0.0,
      "reasoning": "..."
    }},
    "question_quality": {{
      "score": 0.0,
      "reasoning": "..."
    }},
    "difficulty_calibration": {{
      "score": 0.0,
      "reasoning": "..."
    }},
    "communication_quality": {{
      "score": 0.0,
      "reasoning": "..."
    }},
    "context_awareness": {{
      "score": 0.0,
      "reasoning": "..."
    }}
  }}
}}

IMPORTANT:

- Every criterion must be present.
- Every score must be exactly 0.0, 0.25, 0.5, 0.75, or 1.0.
- Every criterion must contain reasoning.
- Reasoning must be based on evidence from the transcript.
- Do not evaluate the candidate's skill level directly.
- Evaluate how effectively the AI interviewer conducted the interview.
- Do not calculate an overall score. Python will calculate it.

============================================================
INTERVIEW TRANSCRIPT
============================================================

{interview_transcript}
"""


# ============================================================
# Main evaluation
# ============================================================

def evaluate_interview(
    interview_transcript: str,
) -> dict[str, Any]:
    """
    Evaluate one interview transcript.

    Returns validated criterion scores and a Python-calculated
    overall score.
    """

    if not interview_transcript.strip():
        raise ValueError(
            "Interview transcript cannot be empty."
        )

    prompt = build_evaluation_prompt(
        interview_transcript
    )

    raw_response = evaluate_with_nvidia(
        prompt,
        response_format=build_response_schema(),
    )

    parsed_response = extract_json(
        raw_response
    )

    validated_result = validate_evaluation(
        parsed_response
    )

    return validated_result


# ============================================================
# Simple manual test
# ============================================================

if __name__ == "__main__":

    sample_transcript = """
AI Interviewer:
Can you explain what React is?

Candidate:
React is a JavaScript library used to build user interfaces.

AI Interviewer:
Good. Can you explain why components are useful in React?

Candidate:
Components let us break the UI into reusable pieces.

AI Interviewer:
Can you give an example of when you would use state?
"""

    print("=" * 70)
    print("INTERVIEW QUALITY EVALUATION TEST")
    print("=" * 70)

    result = evaluate_interview(
        sample_transcript
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )
