"""
Interview evaluation rubric for the EMH AI interviewer.

Every criterion MUST be scored using one of these values:

0.00
0.25
0.50
0.75
1.00
"""

ALLOWED_SCORES = {
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
}


RUBRIC = {
    "technical_correctness": {
        "description": (
            "How accurately the interviewer evaluates the candidate's "
            "technical answer."
        ),
        "scores": {
            0.00: "Completely incorrect evaluation or misses the technical issue.",
            0.25: "Mostly incorrect evaluation with major technical problems.",
            0.50: "Partially correct evaluation but misses important technical details.",
            0.75: "Mostly correct evaluation with only minor omissions.",
            1.00: "Completely accurate and technically sound evaluation.",
        },
    },

    "relevance": {
        "description": (
            "Whether the interviewer's question or response is relevant "
            "to the interview topic."
        ),
        "scores": {
            0.00: "Completely irrelevant.",
            0.25: "Mostly irrelevant and poorly connected to the topic.",
            0.50: "Partially relevant but contains unnecessary or unrelated content.",
            0.75: "Relevant with minor unnecessary content.",
            1.00: "Directly relevant and focused on the interview objective.",
        },
    },

    "follow_up_quality": {
        "description": (
            "How well the interviewer follows up on the candidate's "
            "previous answer."
        ),
        "scores": {
            0.00: "No meaningful follow-up or completely disconnected follow-up.",
            0.25: "Weak follow-up that barely relates to the candidate's answer.",
            0.50: "Somewhat related follow-up but misses useful opportunities.",
            0.75: "Good follow-up that builds on the candidate's answer.",
            1.00: "Excellent, natural follow-up that meaningfully probes the answer.",
        },
    },

    "question_quality": {
        "description": (
            "Quality, clarity, and usefulness of the interviewer's questions."
        ),
        "scores": {
            0.00: "Question is unusable, confusing, or inappropriate.",
            0.25: "Poorly constructed question with major clarity problems.",
            0.50: "Understandable but generic or lacking depth.",
            0.75: "Clear and useful question with minor weaknesses.",
            1.00: "Clear, specific, well-structured, and highly useful question.",
        },
    },

    "difficulty_calibration": {
        "description": (
            "Whether the interviewer maintains an appropriate difficulty "
            "for the candidate's demonstrated level."
        ),
        "scores": {
            0.00: "Difficulty is completely inappropriate.",
            0.25: "Difficulty is substantially mismatched to the candidate.",
            0.50: "Difficulty is somewhat appropriate but inconsistent.",
            0.75: "Difficulty is generally appropriate with minor issues.",
            1.00: "Difficulty adapts appropriately to the candidate's performance.",
        },
    },

    "communication_quality": {
        "description": (
            "Clarity, naturalness, professionalism, and conversational "
            "quality of the interviewer."
        ),
        "scores": {
            0.00: "Communication is unusable or highly confusing.",
            0.25: "Frequently confusing, unnatural, or unprofessional.",
            0.50: "Understandable but noticeably awkward or inconsistent.",
            0.75: "Clear and professional with minor issues.",
            1.00: "Clear, natural, professional, and conversational.",
        },
    },

    "context_awareness": {
        "description": (
            "Whether the interviewer correctly remembers and uses "
            "information from earlier in the interview."
        ),
        "scores": {
            0.00: "Completely ignores or contradicts previous context.",
            0.25: "Frequently loses context or repeats information unnecessarily.",
            0.50: "Uses some context but misses important information.",
            0.75: "Generally maintains context with minor lapses.",
            1.00: "Consistently maintains and appropriately uses interview context.",
        },
    },

    "overall_interview_quality": {
        "description": (
            "Overall quality of the interviewer based on the individual "
            "criteria above."
        ),
        "scores": {
            0.00: "Interview quality is unacceptable.",
            0.25: "Interview quality is poor.",
            0.50: "Interview quality is acceptable but has significant weaknesses.",
            0.75: "Interview quality is good with minor weaknesses.",
            1.00: "Interview quality is excellent and consistently effective.",
        },
    },
}


def validate_score(score: float) -> bool:
    """
    Check whether a score belongs to the allowed discrete buckets.
    """

    return score in ALLOWED_SCORES


def validate_criterion(criterion: str) -> bool:
    """
    Check whether a criterion exists in the rubric.
    """

    return criterion in RUBRIC


def get_rubric() -> dict:
    """
    Return the complete evaluation rubric.
    """

    return RUBRIC
