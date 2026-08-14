"""
AI Quality Test: Overall Interview Quality

Purpose:
    Evaluate the overall quality of a complete AI technical
    interview.

The evaluator considers:

    - Question relevance
    - Context retention
    - Follow-up quality
    - Adaptability
    - Technical accuracy
    - Professionalism
    - Question diversity
    - Technical depth
    - Problem solving
    - Interview completeness

Evaluation model:
    NVIDIA Nemotron (OpenAI-compatible API)
"""

from deepeval import assert_test
from config.settings import NVIDIA_BASE_URL, NVIDIA_MODEL
from evaluators.nvidia_judge import get_nvidia_judge
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
)
from deepeval.metrics import GEval


# ============================================================
# Configuration
# ============================================================


OVERALL_QUALITY_THRESHOLD = 0.70


# ============================================================
# Evaluation model
# ============================================================

def create_evaluation_model():
    """
    Create the NVIDIA Nemotron judge used as the
    DeepEval evaluation judge.
    """

    return get_nvidia_judge()


# ============================================================
# Interview transcript
# ============================================================

def create_interview_test_case():
    """
    Create a realistic complete frontend interview.

    The transcript contains both technical and behavioral
    discussion and demonstrates context-aware follow-ups.
    """

    interview_context = """
    Interview role:
        Frontend Developer

    Candidate background:
        The candidate has JavaScript and React experience
        and built a student management application.

    The AI interviewer should conduct a professional,
    relevant, technically accurate and adaptive interview.

    A high-quality interview should:

        - understand the candidate's previous answers
        - ask relevant questions
        - use meaningful follow-ups
        - explore technical reasoning
        - test practical problem solving
        - avoid unnecessary repetition
        - maintain professional communication
        - cover multiple relevant interview areas
        - progressively explore technical depth
        - conclude after gathering enough information
    """

    interview_transcript = """

    INTERVIEWER:
    Could you introduce yourself and tell me about your
    frontend development experience?

    CANDIDATE:
    I have experience with JavaScript and React. I built
    a student management application where I worked on
    reusable components, forms, tables and REST API
    integration.

    INTERVIEWER:
    Tell me more about that application. What were the
    main features and what parts did you personally build?

    CANDIDATE:
    I worked mainly on the frontend. I created reusable
    components for students and courses, implemented
    forms and tables, and connected the frontend to
    REST APIs.

    INTERVIEWER:
    You mentioned reusable components. How did you decide
    which parts of the UI should become reusable components?

    CANDIDATE:
    I looked for UI elements that appeared in multiple
    places. I tried to keep the components focused and
    passed the required data through props.

    INTERVIEWER:
    What tradeoffs did you consider when making a component
    reusable?

    CANDIDATE:
    I tried to avoid making a component so configurable
    that it became difficult to understand. If there were
    too many conditions, I would consider splitting it
    into smaller components.

    INTERVIEWER:
    How did you manage state in your React application?

    CANDIDATE:
    I used useState for local state and props for passing
    information between components. For shared state I
    considered lifting state to a common parent.

    INTERVIEWER:
    When would you avoid putting state in a global store?

    CANDIDATE:
    If only one component or a small part of the component
    tree needs the state, I would keep it local rather than
    making it global unnecessarily.

    INTERVIEWER:
    How would you identify unnecessary React re-renders?

    CANDIDATE:
    I would use React Developer Tools and inspect component
    renders. I would check whether props or state are
    changing unnecessarily.

    INTERVIEWER:
    Let's talk about JavaScript. What is a closure and
    where could you use one in a real application?

    CANDIDATE:
    A closure allows a function to retain access to variables
    from its outer scope. It can be useful for maintaining
    private state or creating functions that remember data.

    INTERVIEWER:
    How did you integrate APIs into your application?

    CANDIDATE:
    I used fetch to communicate with the backend and
    handled loading, success and error states.

    INTERVIEWER:
    Suppose the API request succeeds but the UI displays
    stale data. How would you investigate it?

    CANDIDATE:
    I would first check the network response. Then I would
    verify whether the React state was updated and whether
    the component was receiving the updated value.

    INTERVIEWER:
    Now imagine the application becomes slow when displaying
    thousands of student records. What would you investigate?

    CANDIDATE:
    I would check unnecessary renders, expensive operations
    during rendering and the amount of data being requested.
    I could consider pagination or virtualization.

    INTERVIEWER:
    Would you immediately add useMemo and useCallback?

    CANDIDATE:
    No. I would first identify the actual bottleneck.
    Memoization should be used when it provides a measurable
    benefit rather than automatically everywhere.

    INTERVIEWER:
    How would you debug a JavaScript issue that you cannot
    immediately understand?

    CANDIDATE:
    I would reproduce the issue, inspect the console and
    network requests, use breakpoints and isolate the part
    of the code causing the problem.

    INTERVIEWER:
    Tell me about a difficult problem you faced while
    developing your project.

    CANDIDATE:
    The application became slow when displaying a large
    amount of student data. I investigated component
    rendering and optimized how the data was displayed.

    INTERVIEWER:
    How did you determine that rendering was actually
    contributing to the problem?

    CANDIDATE:
    I used browser performance tools and React Developer
    Tools to inspect renders and identify unnecessary work.

    INTERVIEWER:
    How would you make your application responsive?

    CANDIDATE:
    I would use responsive CSS, Flexbox, Grid and media
    queries and test the application at different viewport
    sizes.

    INTERVIEWER:
    When would you choose CSS Grid instead of Flexbox?

    CANDIDATE:
    Grid is useful for two-dimensional layouts while
    Flexbox is generally useful for one-dimensional layouts.

    INTERVIEWER:
    Tell me about a situation where you had a technical
    disagreement with another developer.

    CANDIDATE:
    We had different approaches for implementing a feature.
    We discussed the tradeoffs and chose the approach that
    was easier to maintain and integrate.

    INTERVIEWER:
    Looking back at your student management application,
    what would you improve?

    CANDIDATE:
    I would improve the separation between API logic and
    presentation components and make error and loading
    handling more consistent.

    INTERVIEWER:
    Why would separating API logic from presentation
    components be useful?

    CANDIDATE:
    It can make the components easier to test and maintain
    because the UI does not need to contain all the API
    communication logic.

    INTERVIEWER:
    Thank you. Your answers gave me a good understanding
    of your React, JavaScript, API integration, debugging,
    performance and project experience. That completes
    the interview.
    """

    return LLMTestCase(
        input=(
            interview_context
            + "\n\nFULL INTERVIEW TRANSCRIPT:\n"
            + interview_transcript
        ),
        actual_output=interview_transcript,
    )


# ============================================================
# Overall quality metric
# ============================================================

def create_overall_quality_metric(model):
    """
    Create a GEval metric that evaluates the complete
    interview as a whole.
    """

    return GEval(
        name="Overall Interview Quality",

        criteria=(
            "Evaluate the overall quality of the AI "
            "interview for a Frontend Developer candidate. "
            "The interview should be relevant, professional, "
            "technically accurate, context-aware, adaptive, "
            "non-repetitive and sufficiently complete. "
            "The interviewer should use the candidate's "
            "answers to guide meaningful follow-up questions "
            "and should evaluate both technical knowledge "
            "and practical experience."
        ),

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],

        evaluation_steps=[

            # ------------------------------------------------
            # Relevance
            # ------------------------------------------------

            (
                "Evaluate whether the interview questions "
                "are relevant to the Frontend Developer role "
                "and the candidate's stated experience."
            ),

            # ------------------------------------------------
            # Context retention
            # ------------------------------------------------

            (
                "Check whether the interviewer correctly uses "
                "information from earlier candidate answers."
            ),

            # ------------------------------------------------
            # Follow-up quality
            # ------------------------------------------------

            (
                "Evaluate whether follow-up questions meaningfully "
                "explore details from the candidate's previous "
                "responses."
            ),

            # ------------------------------------------------
            # Adaptability
            # ------------------------------------------------

            (
                "Check whether the interviewer adapts its "
                "questions based on the candidate's responses "
                "rather than appearing to follow an unrelated "
                "fixed sequence."
            ),

            # ------------------------------------------------
            # Technical accuracy
            # ------------------------------------------------

            (
                "Check whether the interviewer's technical "
                "questions and assumptions are factually and "
                "conceptually accurate."
            ),

            # ------------------------------------------------
            # Professionalism
            # ------------------------------------------------

            (
                "Evaluate whether the interviewer maintains "
                "a professional, respectful, neutral and "
                "appropriate tone."
            ),

            # ------------------------------------------------
            # Repetition
            # ------------------------------------------------

            (
                "Check whether the interviewer avoids asking "
                "the same question or repeatedly evaluating "
                "the same concept without a useful reason."
            ),

            # ------------------------------------------------
            # Technical depth
            # ------------------------------------------------

            (
                "Evaluate whether the interviewer explores "
                "technical reasoning, tradeoffs and practical "
                "decision making rather than only asking "
                "basic definition questions."
            ),

            # ------------------------------------------------
            # Problem solving
            # ------------------------------------------------

            (
                "Check whether the interview provides the "
                "candidate with meaningful opportunities to "
                "demonstrate debugging and problem-solving "
                "ability."
            ),

            # ------------------------------------------------
            # Interview breadth
            # ------------------------------------------------

            (
                "Evaluate whether the interview covers a "
                "reasonable range of relevant areas including "
                "React, JavaScript, APIs, debugging, performance, "
                "UI development, project experience and teamwork."
            ),

            # ------------------------------------------------
            # Interview progression
            # ------------------------------------------------

            (
                "Check whether the interview progresses "
                "naturally from general experience toward "
                "deeper technical and practical questions."
            ),

            # ------------------------------------------------
            # Completeness
            # ------------------------------------------------

            (
                "Determine whether enough information was "
                "collected to reasonably evaluate the candidate "
                "for the role."
            ),

            # ------------------------------------------------
            # Final evaluation
            # ------------------------------------------------

            (
                "Give a final overall score based on relevance, "
                "context awareness, follow-up quality, "
                "adaptability, technical accuracy, "
                "professionalism, diversity, technical depth, "
                "problem solving and completeness."
            ),
        ],

        model=model,
        threshold=OVERALL_QUALITY_THRESHOLD,
    )


# ============================================================
# Test
# ============================================================

def test_overall_quality():
    """
    Evaluate the overall quality of the complete
    AI interview.
    """

    print()
    print("=" * 70)
    print("AI OVERALL INTERVIEW QUALITY TEST")
    print("=" * 70)

    print()
    print(
        f"Evaluation model: {NVIDIA_MODEL}"
    )

    print(
        f"NVIDIA base URL: {NVIDIA_BASE_URL}"
    )

    print(
        f"Quality threshold: "
        f"{OVERALL_QUALITY_THRESHOLD}"
    )

    # --------------------------------------------------------
    # 1. Create evaluation model
    # --------------------------------------------------------

    print()
    print("Creating evaluation model...")

    model = create_evaluation_model()

    print(
        "[PASS] NVIDIA Nemotron evaluation model configured."
    )

    # --------------------------------------------------------
    # 2. Create interview
    # --------------------------------------------------------

    print()
    print("Creating complete interview test case...")

    test_case = create_interview_test_case()

    print(
        "[PASS] Complete interview test case created."
    )

    # --------------------------------------------------------
    # 3. Create metric
    # --------------------------------------------------------

    print()
    print("Creating overall quality metric...")

    metric = create_overall_quality_metric(
        model
    )

    print(
        "[PASS] Overall quality metric configured."
    )

    # --------------------------------------------------------
    # 4. Evaluate
    # --------------------------------------------------------

    print()
    print(
        "Evaluating overall interview quality..."
    )

    assert_test(
        test_case,
        [metric],
    )

    # --------------------------------------------------------
    # 5. Result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OVERALL INTERVIEW QUALITY TEST PASSED")
    print("=" * 70)