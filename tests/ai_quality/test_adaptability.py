
"""
AI Quality Test: Interviewer Adaptability

Purpose:
    Verify that the AI interviewer adapts its behavior
    according to the candidate's response.

Evaluation model:
    NVIDIA Nemotron (OpenAI-compatible API)
"""

from deepeval import assert_test
from deepeval.metrics import GEval
from config.settings import NVIDIA_BASE_URL, NVIDIA_MODEL
from evaluators.nvidia_judge import get_nvidia_judge
from deepeval.test_case import LLMTestCase, SingleTurnParams


THRESHOLD = 0.70


def create_model():
    """Create the NVIDIA Nemotron judge."""

    return get_nvidia_judge()


def create_adaptability_metric(name: str, criteria: str):
    """Create the GEval metric used for adaptability."""

    return GEval(
        name=name,
        criteria=criteria,
        # DeepEval only sends `evaluation_steps` to the judge
        # (`criteria` is used to auto-generate steps when none
        # are given), so the scenario-specific expectations
        # must be part of the steps themselves.
        evaluation_steps=[
            f"Read the scenario-specific expectations:\n{criteria}",
            "Identify the candidate's situation.",
            "Identify what adaptation the interviewer should make, "
            "using the scenario-specific expectations above.",
            "Determine whether the interviewer response makes that adaptation.",
            "Check that the response stays relevant to the interview.",
            "Check that the response is professional and natural.",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        model=create_model(),
        threshold=THRESHOLD,
    )


def run_adaptability_test(
    test_name: str,
    candidate_situation: str,
    interviewer_response: str,
    criteria: str,
):
    """Run one adaptability evaluation."""

    print()
    print("=" * 70)
    print(f"AI ADAPTABILITY TEST: {test_name}")
    print("=" * 70)

    print()
    print(f"Evaluation model: {NVIDIA_MODEL}")
    print(f"NVIDIA base URL: {NVIDIA_BASE_URL}")
    print(f"Quality threshold: {THRESHOLD}")

    print()
    print("Creating evaluation model...")
    create_model()
    print("[PASS] NVIDIA Nemotron evaluation model configured.")

    print()
    print("Creating adaptability metric...")

    metric = create_adaptability_metric(
        name=f"Adaptability - {test_name}",
        criteria=criteria,
    )

    print("[PASS] Adaptability metric configured.")

    test_case = LLMTestCase(
        input=candidate_situation,
        actual_output=interviewer_response,
    )

    print()
    print("Evaluating interviewer adaptability...")

    assert_test(
        test_case,
        [metric],
    )

    print()
    print("=" * 70)
    print(f"ADAPTABILITY TEST PASSED: {test_name}")
    print("=" * 70)


def test_adaptability_to_short_answer():
    """The interviewer should probe when the candidate gives little detail."""

    run_adaptability_test(
        test_name="Short Candidate Answer",
        candidate_situation="""
The interviewer asked:
"Tell me about a challenging frontend project you worked on."

The candidate answered:
"My student management application."

The candidate has provided very little detail.
""",
        interviewer_response="""
Could you tell me a little more about that project?

What was the most challenging technical problem
you personally had to solve?
""",
        criteria="""
The candidate gave a short answer.

The interviewer should adapt by asking a relevant,
focused follow-up question that encourages more detail.

The response should:
- recognize that more information is needed
- remain related to the candidate's project
- avoid inventing candidate information
- continue the interview naturally
""",
    )


def test_adaptability_to_confusion():
    """The interviewer should simplify a confusing question."""

    run_adaptability_test(
        test_name="Candidate Confusion",
        candidate_situation="""
The interviewer asked:
"How would you optimize the rendering pipeline
of a React application?"

The candidate answered:
"I'm not sure what you mean by rendering pipeline."

The candidate clearly does not understand the wording.
""",
        interviewer_response="""
No problem.

Let me put it another way.

Suppose a React page feels slow because some
components are rendering more often than necessary.

How would you investigate which components are
causing those unnecessary renders?
""",
        criteria="""
The candidate did not understand the interviewer's
technical wording.

The interviewer should adapt by rephrasing the
question in simpler and more concrete language.

The response should:
- acknowledge the confusion
- simplify the wording
- preserve the original technical objective
- give the candidate another opportunity to answer
""",
    )


def test_adaptability_to_strong_answer():
    """The interviewer should increase depth after a strong answer."""

    run_adaptability_test(
        test_name="Strong Candidate Answer",
        candidate_situation="""
The interviewer asked:
"What happens when a React component re-renders?"

The candidate gave a detailed and technically strong
answer explaining component evaluation, reconciliation,
virtual DOM comparison, parent-child rendering behavior,
and unnecessary child renders.

The candidate demonstrated strong knowledge.
""",
        interviewer_response="""
That's a good explanation.

Let's go one level deeper.

Suppose a child component is re-rendering even though
the data it receives has not changed.

How would you investigate the cause and decide whether
an optimization such as React.memo is appropriate?
""",
        criteria="""
The candidate demonstrated strong technical knowledge.

The interviewer should adapt by increasing the
technical depth of the next question.

The response should:
- recognize the strong answer
- build upon the demonstrated knowledge
- ask a more challenging follow-up
- remain relevant to React performance
""",
    )


def test_adaptability_to_clarification_request():
    """The interviewer should clarify when the candidate asks for an example."""

    run_adaptability_test(
        test_name="Candidate Clarification Request",
        candidate_situation="""
The interviewer asked:
"How would you approach state management in
a large React application?"

The candidate answered:
"Could you give me an example of the kind of
situation you're referring to?"

The candidate is asking for clarification.
""",
        interviewer_response="""
Sure.

Imagine that several unrelated components need
access to the same logged-in user's information.

How would you decide whether to keep that state
local, lift it to a parent, or use shared state?
""",
        criteria="""
The candidate asked for clarification.

The interviewer should adapt by providing a concrete
example that makes the original question easier
to understand.

The response should:
- directly answer the clarification request
- provide a concrete example
- preserve the original interview topic
- allow the candidate to answer afterward
""",
    )


def test_adaptability_to_incorrect_answer():
    """The interviewer should correct and reassess an incorrect answer."""

    run_adaptability_test(
        test_name="Incorrect Candidate Answer",
        candidate_situation="""
The interviewer asked:
"What is the purpose of React props?"

The candidate answered:
"Props are mainly used to store state inside
a component."

The candidate appears to have confused props
with component state.
""",
        interviewer_response="""
Props are used to pass data into a component,
while state represents data managed by the component.

Let's check that distinction with a simple example.

If a parent component needs to pass a user's name
to a child component, how would you use props to
do that?
""",
        criteria="""
The candidate gave an incorrect technical answer.

The interviewer should adapt by briefly addressing
the misconception and then asking a simpler follow-up
question to check the candidate's understanding.

The response should:
- address the misconception
- remain professional
- avoid embarrassing the candidate
- give the candidate another opportunity to answer
- continue evaluating the same React concept
""",
    )
