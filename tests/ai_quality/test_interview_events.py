"""
AI Quality Tests: Interview Events

These tests verify that the AI interviewer handles
different interview situations correctly.

Evaluation model:
    NVIDIA Nemotron (OpenAI-compatible API)
"""

from deepeval import assert_test
from deepeval.metrics import GEval
from config.settings import NVIDIA_BASE_URL, NVIDIA_MODEL
from evaluators.nvidia_judge import get_nvidia_judge
from deepeval.test_case import LLMTestCase, SingleTurnParams


# ============================================================
# Configuration
# ============================================================

QUALITY_THRESHOLD = 0.70


# ============================================================
# Evaluation Model
# ============================================================

def create_model():
    """
    Create the NVIDIA Nemotron judge.
    """

    return get_nvidia_judge()


# ============================================================
# Generic Event Evaluator
# ============================================================

def evaluate_event(
    name: str,
    conversation: str,
    interviewer_response: str,
    criteria: str,
) -> None:
    """
    Evaluate one interview event.

    The evaluator receives:
        - the interview context
        - the AI interviewer's response
        - event-specific quality criteria
    """

    print()
    print("=" * 70)
    print(f"EVENT: {name}")
    print("=" * 70)

    print(f"Evaluation model: {NVIDIA_MODEL}")
    print(f"Quality threshold: {QUALITY_THRESHOLD}")

    model = create_model()

    print("[PASS] NVIDIA Nemotron evaluation model configured.")

    metric = GEval(
        name=f"Interview Event - {name}",

        criteria=criteria,

        # DeepEval only sends `evaluation_steps` to the judge
        # (`criteria` is used to auto-generate steps when none
        # are given), so the event-specific expectations must
        # be part of the steps themselves.
        evaluation_steps=[
            "Read the event-specific expectations for the "
            f"interviewer:\n{criteria}",

            "Identify what happened in the interview.",

            "Identify what the interviewer is expected "
            "to do in this situation, using the "
            "event-specific expectations above.",

            "Check whether the interviewer response "
            "matches the expected behavior.",

            "Check whether the response stays relevant "
            "to the current interview topic.",

            "Check whether the response maintains "
            "conversation context.",

            "Check whether the response is professional "
            "and natural.",
        ],

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],

        model=model,

        threshold=QUALITY_THRESHOLD,
    )

    test_case = LLMTestCase(
        input=conversation,
        actual_output=interviewer_response,
    )

    print()
    print("Evaluating event...")

    assert_test(
        test_case,
        [metric],
    )

    print()
    print(f"[PASS] {name}")


# ============================================================
# 1. Interview Start
# ============================================================

def test_interview_start():
    """
    Verify that the interviewer starts the interview naturally.
    """

    evaluate_event(
        name="Interview Start",

        conversation="""
The interview has just started.

Candidate:
Hello, I am ready for the interview.

Interviewer:
""",

        interviewer_response="""
Welcome. Let's begin with a brief introduction.

Could you tell me about yourself and your experience
with frontend development?
""",

        criteria="""
This is the beginning of an interview.

The interviewer should:

- acknowledge or welcome the candidate
- maintain a professional tone
- begin the interview naturally
- ask an appropriate introductory question

The response should not jump to an unrelated topic.

The response should be considered successful if it
appropriately starts the interview.
""",
    )


# ============================================================
# 2. Technical Answer Follow-Up
# ============================================================

def test_technical_answer_follow_up():
    """
    Verify that the interviewer follows up after
    a candidate gives a valid technical answer.
    """

    evaluate_event(
        name="Technical Answer Follow-Up",

        conversation="""
Interviewer:
What is the difference between state and props in React?

Candidate:
Props are passed from a parent component to a child,
while state is managed by the component and can change
over time.

Interviewer:
""",

        interviewer_response="""
That's correct.

Can you give me an example of when you would use state
instead of props in a React application?
""",

        criteria="""
The candidate has already answered the interviewer's
React question.

The interviewer response should be evaluated only as
the NEXT interviewer turn.

A successful response should:

- acknowledge the candidate's answer
- remain on the React state/props topic
- ask a relevant follow-up question
- explore the candidate's understanding more deeply
- avoid repeating the exact original question

IMPORTANT:

The interviewer is NOT expected to answer the
follow-up question.

The interviewer is expected to ASK the follow-up question.

Therefore, a response such as:

"That's correct. Can you give me an example..."

is correct behavior.

Do not penalize the response because the follow-up
question itself has not been answered yet.
""",
    )


# ============================================================
# 3. Clarification Request
# ============================================================

def test_clarification_request():
    """
    Verify that the interviewer handles a clarification
    request without losing context.
    """

    evaluate_event(
        name="Clarification Request",

        conversation="""
Interviewer:
How would you optimize a React application?

Candidate:
Could you clarify what kind of performance problem
you mean?

Interviewer:
""",

        interviewer_response="""
Sure.

By optimization, I mean improving problems such as
slow rendering, unnecessary re-renders, or inefficient
handling of large amounts of data.

How would you investigate and improve those issues
in a React application?
""",

        criteria="""
The candidate has asked the interviewer to clarify
the original question.

The interviewer should:

- acknowledge the clarification request
- clarify what is meant by optimization
- provide useful examples
- remain focused on React performance
- ask the candidate to answer the clarified question

The interviewer should NOT answer the technical
question on behalf of the candidate.

The purpose of this event is clarification.
""",
    )


# ============================================================
# 4. Incomplete Answer
# ============================================================

def test_incomplete_answer():
    """
    Verify that the interviewer recognizes an incomplete answer.
    """

    evaluate_event(
        name="Incomplete Answer",

        conversation="""
Interviewer:
How do you handle API errors in a React application?

Candidate:
I usually use fetch and...

Interviewer:
""",

        interviewer_response="""
Please continue.

How would you handle the error state and communicate
the failure to the user?
""",

        criteria="""
The candidate has not finished answering the question.

The interviewer should give the candidate an
opportunity to continue.

A successful response should:

- recognize that the answer is incomplete
- encourage the candidate to continue
- remain on API error handling
- optionally ask a focused prompt

The interviewer should not pretend that the
candidate has completed the answer.
""",
    )


# ============================================================
# 5. Off-Topic Answer
# ============================================================

def test_off_topic_answer():
    """
    Verify that the interviewer brings an off-topic
    candidate response back to the interview topic.
    """

    evaluate_event(
        name="Off Topic Answer",

        conversation="""
Interviewer:
How would you manage state in a React application?

Candidate:
I really enjoy working on UI design and I recently
started learning Figma.

Interviewer:
""",

        interviewer_response="""
That's useful frontend experience.

Coming back to React state management, how would you
decide whether state should remain local or be shared
between multiple components?
""",

        criteria="""
The candidate's response is unrelated to the
current React state-management question.

The interviewer should:

- remain professional
- avoid getting distracted
- bring the conversation back to React state management
- ask a relevant question
- preserve the original interview context

The response should not completely abandon
the original topic.
""",
    )


# ============================================================
# 6. Incorrect Technical Answer
# ============================================================

def test_incorrect_technical_answer():
    """
    Verify that the interviewer handles an incorrect
    technical answer professionally.
    """

    evaluate_event(
        name="Incorrect Technical Answer",

        conversation="""
Interviewer:
What is a JavaScript closure?

Candidate:
A closure is when JavaScript automatically converts
a variable into a global variable.

Interviewer:
""",

        interviewer_response="""
That's not quite correct.

A closure occurs when a function retains access to
variables from its surrounding lexical scope.

Can you give me a practical example where closures
could be useful?
""",

        criteria="""
The candidate's technical answer is incorrect.

The interviewer should:

- handle the mistake professionally
- avoid insulting the candidate
- correct or challenge the misconception
- continue evaluating the candidate
- ask a relevant follow-up question

The response should remain constructive.
""",
    )


# ============================================================
# 7. Repeat Question
# ============================================================

def test_repeat_question():
    """
    Verify that the interviewer repeats a question
    when the candidate did not hear it.
    """

    evaluate_event(
        name="Repeat Question",

        conversation="""
Interviewer:
What is the difference between let and const?

Candidate:
Sorry, I didn't hear the question.
Could you repeat it?

Interviewer:
""",

        interviewer_response="""
Of course.

What is the difference between let and const
in JavaScript?
""",

        criteria="""
The candidate explicitly asked the interviewer
to repeat the question.

The interviewer should:

- acknowledge the request
- repeat or clearly restate the same question
- preserve the original meaning
- avoid changing topics

The response should be concise and professional.
""",
    )


# ============================================================
# 8. Strong Technical Answer
# ============================================================

def test_strong_technical_answer():
    """
    Verify that the interviewer increases technical
    depth after a strong candidate answer.
    """

    evaluate_event(
        name="Strong Technical Answer",

        conversation="""
Interviewer:
How would you debug a React component that is
rendering unnecessarily?

Candidate:
I would first reproduce the problem and use React
Developer Tools to identify unnecessary renders.

Then I would check changing props, state and parent
renders before considering memoization.

Interviewer:
""",

        interviewer_response="""
That's a good approach.

How would you determine whether React.memo would
actually improve this component rather than simply
adding unnecessary optimization?
""",

        criteria="""
The candidate has provided a strong answer about
debugging React rendering.

The interviewer should:

- acknowledge the answer
- avoid repeating the same basic question
- increase the technical depth
- ask a meaningful reasoning-based follow-up
- remain focused on React performance

The interviewer should continue evaluating the
candidate rather than ending the discussion.
""",
    )


# ============================================================
# 9. Candidate Does Not Know
# ============================================================

def test_candidate_does_not_know():
    """
    Verify that the interviewer responds professionally
    when the candidate does not know an answer.
    """

    evaluate_event(
        name="Candidate Does Not Know",

        conversation="""
Interviewer:
Can you explain event delegation in JavaScript?

Candidate:
I am not sure about that.

Interviewer:
""",

        interviewer_response="""
No problem.

Let's move to another JavaScript topic.

How would you explain event bubbling?
""",

        criteria="""
The candidate does not know the answer.

The interviewer should:

- remain professional
- avoid insulting the candidate
- accept the response
- continue the interview naturally
- ask another relevant question or move forward

The response should not embarrass or pressure
the candidate.
""",
    )


# ============================================================
# 10. Interview Completion
# ============================================================

def test_interview_completion():
    """
    Verify that the interviewer closes the interview
    appropriately.
    """

    evaluate_event(
        name="Interview Completion",

        conversation="""
The interview has already covered:

- React
- JavaScript
- REST APIs
- Debugging
- Performance
- HTML/CSS
- Project experience
- Problem solving

The candidate has answered the final question.

Interviewer:
""",

        interviewer_response="""
Thank you for taking the time to complete the interview.

We've covered your frontend experience, React,
JavaScript, APIs, debugging and performance.

This concludes the interview.
""",

        criteria="""
The interview has reached its intended conclusion.

The interviewer should:

- clearly indicate that the interview is ending
- thank the candidate
- maintain a professional tone
- provide a natural closing
- avoid introducing an unnecessary new question

The response should appropriately conclude the interview.
""",
    )
