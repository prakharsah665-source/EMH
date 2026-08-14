"""
AI Quality Test: Interview Events

Purpose:
    Verify that the AI interviewer handles important
    interview events correctly.

Evaluation model:
    NVIDIA Nemotron (OpenAI-compatible API)
"""

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from config.settings import NVIDIA_BASE_URL, NVIDIA_MODEL
from evaluators.nvidia_judge import NemotronJudge, get_nvidia_judge


# ============================================================
# Configuration
# ============================================================

QUALITY_THRESHOLD = 0.70


# ============================================================
# Evaluation Model
# ============================================================

def create_evaluation_model() -> NemotronJudge:
    """
    Create the shared NVIDIA Nemotron judge.
    """

    return get_nvidia_judge()


# ============================================================
# Interview Events
# ============================================================

INTERVIEW_EVENTS = [
    {
        "name": "Interview Start",

        "input": """
Interview state: START

The interview has just started.

Candidate:
Hello, I am ready for the interview.

Interviewer:
""",

        "actual_output": """
Welcome. Let's begin with a brief introduction.

Could you tell me about yourself and your experience
with frontend development?
""",

        "criteria": """
Evaluate the interviewer response for an interview START event.

A good response should:

1. Welcome or acknowledge the candidate.
2. Begin the interview naturally.
3. Ask an appropriate introductory question.
4. Maintain a professional tone.
5. Avoid unrelated technical questions at this stage.

The response should feel like a natural beginning
to a professional interview.
""",
    },

    {
        "name": "Technical Answer Follow-Up",

        "input": """
Interview state: TECHNICAL ANSWER

Interviewer:
What is the difference between state and props in React?

Candidate:
Props are passed from a parent component to a child,
while state is managed by the component and can change
over time.

Interviewer:
""",

        "actual_output": """
That's correct.

Can you give me an example of when you would use state
instead of props in a React application?
""",

        "criteria": """
Evaluate the interviewer response after a candidate
has provided a relevant technical answer.

A good response should:

1. Recognize that the candidate answered the question.
2. Stay on the React topic.
3. Ask a meaningful follow-up question.
4. Explore the candidate's understanding further.
5. Avoid repeating the exact same question.

The response should move the technical interview forward.
""",
    },

    {
        "name": "Clarification Request",

        "input": """
Interview state: CANDIDATE REQUESTS CLARIFICATION

Interviewer:
How would you optimize a React application?

Candidate:
Could you clarify what kind of performance problem
you mean?

Interviewer:
""",

        "actual_output": """
Sure.

By optimizing a React application, I mean identifying
and improving performance problems such as slow rendering,
unnecessary re-renders, or inefficient handling of
large amounts of data.

With that clarification, how would you investigate
and improve the performance of a React application?
""",

        "criteria": """
Evaluate the interviewer response to a candidate who
explicitly requested clarification.

IMPORTANT:

The candidate has NOT answered the original question yet.
The interviewer should clarify the question before expecting
the candidate to answer it.

A good response MUST:

1. Acknowledge the clarification request.
2. Explain what the interviewer means by optimization.
3. Give useful examples such as slow rendering,
   unnecessary re-renders, or inefficient data handling.
4. Keep the conversation focused on React performance.
5. Ask the candidate to answer the clarified question.

Do NOT penalize the response for failing to provide
optimization techniques itself. The purpose of this
event is clarification, not answering the technical
question on behalf of the candidate.
""",
    },

    {
        "name": "Incomplete Answer",

        "input": """
Interview state: INCOMPLETE ANSWER

Interviewer:
How do you handle API errors in a React application?

Candidate:
I usually use fetch and...

Interviewer:
""",

        "actual_output": """
Please continue.

How would you handle the error state and communicate
the failure to the user?
""",

        "criteria": """
Evaluate the interviewer response when the candidate
has started an answer but has not finished.

A good response should:

1. Recognize that the answer is incomplete.
2. Give the candidate an opportunity to continue.
3. Optionally provide a focused prompt.
4. Stay on the API error-handling topic.
5. Avoid incorrectly treating the answer as complete.
""",
    },

    {
        "name": "Off Topic Answer",

        "input": """
Interview state: OFF TOPIC ANSWER

Interviewer:
How would you manage state in a React application?

Candidate:
I really enjoy working on UI design and I recently
started learning Figma.

Interviewer:
""",

        "actual_output": """
That's useful frontend experience.

Coming back to React state management, how would you
decide whether state should remain local or be shared
between multiple components?
""",

        "criteria": """
Evaluate the interviewer response when the candidate
gives an answer unrelated to the current question.

A good response should:

1. Remain professional.
2. Avoid becoming distracted by the unrelated topic.
3. Bring the conversation back to React state management.
4. Ask a relevant question.
5. Preserve the original interview context.
""",
    },

    {
        "name": "Incorrect Technical Answer",

        "input": """
Interview state: INCORRECT TECHNICAL ANSWER

Interviewer:
What is a JavaScript closure?

Candidate:
A closure is when JavaScript automatically converts
a variable into a global variable.

Interviewer:
""",

        "actual_output": """
That's not quite correct.

A closure occurs when a function retains access to
variables from its surrounding lexical scope.

Can you give me a practical example where closures
could be useful?
""",

        "criteria": """
Evaluate the interviewer response when the candidate
gives a technically incorrect answer.

A good response should:

1. Handle the incorrect answer professionally.
2. Avoid insulting or embarrassing the candidate.
3. Correct or challenge the misconception appropriately.
4. Continue evaluating the candidate.
5. Ask a relevant follow-up question.

The interviewer should remain professional and constructive.
""",
    },

    {
        "name": "Repeat Question",

        "input": """
Interview state: CANDIDATE REQUESTS REPETITION

Interviewer:
What is the difference between let and const?

Candidate:
Sorry, I didn't hear the question.
Could you repeat it?

Interviewer:
""",

        "actual_output": """
Of course.

What is the difference between let and const
in JavaScript?
""",

        "criteria": """
Evaluate the interviewer response when the candidate
asks for the question to be repeated.

A good response should:

1. Acknowledge the request.
2. Repeat or clearly restate the same question.
3. Preserve the original meaning.
4. Avoid changing to an unrelated question.
5. Maintain a professional tone.
""",
    },

    {
        "name": "Strong Technical Answer",

        "input": """
Interview state: STRONG TECHNICAL ANSWER

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

        "actual_output": """
That's a good approach.

How would you determine whether React.memo would
actually improve this component rather than simply
adding unnecessary optimization?
""",

        "criteria": """
Evaluate the interviewer response after a strong
technical answer.

A good response should:

1. Recognize the quality of the candidate's answer.
2. Avoid asking the exact same question again.
3. Increase the technical depth.
4. Explore reasoning or practical decision-making.
5. Remain related to React performance.
""",
    },

    {
        "name": "Candidate Does Not Know",

        "input": """
Interview state: CANDIDATE DOES NOT KNOW

Interviewer:
Can you explain event delegation in JavaScript?

Candidate:
I am not sure about that.

Interviewer:
""",

        "actual_output": """
No problem.

Let's move to another JavaScript topic.

How would you explain event bubbling?
""",

        "criteria": """
Evaluate the interviewer response when the candidate
does not know the answer.

A good response should:

1. Remain professional.
2. Avoid insulting or pressuring the candidate.
3. Accept that the candidate does not know.
4. Continue the interview appropriately.
5. Ask another relevant question or move forward naturally.
""",
    },

    {
        "name": "Interview Completion",

        "input": """
Interview state: INTERVIEW COMPLETE

The interviewer has already covered:

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

        "actual_output": """
Thank you for taking the time to complete the interview.

We've covered your frontend experience, React,
JavaScript, APIs, debugging and performance.

This concludes the interview.
""",

        "criteria": """
Evaluate the interviewer response when the interview
has reached its intended conclusion.

A good response should:

1. Clearly indicate that the interview is ending.
2. Thank the candidate professionally.
3. Provide a natural closing.
4. Avoid introducing an unrelated new question.
5. Avoid unnecessarily extending the interview.
""",
    },
]


# ============================================================
# Metric
# ============================================================

def create_event_metric(
    model: NemotronJudge,
    criteria: str,
) -> GEval:
    """
    Create an event-specific GEval metric.

    Each event gets its own criteria so the evaluator
    understands exactly what behavior is expected.
    """

    return GEval(
        name="Interview Event Handling",

        criteria=criteria,

        # DeepEval only sends `evaluation_steps` to the judge
        # (`criteria` is used to auto-generate steps when none
        # are given), so the event-specific expectations must
        # be part of the steps themselves.
        evaluation_steps=[
            "Read the event-specific expectations for the "
            f"interviewer:\n{criteria}",

            "Identify the interview event.",

            "Determine what behavior is expected for this "
            "event, using the event-specific expectations "
            "above.",

            "Compare the actual interviewer response with "
            "the expected behavior.",

            "Check whether the response is relevant.",

            "Check whether the response is professional "
            "and natural.",

            "Determine whether the interviewer handles "
            "the event appropriately.",
        ],

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],

        model=model,

        threshold=QUALITY_THRESHOLD,
    )


# ============================================================
# Evaluate Event
# ============================================================

def evaluate_event(
    event: dict,
    model: NemotronJudge,
) -> None:
    """
    Evaluate one interview event.
    """

    metric = create_event_metric(
        model=model,
        criteria=event["criteria"],
    )

    test_case = LLMTestCase(
        input=event["input"],
        actual_output=event["actual_output"],
    )

    assert_test(
        test_case,
        [metric],
    )


# ============================================================
# Main Test
# ============================================================

def test_interview_events():
    """
    Evaluate all interview event scenarios.
    """

    print()
    print("=" * 70)
    print("AI INTERVIEW EVENTS TEST")
    print("=" * 70)

    print()
    print(f"Evaluation model: {NVIDIA_MODEL}")
    print(f"NVIDIA base URL: {NVIDIA_BASE_URL}")
    print(f"Quality threshold: {QUALITY_THRESHOLD}")

    # --------------------------------------------------------
    # Create evaluation model
    # --------------------------------------------------------

    print()
    print("Creating evaluation model...")

    model = create_evaluation_model()

    print(
        "[PASS] NVIDIA Nemotron evaluation model configured."
    )

    # --------------------------------------------------------
    # Load scenarios
    # --------------------------------------------------------

    print()
    print("Loading interview event scenarios...")

    events = INTERVIEW_EVENTS

    print(
        f"[PASS] Loaded {len(events)} interview events."
    )

    # --------------------------------------------------------
    # Evaluate scenarios
    # --------------------------------------------------------

    print()
    print("Evaluating interview events...")

    for index, event in enumerate(
        events,
        start=1,
    ):

        print()
        print("-" * 70)
        print(
            f"EVENT {index}/{len(events)}: "
            f"{event['name']}"
        )
        print("-" * 70)

        evaluate_event(
            event=event,
            model=model,
        )

        print(
            f"[PASS] {event['name']}"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AI INTERVIEW EVENTS TEST PASSED")
    print("=" * 70)