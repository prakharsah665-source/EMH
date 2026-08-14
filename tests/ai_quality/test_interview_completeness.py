"""
AI Quality Test: Interview Completeness

Purpose:
    Verify that the AI interviewer conducts a sufficiently
    complete and deep technical interview.

The test evaluates:

    - Experience
    - Project understanding
    - JavaScript
    - React
    - State management
    - API integration
    - Debugging
    - Performance
    - HTML/CSS
    - Problem solving
    - Communication
    - Technical decision making
    - Follow-up depth

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


COMPLETENESS_THRESHOLD = 0.70


# ============================================================
# Evaluation model
# ============================================================

def create_evaluation_model():
    """
    Create the NVIDIA Nemotron judge.
    """

    return get_nvidia_judge()


# ============================================================
# Interview scenario
# ============================================================

def create_interview_test_case():
    """
    Create a detailed frontend interview transcript.

    The transcript intentionally contains follow-up
    questions so the evaluator can measure depth.
    """

    interview_requirements = """
    Role:
        Frontend Developer

    The interviewer should reasonably evaluate:

        1. Candidate experience
        2. Project experience
        3. JavaScript fundamentals
        4. React fundamentals
        5. React state management
        6. Component design
        7. REST API integration
        8. Error handling
        9. Debugging
        10. Performance optimization
        11. HTML/CSS
        12. Responsive UI
        13. Problem solving
        14. Technical decision making
        15. Teamwork and communication

    IMPORTANT:

    Completeness is not only about breadth.

    The interviewer should also demonstrate reasonable
    technical depth by asking meaningful follow-up questions
    when the candidate gives a specific answer.

    A good interview should:

        - explore project decisions
        - ask why a particular approach was chosen
        - ask about tradeoffs
        - ask practical debugging questions
        - investigate performance decisions
        - explore React concepts beyond definitions
        - avoid abruptly ending the interview
        - avoid repeatedly asking unrelated questions
    """

    interview_transcript = """

============================================================
1. EXPERIENCE
============================================================

INTERVIEWER:

Tell me about your experience with frontend development.

CANDIDATE:

I have worked mainly with JavaScript and React. I also
built a student management application where I worked
on the frontend, reusable components, forms, tables and
REST API integration.


============================================================
2. PROJECT EXPERIENCE
============================================================

INTERVIEWER:

Tell me about the student management application.
What problem was it solving and what parts did you
personally implement?

CANDIDATE:

It was used to manage student information. I worked
mainly on the React frontend. I created components for
students and courses, forms for adding and editing data,
tables for displaying records and API integration.

INTERVIEWER:

You mentioned reusable components. Can you give me an
example of a component you designed to be reusable and
explain what made it reusable?

CANDIDATE:

I created reusable form and table components. Instead of
hardcoding the data, I passed the required data and
configuration through props.

INTERVIEWER:

What tradeoff did you consider when making those
components reusable?

CANDIDATE:

I tried not to make the component too complicated.
If a component required too many configuration options,
I considered whether splitting it into smaller components
would make it easier to maintain.


============================================================
3. REACT STATE MANAGEMENT
============================================================

INTERVIEWER:

How did you manage state in the application?

CANDIDATE:

I used useState for local component state and props for
passing data between components.

INTERVIEWER:

How would you decide whether state should remain local
or be shared between multiple components?

CANDIDATE:

If only one component needs the state, I would keep it
local. If multiple components need the same state, I would
consider lifting the state to their common parent or using
another shared state solution.

INTERVIEWER:

What problems can occur if state is unnecessarily lifted
too high in the component tree?

CANDIDATE:

It can cause more components to receive props and can make
the data flow harder to understand. It can also result in
unnecessary re-renders.

INTERVIEWER:

How would you investigate whether those re-renders are
actually happening?

CANDIDATE:

I would use React Developer Tools and inspect component
renders. I would also check whether props or state values
are changing unnecessarily.


============================================================
4. JAVASCRIPT
============================================================

INTERVIEWER:

What is the difference between let, const and var?

CANDIDATE:

let and const are block scoped while var is function
scoped. const cannot be reassigned while let can be
reassigned.

INTERVIEWER:

Can you explain a practical situation where closure
would be useful in JavaScript?

CANDIDATE:

A closure allows a function to remember variables from
its outer scope. It can be useful for maintaining private
state or creating functions that preserve access to
specific data.


============================================================
5. API INTEGRATION
============================================================

INTERVIEWER:

How did you integrate REST APIs into the application?

CANDIDATE:

I used fetch to make requests and updated React state
with the returned data.

INTERVIEWER:

How would you handle loading, success and error states
for an API request?

CANDIDATE:

I would maintain loading and error state and update the
data state when the request succeeds. The UI would show
a loading indicator while the request is running and an
error message if it fails.

INTERVIEWER:

Suppose the API returns a successful response but the
UI still displays stale data. How would you debug that?

CANDIDATE:

I would first inspect the network response. Then I would
check whether the React state was updated with the new
response and whether the component was receiving the
updated state.


============================================================
6. DEBUGGING
============================================================

INTERVIEWER:

Imagine a button works sometimes but fails after a state
update. What would your debugging process look like?

CANDIDATE:

First I would reproduce the issue. Then I would inspect
the browser console, event handler and component state.
I would use breakpoints to see whether the handler is
being called and whether the state contains the expected
value.

INTERVIEWER:

If you discovered that the handler was using an outdated
state value, what would you investigate next?

CANDIDATE:

I would check how the state is being captured by the
function and whether the function is recreated correctly.
I would also inspect closures and React state update
behavior.


============================================================
7. PERFORMANCE
============================================================

INTERVIEWER:

Suppose the application becomes slow when displaying
thousands of student records. What would you investigate?

CANDIDATE:

I would investigate unnecessary renders, large API
responses and expensive operations during rendering.
I could also use pagination or virtualization.

INTERVIEWER:

How would you determine whether unnecessary renders are
actually responsible for the performance problem?

CANDIDATE:

I would profile the application using React Developer
Tools and browser performance tools. I would identify
which components are rendering frequently and determine
what props or state changes are causing those renders.

INTERVIEWER:

Would you automatically use useMemo and useCallback
to solve the problem?

CANDIDATE:

No. I would first identify the actual performance issue.
Memoization has its own cost, so I would only use it when
there is a measurable benefit.


============================================================
8. HTML/CSS
============================================================

INTERVIEWER:

How would you make the application responsive?

CANDIDATE:

I would use responsive CSS, Flexbox, Grid and media
queries.

INTERVIEWER:

When would you choose CSS Grid over Flexbox?

CANDIDATE:

I would generally use Grid when I need two-dimensional
layout control and Flexbox when the layout is primarily
one-dimensional.


============================================================
9. PROBLEM SOLVING
============================================================

INTERVIEWER:

Tell me about a difficult technical problem you faced
during development.

CANDIDATE:

The application became slow when displaying a large
amount of student data. I investigated the rendering
behavior and optimized how the data was displayed.

INTERVIEWER:

How did you determine what the actual bottleneck was
instead of guessing?

CANDIDATE:

I reproduced the problem, checked the browser performance
tools and inspected component renders. That helped me
identify where the unnecessary work was happening.


============================================================
10. TEAMWORK
============================================================

INTERVIEWER:

Tell me about a technical disagreement you had with
another developer.

CANDIDATE:

We had different approaches for implementing a feature.
We discussed the tradeoffs and selected the approach that
was easier to maintain and integrate.


============================================================
11. FINAL TECHNICAL FOLLOW-UP
============================================================

INTERVIEWER:

Looking at your student management application today,
what is one technical decision you would change and why?

CANDIDATE:

I would improve the application's state and data fetching
architecture and make the components more modular.

INTERVIEWER:

What would you change specifically?

CANDIDATE:

I would separate API logic from presentation components,
centralize common data-fetching behavior where appropriate,
and improve error and loading state handling.


============================================================
12. CONCLUSION
============================================================

INTERVIEWER:

Thank you. That gives me a good understanding of your
frontend experience and technical approach. That completes
the interview.
"""

    return LLMTestCase(
        input=(
            interview_requirements
            + "\n\n"
            + "INTERVIEW TRANSCRIPT:\n"
            + interview_transcript
        ),
        actual_output=interview_transcript,
    )


# ============================================================
# Completeness metric
# ============================================================

def create_completeness_metric(model):
    """
    Create the DeepEval metric that evaluates the
    completeness and depth of the interview.
    """

    return GEval(
        name="Interview Completeness",

        criteria=(
            "Evaluate whether the AI interviewer conducts "
            "a sufficiently complete, balanced and technically "
            "deep interview for a Frontend Developer role. "
            "The interviewer should evaluate multiple relevant "
            "areas while also following up on important answers "
            "to assess the candidate's actual understanding."
        ),

        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],

        evaluation_steps=[

            # ------------------------------------------------
            # Breadth
            # ------------------------------------------------

            (
                "Identify the major interview areas specified "
                "in the requirements."
            ),

            (
                "Determine whether the interview covers a "
                "reasonable breadth of those areas."
            ),

            (
                "Check whether candidate experience and "
                "project experience are explored."
            ),

            (
                "Check whether JavaScript and React knowledge "
                "are evaluated."
            ),

            (
                "Check whether state management and component "
                "design are discussed."
            ),

            (
                "Check whether REST API integration and "
                "error handling are evaluated."
            ),

            (
                "Check whether debugging and troubleshooting "
                "are evaluated using practical scenarios."
            ),

            (
                "Check whether frontend performance and "
                "optimization are evaluated."
            ),

            (
                "Check whether HTML, CSS and responsive "
                "design are reasonably covered."
            ),

            (
                "Check whether problem solving and technical "
                "decision making are evaluated."
            ),

            (
                "Check whether teamwork and communication "
                "are reasonably covered."
            ),

            # ------------------------------------------------
            # Depth
            # ------------------------------------------------

            (
                "Check whether the interviewer asks meaningful "
                "follow-up questions instead of simply moving "
                "from one topic to another."
            ),

            (
                "Check whether follow-up questions reference "
                "specific information from the candidate's "
                "previous answers."
            ),

            (
                "Check whether the interviewer explores the "
                "candidate's reasoning rather than only asking "
                "definition-based questions."
            ),

            (
                "Check whether the interviewer asks about "
                "tradeoffs and technical decision making."
            ),

            (
                "Check whether debugging questions require "
                "the candidate to explain a realistic process."
            ),

            (
                "Check whether performance questions explore "
                "how the candidate would identify the actual "
                "bottleneck rather than simply naming "
                "optimization techniques."
            ),

            (
                "Check whether the interviewer gives the "
                "candidate enough opportunity to demonstrate "
                "practical technical knowledge."
            ),

            # ------------------------------------------------
            # Interview progression
            # ------------------------------------------------

            (
                "Check whether the interview progresses "
                "naturally from broad experience to deeper "
                "technical questions."
            ),

            (
                "Check whether the interviewer avoids "
                "repeatedly asking unrelated questions."
            ),

            (
                "Check whether the interview ends only after "
                "a reasonable amount of technical and "
                "behavioral information has been collected."
            ),

            # ------------------------------------------------
            # Final score
            # ------------------------------------------------

            (
                "Give a final score based on breadth, depth, "
                "follow-up quality, technical coverage, "
                "problem-solving evaluation, behavioral "
                "coverage and overall completeness."
            ),
        ],

        model=model,
        threshold=COMPLETENESS_THRESHOLD,
    )


# ============================================================
# Test
# ============================================================

def test_interview_completeness():
    """
    Verify that the AI interviewer conducts a complete
    and sufficiently deep interview.
    """

    print()
    print("=" * 70)
    print("AI INTERVIEW COMPLETENESS TEST")
    print("=" * 70)

    print()
    print(
        f"Evaluation model: {NVIDIA_MODEL}"
    )

    print(
        f"NVIDIA base URL: {NVIDIA_BASE_URL}"
    )

    print(
        f"Quality threshold: {COMPLETENESS_THRESHOLD}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("Creating evaluation model...")

    model = create_evaluation_model()

    print(
        "[PASS] NVIDIA Nemotron evaluation model configured."
    )

    # --------------------------------------------------------
    # Create test case
    # --------------------------------------------------------

    print()
    print("Creating interview test case...")

    test_case = create_interview_test_case()

    print(
        "[PASS] Interview test case created."
    )

    # --------------------------------------------------------
    # Create metric
    # --------------------------------------------------------

    print()
    print(
        "Creating interview completeness metric..."
    )

    metric = create_completeness_metric(
        model
    )

    print(
        "[PASS] Interview completeness metric configured."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print(
        "Evaluating interview completeness..."
    )

    assert_test(
        test_case,
        [metric],
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INTERVIEW COMPLETENESS TEST PASSED")
    print("=" * 70)