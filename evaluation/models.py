import os
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)


load_dotenv()


class NvidiaInferenceError(RuntimeError):
    """
    NVIDIA API / inference failure.

    This is an infrastructure failure, NOT an interview
    quality failure. Callers must report it separately from
    metric threshold failures.
    """


# Transient failures worth retrying: connection problems,
# timeouts, rate limits, and provider 5xx responses.
# Anything else (401/403 auth, 400 bad request, ...) is
# permanent and raised immediately.
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 4.0


def _is_transient(error: Exception) -> bool:

    if isinstance(
        error,
        (APIConnectionError, APITimeoutError, RateLimitError),
    ):
        return True

    if isinstance(error, APIStatusError):
        return 500 <= error.status_code < 600

    return False


NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3-nano-30b-a3b",
)


if not NVIDIA_API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY is not configured. "
        "Add NVIDIA_API_KEY to your .env file."
    )


client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
)


def evaluate_with_nvidia(
    prompt: str,
    response_format: dict | None = None,
) -> str:
    """
    Send an evaluation prompt to NVIDIA Nemotron.

    When a response_format is provided (e.g. a json_schema
    built from the rubric), it is enforced server-side so
    the model cannot return malformed keys or prose around
    the JSON.

    Transient API failures (connection errors, timeouts,
    rate limits, 5xx) are retried with backoff. When retries
    are exhausted, or the error is permanent, an
    NvidiaInferenceError is raised so the caller can report
    an infrastructure failure instead of a quality failure.
    """

    extra_kwargs: dict = {}

    if response_format is not None:
        extra_kwargs["response_format"] = response_format

    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:

            completion = client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                top_p=1,
                max_tokens=16384,
                extra_body={
                    "reasoning_budget": 16384,
                },
                stream=False,
                **extra_kwargs,
            )

        except Exception as error:

            last_error = error

            status = getattr(error, "status_code", None)

            if not _is_transient(error):
                raise NvidiaInferenceError(
                    f"NVIDIA request failed permanently "
                    f"(status={status}, "
                    f"error={type(error).__name__}): {error}"
                ) from error

            print(
                f"NVIDIA transient failure on attempt "
                f"{attempt}/{MAX_ATTEMPTS} "
                f"(status={status}, "
                f"error={type(error).__name__}). "
                + (
                    "Retrying..."
                    if attempt < MAX_ATTEMPTS
                    else "Giving up."
                )
            )

            if attempt < MAX_ATTEMPTS:
                time.sleep(
                    RETRY_BACKOFF_SECONDS * attempt
                )

            continue

        if not completion.choices:
            raise NvidiaInferenceError(
                "NVIDIA returned no completion choices."
            )

        content = completion.choices[0].message.content

        if content is None:
            raise NvidiaInferenceError(
                "NVIDIA returned an empty response."
            )

        return content

    # The same request failed on every attempt even though
    # the errors were transient: surface that clearly.
    raise NvidiaInferenceError(
        f"NVIDIA request failed {MAX_ATTEMPTS} consecutive "
        f"times (transient errors, retries exhausted). "
        f"Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error


def test_nvidia_connection() -> str:
    """
    Verify that NVIDIA Nemotron is reachable.
    """

    prompt = """
Respond with exactly:

NVIDIA_CONNECTION_OK

Do not provide anything else.
"""

    return evaluate_with_nvidia(prompt)


if __name__ == "__main__":

    print("=" * 60)
    print("NVIDIA NEMOTRON CONNECTION TEST")
    print("=" * 60)

    print(f"Model: {NVIDIA_MODEL}")
    print(f"Base URL: {NVIDIA_BASE_URL}")
    print()

    try:

        response = test_nvidia_connection()

        print("Response:")
        print(response)

        print()
        print("=" * 60)
        print("NVIDIA CONNECTION TEST PASSED")
        print("=" * 60)

    except Exception as error:

        print()
        print("=" * 60)
        print("NVIDIA CONNECTION TEST FAILED")
        print("=" * 60)

        print(f"Error: {error}")

        raise
