import os

from dotenv import load_dotenv


load_dotenv()


INTERVIEW_URL = os.getenv("INTERVIEW_URL")

EMH_API_URL = os.getenv(
    "EMH_API_URL",
    "https://im-api-v1-qa.easemyhiring.ai"
    "/trackInterview/interview-event",
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "gemma3:4b",
)

# LLM judge shared by every evaluation suite (DeepEval GEval
# metrics, the full-transcript rubric evaluator and the per-turn
# interviewer judge). GPT-OSS served through the NVIDIA NIM
# OpenAI-compatible endpoint; replaced the retired
# nvidia/nemotron-3-super-120b-a12b judge on 2026-09-08.
#
# Configured by the generic .env keys API_KEY / BASE_URL /
# MODEL_NAME (values are stripped: the .env lines carry spaces
# around "=").


def _env(name: str, default: str | None = None) -> str | None:
    value = (os.getenv(name) or "").strip()
    return value or default


JUDGE_API_KEY = _env("API_KEY")

JUDGE_BASE_URL = _env(
    "BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)

JUDGE_MODEL = _env(
    "MODEL_NAME",
    "openai/gpt-oss-20b",
)

BROWSER_TIMEOUT = 30_000
INTERVIEW_TIMEOUT = 120_000