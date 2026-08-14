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

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3-nano-30b-a3b",
)

BROWSER_TIMEOUT = 30_000
INTERVIEW_TIMEOUT = 120_000