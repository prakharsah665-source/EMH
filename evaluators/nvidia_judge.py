"""
Shared DeepEval judge backed by GPT-OSS via the NVIDIA NIM
OpenAI-compatible API (https://integrate.api.nvidia.com/v1).

All AI quality tests use this judge so the evaluation model
is configured in exactly one place (.env: API_KEY, BASE_URL,
MODEL_NAME, read by config.settings as JUDGE_*).

The judge subclasses DeepEval's LocalModel to enforce NVIDIA
structured output on every call: when DeepEval requests a
pydantic schema, the exact JSON schema is sent as
response_format, so the model cannot return malformed keys
(e.g. ".score") or prose around the JSON. GPT-OSS honours
json_schema on this endpoint and keeps its reasoning out of
message.content (verified 2026-09-08).
"""

from typing import Optional, Tuple, Union

from pydantic import BaseModel

from deepeval.constants import ProviderSlug as PS
from deepeval.models.llms.local_model import LocalModel
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.retry_policy import create_retry_decorator

from config.settings import (
    JUDGE_API_KEY,
    JUDGE_BASE_URL,
    JUDGE_MODEL,
)


retry_nvidia = create_retry_decorator(PS.LOCAL)


class NvidiaStructuredJudge(LocalModel):
    """
    LocalModel with per-call structured output enforcement.

    Deterministic (temperature=0) and non-streaming.
    """

    @staticmethod
    def _response_format(
        schema: Optional[type[BaseModel]],
    ) -> dict:
        if schema is None:
            return {"type": "json_object"}

        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
            },
        }

    @retry_nvidia
    def generate(
        self,
        prompt: str,
        schema: Optional[type[BaseModel]] = None,
    ) -> Tuple[Union[str, BaseModel], float]:

        client = self.load_model(async_mode=False)

        response = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=False,
            response_format=self._response_format(schema),
            **self.generation_kwargs,
        )

        content = response.choices[0].message.content

        if schema:
            return (
                schema.model_validate(
                    trim_and_load_json(content)
                ),
                0.0,
            )

        return content, 0.0

    @retry_nvidia
    async def a_generate(
        self,
        prompt: str,
        schema: Optional[type[BaseModel]] = None,
    ) -> Tuple[Union[str, BaseModel], float]:

        client = self.load_model(async_mode=True)

        response = await client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=False,
            response_format=self._response_format(schema),
            **self.generation_kwargs,
        )

        content = response.choices[0].message.content

        if schema:
            return (
                schema.model_validate(
                    trim_and_load_json(content)
                ),
                0.0,
            )

        return content, 0.0


def get_nvidia_judge() -> NvidiaStructuredJudge:
    """
    Return the shared judge (GPT-OSS on the NVIDIA endpoint)
    for DeepEval metrics.
    """

    if not JUDGE_API_KEY:
        raise RuntimeError(
            "API_KEY (judge model key) is not configured. "
            "Add API_KEY, BASE_URL and MODEL_NAME to the .env file."
        )

    return NvidiaStructuredJudge(
        model=JUDGE_MODEL,
        base_url=JUDGE_BASE_URL,
        api_key=JUDGE_API_KEY,
        temperature=0,
    )
