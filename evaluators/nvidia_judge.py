"""
Shared DeepEval judge backed by NVIDIA Nemotron via the
OpenAI-compatible API (https://integrate.api.nvidia.com/v1).

All AI quality tests use this judge so the evaluation model
is configured in exactly one place (.env).

The judge subclasses DeepEval's LocalModel to enforce NVIDIA
structured output on every call: when DeepEval requests a
pydantic schema, the exact JSON schema is sent as
response_format, so the model cannot return malformed keys
(e.g. ".score") or prose around the JSON.
"""

from typing import Optional, Tuple, Union

from pydantic import BaseModel

from deepeval.constants import ProviderSlug as PS
from deepeval.models.llms.local_model import LocalModel
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.retry_policy import create_retry_decorator

from config.settings import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
)


retry_nvidia = create_retry_decorator(PS.LOCAL)


class NemotronJudge(LocalModel):
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


def get_nvidia_judge() -> NemotronJudge:
    """
    Return the NVIDIA Nemotron judge for DeepEval metrics.
    """

    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not configured. "
            "Add NVIDIA_API_KEY to the .env file."
        )

    return NemotronJudge(
        model=NVIDIA_MODEL,
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY,
        temperature=0,
    )
