import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import backoff
import httpx
import openai
import requests

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Wraps the main fields returned by OpenRouter."""

    content: Optional[str]
    usage: Dict[str, Any]
    raw: Any


class OpenRouterClient:
    """
    Minimal wrapper around OpenRouter Chat Completions, without camyla dependencies.

    Example:
        client = OpenRouterClient()
        resp = client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="openrouter/z-ai/glm-4.7",
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 0,
    ):
        from camyla.model_config import get_role
        ep = get_role("writer", group="paper_writing")

        self.api_key = api_key or ep["api_key"]
        if not self.api_key:
            raise ValueError("API key not provided; configure the endpoint used by 'writer' in llm_endpoints or via environment variables.")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url or ep["base_url"],
            max_retries=max_retries,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Call the OpenRouter chat.completions API.

        Args:
            messages: message list in OpenAI Chat format
            model: OpenRouter model name (preferably with the `openrouter/` prefix)
            tools: optional function-calling descriptors
            tool_choice: optional tool-choice strategy
            **kwargs: passed through to the openai SDK (e.g. temperature, max_tokens)
        """

        @backoff.on_exception(
            backoff.expo,
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.APIError,
                requests.exceptions.RequestException,
                httpx.RequestError,
            ),
            max_tries=5,
        )
        def _call():
            payload = {
                "messages": messages,
                "model": model,
                **kwargs,
            }
            if tools:
                payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

            return self.client.chat.completions.create(**payload)

        completion = _call()
        if not completion.choices:
            raise RuntimeError("Empty OpenRouter response or no choices returned.")

        choice = completion.choices[0]
        content = choice.message.content

        usage = {
            "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
            "completion_tokens": getattr(completion.usage, "completion_tokens", None),
            "total_tokens": getattr(completion.usage, "total_tokens", None),
        }

        return ChatResponse(content=content, usage=usage, raw=completion)

