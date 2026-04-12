from . import backend_openai
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
import backoff
import requests
import time
from typing import Dict, List, Optional, Union
import openai
import httpx
import json

def on_backoff(details: Dict) -> None:
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )

'''
@backoff.on_exception(
    backoff.expo,
    # --- edit here ---
    (
        requests.exceptions.HTTPError, # in case some code still uses requests directly
        requests.exceptions.ConnectionError, # same as above
        openai.APIConnectionError, # catches OpenAI connection errors
        openai.APITimeoutError,    # also consider catching OpenAI timeout errors
        openai.RateLimitError,     # rate-limit errors (retry as desired)
        httpx.RequestError,        # broader httpx network errors (optional fallback)
        AssertionError,
        json.JSONDecodeError
        # Add more httpx or other library-specific network exceptions as needed,
        # e.g. httpx.RemoteProtocolError, httpx.ConnectError.
    ),
    # ---------------
    on_backoff=on_backoff,
    max_tries=10, # explicit max retry count (optional)
)'''
def query(
    system_message: Optional[PromptType],
    user_message: Optional[PromptType],
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    func_spec: Optional[FunctionSpec] = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (Optional[PromptType]): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (Optional[PromptType]): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (Optional[float], optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (Optional[int], optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (Optional[FunctionSpec], optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # Handle models with beta limitations
    # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
    if model.startswith("o1"):
        if system_message and user_message is None:
            user_message = system_message
        elif system_message is None and user_message:
            pass
        elif system_message and user_message:
            system_message["Main Instructions"] = {}
            system_message["Main Instructions"] |= user_message
            user_message = system_message
        system_message = None
        # model_kwargs["temperature"] = 0.5
        model_kwargs["reasoning_effort"] = "high"
        model_kwargs["max_completion_tokens"] = 100000  # max_tokens
        # remove 'temperature' from model_kwargs
        model_kwargs.pop("temperature", None)
    else:
        model_kwargs["max_tokens"] = max_tokens

    # All models are accessed through OpenRouter
    query_func = backend_openai.query
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
