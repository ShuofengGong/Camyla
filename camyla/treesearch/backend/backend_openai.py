import json
import logging
import time
import httpx  # imported at module top so it's available in _setup_openai_client
from functools import partial

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print
import os
import backoff
import requests
import time
from typing import Dict, List, Optional, Union, Tuple
import json_repair
import pydantic
try:
    import litellm
except ImportError:
    pass

logger = logging.getLogger("camyla")

_client: openai.OpenAI = None  # type: ignore



OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client():
    global _client
    from camyla.model_config import get_endpoint
    ep = get_endpoint()  # default_endpoint
    _client = openai.OpenAI(
        api_key=ep["api_key"],
        base_url=ep["base_url"],
        max_retries=0,
        timeout=httpx.Timeout(
            connect=50.0,   # connect timeout: 10x (was 5s -> 50s)
            read=600.0,    # read timeout: 10x (was 600s -> 6000s)
            write=50.0,     # write timeout: 10x (was 5s -> 50s)
            pool=50.0,      # pool timeout: 10x (was 5s -> 50s)
        )
    )

# httpx has been moved to the top of the file for import

def on_backoff(details: Dict) -> None:
    exception = details.get('exception', 'Unknown')
    print(
        f"🔄 API call retry: waiting {details['wait']:0.1f}s, attempt {details['tries']} "
        f"(function: {details['target'].__name__}, time: {time.strftime('%X')})"
    )
    print(f"   Error type: {type(exception).__name__}: {exception}")
    # logger.warning(f"API call retry - attempt {details['tries']}: {exception}")



# Exponential backoff: factor=20, base=2 caps intervals at 20s, 40s, 80s, 160s... (full_jitter then randomizes within 0~value)
# Formula: factor * base^n, where n starts at 0
# factor=20, base=2: 20*2^0=20, 20*2^1=40, 20*2^2=80, 20*2^3=160 ...
# Note: changing only base has no effect because base^0 is always 1, so the first retry would only wait 0~1s.
@backoff.on_exception(
    partial(backoff.expo, base=2, factor=20),  # Retry intervals capped around 20s, 40s, 80s, 160s... (average ~10s, 20s, 40s, 80s after full_jitter)
    (
        # OpenAI API errors
        openai.APIConnectionError,     # API connection error
        openai.APITimeoutError,        # API timeout
        openai.RateLimitError,         # rate-limit error
        openai.InternalServerError,    # internal server error
        openai.APIError,               # generic API error

        # Network errors
        requests.exceptions.HTTPError,      # HTTP error
        requests.exceptions.ConnectionError, # connection error
        requests.exceptions.Timeout,        # request timeout
        requests.exceptions.RequestException, # generic request exception
        httpx.RequestError,                 # httpx network error
        httpx.ConnectError,                 # httpx connect error
        httpx.TimeoutException,             # httpx timeout

        # Data-handling errors (often caused by network issues)
        AttributeError,                # attribute error (e.g. accessing attributes of None)
        TypeError,                     # type error
        ValueError,                    # value error
        AssertionError,                # assertion error
        RuntimeError,                  # runtime error (our custom raises)

        # JSON parsing errors (may come from truncated responses)
        json.JSONDecodeError,          # JSON decode error

        # Pydantic validation errors (caused by malformed OpenRouter/Litellm responses)
        pydantic.ValidationError,

        # Litellm-specific errors (if available)
        *(
            (litellm.APIConnectionError, litellm.APIError)
            if 'litellm' in globals() else ()
        ),
    ),
    on_backoff=on_backoff,
    max_tries=15,  # bumped to 15 retries to handle Connection reset issues
    max_time=1800, # max total retry time: 30 minutes
)
def query(
    system_message: Optional[str],
    user_message: Optional[str],
    func_spec: Optional[FunctionSpec] = None,
    **model_kwargs,
) -> Tuple[OutputType, float, int, int, dict]:
    _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)
    
    '''
    # if user_message is None and filtered_kwargs['model'] == 'deepseek-reasoner':
    if user_message is None:
        print('no user message! system prompt: ', len(messages), len(system_message))
        for idx in range(len(messages)):
            messages[idx]['role'] = 'user'
    '''

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        
        # Grok/xAI models do not support forced tool_choice; only 'auto' mode is supported.
        # For other models, force tool_choice to make sure the function is invoked.
        model_name = filtered_kwargs.get("model", "").lower()
        if "grok" not in model_name and "x-ai" not in model_name:
            # Only set forced tool_choice for non-Grok models
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
        
    t0 = time.time()
    
    '''completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    
    '''
        
    completion = _client.chat.completions.create(messages=messages, **filtered_kwargs)

    req_time = time.time() - t0

    # Validate completion (these errors are caught by backoff and retried)
    if not completion:
        raise RuntimeError("API call failed: no response received")
    if not completion.choices:
        raise RuntimeError("API call failed: no choices in response")

    choice = completion.choices[0]

    # Debug info: dump details when function calling fails
    if func_spec is not None and not choice.message.tool_calls:
        print("🔍 Function calling debug info:")
        print(f"   Model: {filtered_kwargs.get('model', 'Unknown')}")
        print(f"   Function Spec: {func_spec.name}")
        print(f"   Tool Choice: {filtered_kwargs.get('tool_choice', 'None')}")
        print(f"   Response Content: {choice.message.content}")
        print(f"   Tool Calls: {choice.message.tool_calls}")
        print(f"   Function Call (deprecated): {getattr(choice.message, 'function_call', 'None')}")
        logger.error(f"Model {filtered_kwargs.get('model')} does not support function calling or returned an invalid format")

    if func_spec is None:
        output = choice.message.content
        if output is None:
            raise RuntimeError("API response content is empty")
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            # print(f"[cyan]Raw func call response: {choice}[/cyan]")
            output = json_repair.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    # Safely extract token info, handling None
    in_tokens = completion.usage.prompt_tokens if completion.usage and completion.usage.prompt_tokens else 0
    out_tokens = completion.usage.completion_tokens if completion.usage and completion.usage.completion_tokens else 0

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info

