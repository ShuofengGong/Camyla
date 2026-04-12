from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import re
from camyla.utils.token_tracker import track_token_usage
from camyla.treesearch.backend import query as backend_query
from camyla.model_config import get_model_name, get_model_temperature

logger = logging.getLogger(__name__)

MAX_NUM_TOKENS = 4096 * 10

class BackendLLMClient:
    """LLM client wrapper for the backend system; uses OpenRouter uniformly."""

    def __init__(self, model_name: str = None, temperature: float = None):
        self.model_name = model_name or get_model_name('default')
        self.temperature = temperature if temperature is not None else get_model_temperature('default')

    def generate(self,
                prompt: str,
                system_message: Optional[str] = None,
                message_history: Optional[List[Dict[str, str]]] = None,
                **kwargs) -> Tuple[str, List[Dict[str, str]]]:
        """Generate text; returns the generated text and the updated message history.

        Args:
            prompt: Prompt text.
            system_message: System message.
            message_history: List of historical messages.
            **kwargs: Additional parameters.

        Returns:
            Tuple[str, List[Dict[str, str]]]: (generated text, updated message history).
        """
        try:
            # Build message history
            messages = []
            if message_history:
                messages.extend(message_history)

            # Append current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })

            # Call backend query — now returns only the output
            response = backend_query(
                system_message=system_message,
                user_message=prompt,
                model=self.model_name,
                temperature=self.temperature,
                **kwargs
            )

            # Update message history
            new_messages = messages + [{
                "role": "assistant",
                "content": response
            }]

            return response, new_messages

        except Exception as e:
            raise Exception(f"Error in BackendLLMClient.generate: {str(e)}")

    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from text."""
        try:
            # Locate the JSON markers and extract the payload
            start = text.find("```json")
            end = text.find("```", start + 7)
            if start != -1 and end != -1:
                json_str = text[start + 7:end].strip()
                return json.loads(json_str)
            return None
        except Exception:
            return None

def get_batch_responses_from_llm(
    prompt: str,
    client: BackendLLMClient,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.7,
    n_responses: int = 1,
) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
    """Get a batch of LLM responses.

    Args:
        prompt: Prompt text.
        client: BackendLLMClient instance.
        model: Model name.
        system_message: System message.
        print_debug: Whether to print debug information.
        msg_history: Previous messages.
        temperature: Temperature parameter.
        n_responses: Number of responses to generate.

    Returns:
        Tuple[List[str], List[List[Dict[str, Any]]]]: (list of response contents, list of message histories).
    """
    msg = prompt
    if msg_history is None:
        msg_history = []
        
    # Set client temperature
    client.temperature = temperature

    # Generate responses in batch
    content = []
    new_msg_history = []
    for _ in range(n_responses):
        response, history = client.generate(
            prompt=msg,
            system_message=system_message,
            message_history=msg_history
        )
        content.append(response)
        new_msg_history.append(history)

    if print_debug:
        logger.debug("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            logger.debug('%d, %s: %s', j, msg["role"], msg["content"])
        logger.debug(content)
        logger.debug("*" * 21 + " LLM END " + "*" * 21)

    return content, new_msg_history

def get_response_from_llm(
    prompt: str,
    client: BackendLLMClient,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.7,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Get a single LLM response.

    Args:
        prompt: Prompt text.
        client: BackendLLMClient instance.
        model: Model name.
        system_message: System message.
        print_debug: Whether to print debug information.
        msg_history: Previous messages.
        temperature: Temperature parameter.

    Returns:
        Tuple[str, List[Dict[str, Any]]]: (response content, updated message history).
    """
    msg = prompt
    logger.debug('LLM call: model=%s, prompt_len=%d', model, len(prompt))
    if msg_history is None:
        msg_history = []
        
    # Set client temperature
    client.temperature = temperature

    # Invoke the generate method
    content, new_msg_history = client.generate(
        prompt=msg,
        system_message=system_message,
        message_history=msg_history
    )

    if print_debug:
        logger.debug("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            logger.debug('%d, %s: %s', j, msg["role"], msg["content"])
        logger.debug(content)
        logger.debug("*" * 21 + " LLM END " + "*" * 21)

    return content, new_msg_history

def extract_json_between_markers(llm_output: str) -> Optional[Dict]:
    """Extract JSON content from LLM output.

    Args:
        llm_output: The text emitted by the LLM.

    Returns:
        Optional[Dict]: Parsed JSON object, or None if nothing valid is found.
    """
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found

def create_client(model: str, is_vlm: bool = False) -> Tuple[BackendLLMClient, str]:
    """Create an LLM client.

    Args:
        model: Model name.
        is_vlm: Whether this is a vision-language model.

    Returns:
        Tuple[BackendLLMClient, str]: (client instance, model name).
    """
    # print(f"Using BackendLLMClient with model {model}.")
    return BackendLLMClient(model_name=model), model
