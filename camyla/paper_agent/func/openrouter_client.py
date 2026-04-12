import base64
import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import backoff
import httpx
import openai
import requests
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Wraps the main fields returned by OpenRouter."""

    content: Optional[str]
    usage: Dict[str, Any]
    raw: Any
    images: List[Dict[str, Any]] = field(default_factory=list)


class OpenRouterClient:
    """
    Minimal wrapper over OpenRouter Chat Completions, free of camyla dependencies.

    Example:
        client = OpenRouterClient()
        resp = client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="openrouter/z-ai/glm-4.5",
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
            raise ValueError("API key not provided. Configure the writer endpoint in llm_endpoints or via environment variables.")

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
            messages: OpenAI chat-format message list.
            model: OpenRouter model name (preferably with the `openrouter/` prefix).
            tools: Optional function-call descriptions.
            tool_choice: Optional tool-choice policy.
            **kwargs: Additional arguments forwarded to the openai SDK, e.g. temperature, max_tokens.
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
                json.JSONDecodeError,  # handle cases where the API returns invalid JSON
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

            try:
                return self.client.chat.completions.create(**payload)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Request payload: {payload}")
                logger.error(f"Response content (if any) will be caught by the caller")
                raise
            except Exception as e:
                logger.error(f"API call exception: {type(e).__name__}: {e}")
                logger.error(f"Request payload: {payload}")
                raise

        completion = _call()
        if not completion.choices:
            raise RuntimeError("OpenRouter response is empty or does not contain choices.")

        choice = completion.choices[0]
        content = choice.message.content

        usage = {
            "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
            "completion_tokens": getattr(completion.usage, "completion_tokens", None),
            "total_tokens": getattr(completion.usage, "total_tokens", None),
        }

        images = []
        model_extra = getattr(choice.message, "model_extra", None)
        if model_extra and "images" in model_extra:
            images = model_extra["images"]

        return ChatResponse(content=content, usage=usage, raw=completion, images=images)

    _IMAGE_SYSTEM_PROMPT = (
        "You are a senior AI and medical imaging researcher with extensive "
        "experience in publishing at top-tier venues (NeurIPS, CVPR, ICML, MICCAI). "
        "You excel at creating clear, elegant, and publication-ready method "
        "illustration figures for scientific papers."
    )

    @staticmethod
    def _compress_reference_image(
        image_path: Path,
        max_long_edge: int = 1024,
        jpeg_quality: int = 85,
    ) -> tuple[bytes, str]:
        """Resize and compress a reference image for API payload efficiency.

        Returns (compressed_bytes, mime_type).
        """
        img = Image.open(image_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        w, h = img.size
        long_edge = max(w, h)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            logger.info(
                f"Reference image resized: {w}x{h} -> {img.size[0]}x{img.size[1]}"
            )

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        compressed = buf.getvalue()
        logger.info(
            f"Reference image compressed: {image_path.stat().st_size} bytes -> {len(compressed)} bytes"
        )
        return compressed, "image/jpeg"

    MAX_FIGURE_BYTES = 2 * 1024 * 1024  # 2 MB

    @staticmethod
    def _compress_saved_image(
        image_path: Path,
        max_long_edge: int = 2048,
        max_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        """Downscale and re-compress an image file in-place until it fits
        *max_bytes*.  Tries optimised PNG first, then falls back to JPEG at
        decreasing quality levels."""
        file_size = image_path.stat().st_size
        if file_size <= max_bytes:
            return

        img = Image.open(image_path)
        if img.mode == "RGBA":
            has_alpha = True
        else:
            has_alpha = False
            if img.mode != "RGB":
                img = img.convert("RGB")

        w, h = img.size
        long_edge = max(w, h)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            logger.info(
                f"Resized {image_path.name}: {w}x{h} -> {img.size[0]}x{img.size[1]}"
            )

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        if buf.tell() <= max_bytes:
            image_path.write_bytes(buf.getvalue())
            logger.info(
                f"Compressed {image_path.name}: {file_size} -> {buf.tell()} bytes (PNG)"
            )
            return

        if has_alpha:
            img = img.convert("RGB")

        for quality in (92, 85, 75):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            if buf.tell() <= max_bytes:
                jpg_path = image_path.with_suffix(".jpg")
                jpg_path.write_bytes(buf.getvalue())
                if image_path.suffix.lower() != ".jpg":
                    image_path.write_bytes(buf.getvalue())
                logger.info(
                    f"Compressed {image_path.name}: {file_size} -> {buf.tell()} bytes "
                    f"(JPEG q={quality})"
                )
                return

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        image_path.write_bytes(buf.getvalue())
        logger.info(
            f"Compressed {image_path.name}: {file_size} -> {buf.tell()} bytes "
            f"(JPEG q=75, best effort)"
        )

    def generate_image(
        self,
        prompt: str,
        output_path: str | Path,
        model: str = "google/gemini-3.1-flash-image-preview",
        aspect_ratio: str = "16:9",
        image_size: str = "2K",
        max_retries: int = 2,
        reference_image_path: Optional[str | Path] = None,
        system_prompt: Optional[str] = None,
    ) -> bool:
        """
        Call an image-generation model and save the generated PNG to output_path.

        Args:
            prompt: Image generation prompt.
            output_path: Output PNG file path.
            model: Image generation model name.
            aspect_ratio: Aspect ratio (e.g. "16:9", "1:1", "4:3").
            image_size: Resolution tier ("1K", "2K", "4K").
            max_retries: Maximum retry attempts.
            reference_image_path: Optional reference image path used for style imitation.
            system_prompt: Optional system prompt; defaults to the built-in role.

        Returns:
            True if the image was generated and saved successfully.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build message content: multimodal if reference image provided
        if reference_image_path:
            ref_path = Path(reference_image_path)
            if not ref_path.exists():
                logger.warning(f"Reference image not found: {ref_path}, generating without it")
                message_content = prompt
            else:
                try:
                    compressed, mime = self._compress_reference_image(ref_path)
                    b64 = base64.b64encode(compressed).decode("utf-8")
                except Exception as e:
                    logger.warning(f"Failed to compress reference image ({e}), using raw bytes")
                    raw = ref_path.read_bytes()
                    b64 = base64.b64encode(raw).decode("utf-8")
                    suffix = ref_path.suffix.lstrip(".").lower()
                    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")
                message_content = [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]
        else:
            message_content = prompt

        # Build messages with system prompt
        sys = system_prompt if system_prompt is not None else self._IMAGE_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": message_content},
        ]

        for attempt in range(max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body={
                        "modalities": ["image", "text"],
                        "image_config": {
                            "aspect_ratio": aspect_ratio,
                            "image_size": image_size,
                        },
                        "reasoning": {
                            "effort": "high",
                            "exclude": True,
                        },
                    },
                )

                if not completion.choices:
                    logger.warning(f"Image generation returned no choices (attempt {attempt + 1})")
                    continue

                msg = completion.choices[0].message
                images = getattr(msg, "model_extra", {}).get("images", [])

                if not images:
                    logger.warning(f"No images in response (attempt {attempt + 1})")
                    continue

                data_url: str = images[0].get("image_url", {}).get("url", "")
                if not data_url.startswith("data:image/"):
                    logger.warning(f"Unexpected image URL format (attempt {attempt + 1})")
                    continue

                b64_data = data_url.split(",", 1)[1]
                raw_bytes = base64.b64decode(b64_data)

                output_path.write_bytes(raw_bytes)
                self._compress_saved_image(output_path)
                final_size = output_path.stat().st_size
                logger.info(f"Image saved: {output_path} ({final_size} bytes)")
                return True

            except Exception as e:
                logger.warning(f"Image generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    logger.error(f"Image generation failed after {max_retries + 1} attempts")
                    return False

        return False

