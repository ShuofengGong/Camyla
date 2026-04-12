import logging
import os
import re
from datetime import datetime

ENABLE_QUERY_LOGGING = True

_query_log_dir: str | None = None

logger = logging.getLogger(__name__)


def set_query_log_dir(path: str):
    """Set the directory for query logs. Call this at experiment startup."""
    global _query_log_dir
    _query_log_dir = path
    os.makedirs(path, exist_ok=True)
    logger.info(f"Query log directory set to: {path}")


def clean_unicode_text_global(text: str) -> str:
    """
    Global Unicode text cleanup function, handling surrogate pairs and encoding issues.

    Args:
        text: raw text

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    try:
        # Step 1: use encode/decode to drop invalid characters
        cleaned_text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

        # Step 2: first replace common math symbols, then strip surrogate-pair characters.
        # Use correct Unicode code points and surrogate-pair forms.
        math_symbol_replacements = {
            # Correct Unicode code points
            '\U0001d4b6': 'α',  # 𝒶 mathematical italic alpha
            '\U0001d4b7': 'β',  # 𝒷 mathematical italic beta
            '\U0001d4b8': 'γ',  # 𝒸 mathematical italic gamma
            '\U0001d4b9': 'δ',  # 𝒹 mathematical italic delta
            '\U0001d4ba': 'ε',  # 𝒺 mathematical italic epsilon
            '\U0001d4bb': 'ζ',  # 𝒻 mathematical italic zeta
            '\U0001d4bc': 'η',  # 𝒼 mathematical italic eta
            '\U0001d4bd': 'θ',  # 𝒽 mathematical italic theta
            '\U0001d4be': 'ι',  # 𝒾 mathematical italic iota
            '\U0001d4bf': 'κ',  # 𝒿 mathematical italic kappa
            '\U0001d4c0': 'λ',  # 𝓀 mathematical italic lambda
            '\U0001d4c1': 'μ',  # 𝓁 mathematical italic mu
            '\U0001d4c2': 'ν',  # 𝓂 mathematical italic nu
            '\U0001d4c3': 'ξ',  # 𝓃 mathematical italic xi
            '\U0001d4c4': 'ο',  # 𝓄 mathematical italic omicron
            '\U0001d4c5': 'π',  # 𝓅 mathematical italic pi
            '\U0001d4c6': 'ρ',  # 𝓆 mathematical italic rho
            '\U0001d4c7': 'σ',  # 𝓇 mathematical italic sigma
            '\U0001d4c8': 'τ',  # 𝓈 mathematical italic tau
            '\U0001d4c9': 'υ',  # 𝓉 mathematical italic upsilon
            '\U0001d4ca': 'φ',  # 𝓊 mathematical italic phi
            '\U0001d4cb': 'χ',  # 𝓋 mathematical italic chi
            '\U0001d4cc': 'ψ',  # 𝓌 mathematical italic psi
            '\U0001d4cd': 'ω',  # 𝓍 mathematical italic omega
        }

        # Apply math-symbol replacements
        for unicode_char, replacement in math_symbol_replacements.items():
            cleaned_text = cleaned_text.replace(unicode_char, replacement)

        # Handle possible surrogate-pair forms (useful for corrupted PDF text)
        surrogate_pairs = [
            (chr(0xd835) + chr(0xdcb6), 'α'),  # surrogate-pair alpha
            (chr(0xd835) + chr(0xdcb7), 'β'),  # surrogate-pair beta
            (chr(0xd835) + chr(0xdcb8), 'γ'),  # surrogate-pair gamma
            (chr(0xd835) + chr(0xdcb9), 'δ'),  # surrogate-pair delta
            (chr(0xd835) + chr(0xdcba), 'ε'),  # surrogate-pair epsilon
            (chr(0xd835) + chr(0xdcbb), 'ζ'),  # surrogate-pair zeta
            (chr(0xd835) + chr(0xdcbc), 'η'),  # surrogate-pair eta
            (chr(0xd835) + chr(0xdcbd), 'θ'),  # surrogate-pair theta
            (chr(0xd835) + chr(0xdcbe), 'ι'),  # surrogate-pair iota
            (chr(0xd835) + chr(0xdcbf), 'κ'),  # surrogate-pair kappa
            (chr(0xd835) + chr(0xdcc0), 'λ'),  # surrogate-pair lambda
            (chr(0xd835) + chr(0xdcc1), 'μ'),  # surrogate-pair mu
            (chr(0xd835) + chr(0xdcc2), 'ν'),  # surrogate-pair nu
            (chr(0xd835) + chr(0xdcc3), 'ξ'),  # surrogate-pair xi
            (chr(0xd835) + chr(0xdcc4), 'ο'),  # surrogate-pair omicron
            (chr(0xd835) + chr(0xdcc5), 'π'),  # surrogate-pair pi
            (chr(0xd835) + chr(0xdcc6), 'ρ'),  # surrogate-pair rho
            (chr(0xd835) + chr(0xdcc7), 'σ'),  # surrogate-pair sigma
            (chr(0xd835) + chr(0xdcc8), 'τ'),  # surrogate-pair tau
            (chr(0xd835) + chr(0xdcc9), 'υ'),  # surrogate-pair upsilon
            (chr(0xd835) + chr(0xdcca), 'φ'),  # surrogate-pair phi
            (chr(0xd835) + chr(0xdccb), 'χ'),  # surrogate-pair chi
            (chr(0xd835) + chr(0xdccc), 'ψ'),  # surrogate-pair psi
            (chr(0xd835) + chr(0xdccd), 'ω'),  # surrogate-pair omega
        ]

        # Apply surrogate-pair replacements (inside try/catch because encoding errors are possible)
        for surrogate_pair, replacement in surrogate_pairs:
            try:
                cleaned_text = cleaned_text.replace(surrogate_pair, replacement)
            except UnicodeEncodeError:
                # Skip surrogate pairs that cannot be handled
                continue

        # Step 3: strip any remaining surrogate characters (\ud800-\udfff range)
        import re
        cleaned_text = re.sub(r'[\ud800-\udfff]', '', cleaned_text)

        # Step 4: final sweep — drop any remaining non-printable characters
        cleaned_text = ''.join(char for char in cleaned_text if char.isprintable() or char.isspace())

        return cleaned_text

    except Exception as e:
        logger.warning(f"Unicode text cleanup failed: {e}")
        # As a last resort, fall back to ASCII-safe content
        try:
            return text.encode('ascii', errors='ignore').decode('ascii')
        except Exception:
            return "[Text encoding error - content removed]"

# ============================================================================
# Token counting utilities
# ============================================================================

def estimate_token_count(text: str) -> int:
    """
    Roughly estimate the number of tokens in a piece of text.
    Uses a simple heuristic: roughly 4 characters per token.
    For more precise counts, use the tiktoken library.
    """
    if not text:
        return 0

    # Rough estimate: ~4 chars/token for English, ~1.5 for Chinese.
    # Using the conservative ~3 chars/token here.
    return len(text) // 3

def format_token_count(count: int) -> str:
    """Format a token count for display."""
    if count < 1000:
        return f"{count}"
    elif count < 1000000:
        return f"{count/1000:.1f}K"
    else:
        return f"{count/1000000:.1f}M"
    
    
def extract_prompt(text, word):
    """Extract content delimited by a specific tag from text (kept for other features)."""
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    extracted_code = "\n".join(code_blocks).strip()
    return extracted_code

def save_query_log(prompt, system_prompt, output, model_str, func_spec=None, req_time=0, in_tokens=0, out_tokens=0):
    """Save the query_model call log to a markdown file."""
    # Check whether logging is enabled
    if not ENABLE_QUERY_LOGGING:
        return

    try:
        if _query_log_dir:
            logs_dir = _query_log_dir
        else:
            logs_dir = os.path.join(os.environ.get("CAMYLA_ROOT", "."), "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Build a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # millisecond precision
        filename = f"query_log_{timestamp}.md"
        filepath = os.path.join(logs_dir, filename)

        # Safely handle potentially long text
        def safe_truncate(text, max_length=1500000):
            """Safely truncate text to avoid oversized files."""
            if not text:
                return "No content"
            text_str = str(text)
            if len(text_str) > max_length:
                return text_str[:max_length] + f"\n\n... [TRUNCATED - Original length: {len(text_str)} characters]"
            return text_str

        # Build function spec info
        func_info = "None"
        if func_spec:
            func_info = f"{func_spec.name}"
            if hasattr(func_spec, 'description'):
                func_info += f" - {func_spec.description[:100]}..."

        # Build the markdown body
        md_content = f"""# LLM Query Log

**Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model:** {model_str}
**Request Time:** {req_time:.2f}s
**Input Tokens:** {in_tokens:,}
**Output Tokens:** {out_tokens:,}
**Total Tokens:** {in_tokens + out_tokens:,}
**Function Spec:** {func_info}

## System Prompt

```
{safe_truncate(system_prompt)}
```

## User Prompt

```
{safe_truncate(prompt)}
```

## Model Response

```
{safe_truncate(output)}
```

## Response Type

**Type:** {type(output).__name__}
**Is Dict:** {isinstance(output, dict)}
**Is Function Call:** {isinstance(output, dict) and 'function_name' in str(output)}

---
*Generated by Camyla Innovation Generator at {datetime.now().isoformat()}*
"""

        # Write the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)

        # Report file size
        file_size = os.path.getsize(filepath)
        logger.info(f"Query log saved: {filename} ({file_size:,} bytes)")

    except Exception as e:
        logger.error(f"Failed to save query log: {e}")
        # Do not raise — logging failure must not break the main flow.