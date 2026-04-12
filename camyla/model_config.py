"""Unified entry point for LLM configuration.

Loads config.yaml and exposes three query APIs:

- get_endpoint(name)        → connection point (api_key + base_url + model + temperature)
- get_role(role)            → role (uses default_endpoint, overridden by llm_roles fields)
- get_api_key(name)         → non-LLM API key (s2 / ncbi)

Schema (see config_example.yaml):

    default_endpoint: my_openrouter
    llm_endpoints:
      my_openrouter:
        api_key: ""
        api_key_env: OPENROUTER_API_KEY
        base_url: "https://openrouter.ai/api/v1"
        model: "deepseek/deepseek-v3.2"
        temperature: 0.5
      ...
    llm_roles:
      log_summary: { temperature: 1.0 }
      ...
    api_keys:
      s2: { value: "", env: S2_API_KEY }
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


# ══════════════════════════════════════════════════════════════
#  Config loading
# ══════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _load_full_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = os.environ.get("QWBE_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file not found at {config_file}, using defaults")
        return {}
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def clear_cache():
    _load_full_config.cache_clear()


# ══════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════

def _resolve_api_key(ep_cfg: Dict[str, Any]) -> str:
    """Prefer ep_cfg.api_key; otherwise read the env var referenced by ep_cfg.api_key_env."""
    key = ep_cfg.get("api_key", "") or ""
    if key:
        return key
    env_var = ep_cfg.get("api_key_env")
    if env_var:
        return os.environ.get(env_var, "")
    return ""


def get_default_endpoint_name() -> str:
    cfg = _load_full_config()
    name = cfg.get("default_endpoint")
    if name:
        return name
    endpoints = cfg.get("llm_endpoints", {})
    if endpoints:
        return next(iter(endpoints))
    raise ValueError("No llm_endpoints configured and no default_endpoint set.")


def get_endpoint(name: Optional[str] = None) -> Dict[str, Any]:
    """Fetch an endpoint by name. When name is None, returns default_endpoint.

    The returned dict always contains: api_key, base_url, model, temperature, name.
    """
    cfg = _load_full_config()
    endpoints = cfg.get("llm_endpoints", {})

    if name is None:
        name = get_default_endpoint_name()

    if name not in endpoints:
        raise KeyError(
            f"Endpoint '{name}' not found in llm_endpoints. "
            f"Available: {list(endpoints.keys())}"
        )

    ep_cfg = dict(endpoints[name])
    resolved = {
        "name": name,
        "api_key": _resolve_api_key(ep_cfg),
        "base_url": ep_cfg.get("base_url", ""),
        "model": ep_cfg.get("model", ""),
        "temperature": ep_cfg.get("temperature", 0.5),
    }
    # Pass through other fields (e.g. max_tokens)
    for k, v in ep_cfg.items():
        if k not in resolved and k not in ("api_key_env",):
            resolved[k] = v
    return resolved


# ══════════════════════════════════════════════════════════════
#  Roles
# ══════════════════════════════════════════════════════════════

def get_role(role: str, group: Optional[str] = None) -> Dict[str, Any]:
    """Fetch the full config for a role: endpoint fields + model/temperature overridden by the role.

    Args:
        role:   Role name (a key under llm_roles), e.g. "log_summary".
        group:  Optional group (e.g. "paper_agent"); looks up llm_roles[group][role],
                falling back to llm_roles[group]["_default"] when the role is missing.

    Inside the role config:
        - endpoint: specifies which endpoint to use (omit = default_endpoint).
        - model / temperature / max_tokens / ...: override the matching fields on the endpoint.
    """
    cfg = _load_full_config()
    roles = cfg.get("llm_roles", {}) or {}

    role_cfg: Dict[str, Any] = {}
    if group is not None:
        group_cfg = roles.get(group, {}) or {}
        role_cfg = group_cfg.get(role) or group_cfg.get("_default") or {}
    else:
        role_cfg = roles.get(role, {}) or {}

    endpoint_name = role_cfg.get("endpoint")
    base = get_endpoint(endpoint_name)  # None → default

    merged = dict(base)
    for k, v in role_cfg.items():
        if k == "endpoint":
            continue
        merged[k] = v
    merged["role"] = role
    if group:
        merged["role_group"] = group
    return merged


# ══════════════════════════════════════════════════════════════
#  API Keys (non-LLM)
# ══════════════════════════════════════════════════════════════

def get_api_key(name: str) -> str:
    """api_keys.<name> supports two shapes:
       - Flat:       `s2: "xxx"`
       - Structured: `s2: { value: "xxx", env: S2_API_KEY }`
    """
    cfg = _load_full_config()
    entry = cfg.get("api_keys", {}).get(name)

    if entry is None:
        return os.environ.get(f"{name.upper()}_API_KEY", "")

    if isinstance(entry, str):
        return entry or os.environ.get(f"{name.upper()}_API_KEY", "")

    if isinstance(entry, dict):
        val = entry.get("value", "") or ""
        if val:
            return val
        env_var = entry.get("env")
        if env_var:
            return os.environ.get(env_var, "")
        return ""

    return ""


# ══════════════════════════════════════════════════════════════
#  Thin backward-compatibility shims (used during Stage 2 migration; can be removed afterwards)
# ══════════════════════════════════════════════════════════════

def get_model(role: str) -> Dict[str, Any]:
    r = get_role(role)
    return {"model": r.get("model", ""), "temperature": r.get("temperature", 0.5)}


def get_model_name(role: str) -> str:
    return get_role(role).get("model", "")


def get_model_temperature(role: str) -> float:
    return get_role(role).get("temperature", 0.5)


def load_llm_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Deprecated: returns llm_roles (historically named llm_models)."""
    return _load_full_config(config_path).get("llm_roles", {}) or {}
