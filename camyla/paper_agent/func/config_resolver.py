import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH_ENV_VAR = "CAMYLA_CONFIG"
DEFAULT_CONFIG_FILENAMES = (
    "config.yaml",
    "config1.yaml",
    "config2.yaml",
    "config3.yaml",
)

PathLike = Union[str, Path]


def _normalize_path(path_value: Optional[PathLike]) -> Optional[Path]:
    if not path_value:
        return None
    return Path(path_value).expanduser().resolve()


def _iter_root_candidates(root: Path) -> Iterable[Tuple[str, Path]]:
    if root.is_file():
        yield "root_file", root
        return

    for filename in DEFAULT_CONFIG_FILENAMES:
        yield f"root/{filename}", root / filename


def resolve_config_path(
    explicit_config_path: Optional[PathLike] = None,
    experiment_dir: Optional[PathLike] = None,
    search_from: Optional[PathLike] = None,
) -> Optional[Path]:
    """Resolve config from explicit path, env, experiment dir, then fallback search."""
    candidates = []

    if explicit_config_path:
        candidates.append(("explicit", _normalize_path(explicit_config_path)))

    env_config = os.environ.get(CONFIG_PATH_ENV_VAR)
    if env_config:
        candidates.append(("env", _normalize_path(env_config)))

    if experiment_dir:
        exp_dir = _normalize_path(experiment_dir)
        if exp_dir:
            candidates.append(("experiment/config.yaml", exp_dir / "config.yaml"))

    root_env = os.environ.get("CAMYLA_ROOT")
    if root_env:
        root_path = _normalize_path(root_env)
        if root_path:
            candidates.extend(_iter_root_candidates(root_path))

    current = _normalize_path(search_from) if search_from else Path(__file__).resolve()
    if current and current.is_file():
        current = current.parent
    for _ in range(8):
        if current is None:
            break
        for filename in DEFAULT_CONFIG_FILENAMES:
            candidates.append((f"upward/{filename}", current / filename))
        if current.parent == current:
            break
        current = current.parent

    seen = set()
    for source, candidate in candidates:
        if candidate is None:
            continue
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            logger.info(f"[config] Using {source}: {candidate}")
            return candidate

    return None


def load_qwbe_config(
    explicit_config_path: Optional[PathLike] = None,
    experiment_dir: Optional[PathLike] = None,
    search_from: Optional[PathLike] = None,
) -> Dict[str, Any]:
    config_path = resolve_config_path(
        explicit_config_path=explicit_config_path,
        experiment_dir=experiment_dir,
        search_from=search_from,
    )
    if not config_path:
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


def set_runtime_config_path(
    explicit_config_path: Optional[PathLike] = None,
    experiment_dir: Optional[PathLike] = None,
    search_from: Optional[PathLike] = None,
) -> Optional[Path]:
    config_path = resolve_config_path(
        explicit_config_path=explicit_config_path,
        experiment_dir=experiment_dir,
        search_from=search_from,
    )
    if config_path:
        os.environ[CONFIG_PATH_ENV_VAR] = str(config_path)
    return config_path
