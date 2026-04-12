import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from string import Template

from camyla.paper_agent.func.config_resolver import load_qwbe_config, resolve_config_path
from camyla.paper_agent.func.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

_DEFAULT_AGENT_CONFIG = {
    "model": "deepseek/deepseek-v3.2",
    "temperature": 0.7
}


def _find_qwbe_config() -> Optional[Path]:
    """Locate the active config file for Paper Agent."""
    return resolve_config_path(search_from=__file__)


class BaseAgent:
    """
    Base class for all agents in the Camyla system.
    Handles LLM interaction, skill (prompt) loading, and basic state management.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        skill_root: Optional[Path] = None,
    ):
        """
        Initialize the BaseAgent.

        Args:
            model: The OpenRouter model identifier. If None, loads from config.
            temperature: LLM generation temperature. If None, loads from config.
            skill_root: Root directory for skills/prompts. Defaults to ../skill relative to this file.
        """
        config = self._load_config()
        
        self.client = OpenRouterClient()
        self.model = model if model is not None else config.get("model", _DEFAULT_AGENT_CONFIG["model"])
        self.temperature = temperature if temperature is not None else config.get("temperature", _DEFAULT_AGENT_CONFIG["temperature"])

        if skill_root:
            self.skill_root = skill_root
        else:
            self.skill_root = Path(__file__).parent.parent / "skill"

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration for this agent class.
        
        Priority:
            1. config.yaml -> llm_roles.paper_agent.{ClassName}
            2. config.yaml -> llm_roles.paper_agent._default
            3. agent_models.yaml -> {ClassName}
            4. agent_models.yaml -> default
            5. Hardcoded defaults
        
        Returns:
            Dictionary with model and temperature settings for this agent.
        """
        class_name = self.__class__.__name__

        # 1. Try config.yaml (primary source)
        qwbe_config = self._load_from_qwbe_config(class_name)
        if qwbe_config:
            return qwbe_config

        # 2. Fallback to agent_models.yaml
        local_config = self._load_from_agent_models_yaml(class_name)
        if local_config:
            return local_config

        return _DEFAULT_AGENT_CONFIG

    def _load_from_qwbe_config(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Try loading agent config from config.yaml llm_roles.paper_agent section.

        Uses model_config.get_role which handles _default fallback automatically.
        Returns a dict with model/temperature (and endpoint merged via default_endpoint).
        """
        try:
            from camyla.model_config import get_role, _load_full_config
            full = _load_full_config()
            paper_agent_cfg = (full.get("llm_roles", {}) or {}).get("paper_agent")
            if not paper_agent_cfg:
                return None

            role = get_role(class_name, group="paper_agent")
            source = "specific" if class_name in paper_agent_cfg else "_default"
            logger.info(f"[config.yaml] Loaded paper_agent.{class_name} ({source}): model={role.get('model')}")
            return {"model": role.get("model"), "temperature": role.get("temperature", 0.6)}
        except Exception as e:
            logger.warning(f"Error reading config.yaml llm_roles.paper_agent: {e}")
            return None

    def _load_from_agent_models_yaml(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Fallback: load agent config from local agent_models.yaml."""
        config_path = Path(__file__).parent / "agent_models.yaml"
        try:
            if not config_path.exists():
                return None
            with open(config_path, "r", encoding="utf-8") as f:
                all_configs = yaml.safe_load(f)
            if not all_configs:
                return None

            agent_config = all_configs.get(class_name)
            if agent_config:
                logger.info(f"[agent_models.yaml] Loaded config for {class_name}: model={agent_config.get('model')}")
                return agent_config

            default_cfg = all_configs.get("default")
            if default_cfg:
                logger.info(f"[agent_models.yaml] No specific config for {class_name}, using default")
                return default_cfg

            return None
        except Exception as e:
            logger.warning(f"Error reading agent_models.yaml: {e}")
            return None


    def load_skill(self, skill_path: str, **kwargs) -> str:
        """
        Load and format a skill (prompt template) from the skill directory.

        Args:
            skill_path: Relative path to the skill file (e.g., "common/summarize.md").
            **kwargs: Variables to substitute into the template.

        Returns:
            The formatted prompt string.
        """
        full_path = self.skill_root / skill_path
        if not full_path.exists():
            raise FileNotFoundError(f"Skill file not found: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        # Process ${include:path} directives to include fragment files
        template_content = self._process_includes(template_content)

        # Use string.Template for safe substitution
        # Template is safer for user-defined prompts that might have curly braces (like LaTeX).
        try:
            template = Template(template_content)
            return template.safe_substitute(**kwargs)
        except Exception as e:
            logger.error(f"Error formatting skill {skill_path}: {e}")
            raise

    def _process_includes(self, content: str) -> str:
        """
        Process ${include:path} directives, replacing them with file contents.
        
        This enables shared fragments (like writing_style.md) to be centrally 
        managed and included in multiple skill files.
        
        Args:
            content: Original content containing include directives
            
        Returns:
            Content with all includes expanded
        """
        import re
        pattern = r'\$\{include:([^}]+)\}'
        
        def replace_include(match):
            include_path = match.group(1).strip()
            full_path = self.skill_root / include_path
            
            if not full_path.exists():
                logger.warning(f"Include file not found: {full_path}")
                return f"<!-- Include not found: {include_path} -->"
            
            with open(full_path, "r", encoding="utf-8") as f:
                included_content = f.read()
            
            logger.debug(f"Included fragment: {include_path}")
            return included_content
        
        return re.sub(pattern, replace_include, content)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat request to the LLM.

        Args:
            messages: List of message dictionaries.
            tools: Optional list of tools.
            tool_choice: Optional tool choice.
            max_tokens: Optional maximum tokens to generate.

        Returns:
            The content of the response.
        """
        kwargs = {}
        if max_tokens is not None:
            kwargs['max_tokens'] = max_tokens
            
        response = self.client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )

        if response.content is None:
            logger.warning("Received empty content from LLM.")
            return ""

        return response.content

    def run(self, *args, **kwargs):
        """
        Main execution method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the run method.")
