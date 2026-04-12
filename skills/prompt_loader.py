"""
Skills Prompt Loader - a simple Markdown prompt loader.

Usage:
    from skills.prompt_loader import load_skill

    # Basic usage
    prompt = load_skill("agents/phd_student/literature_review.md")

    # With variables
    prompt = load_skill(
        "agents/phd_student/literature_review.md",
        reviewed_papers="paper1, paper2"
    )
"""

from pathlib import Path
from typing import Optional, Dict, List
import os


class SkillsLoader:
    """Simple Skills loader that reads markdown files directly."""

    def __init__(self, skills_dir: Optional[str] = None):
        """Initialize the loader.

        Args:
            skills_dir: path to the skills directory (defaults to this file's directory).
        """
        if skills_dir is None:
            # Default to the directory containing this file
            skills_dir = Path(__file__).parent
        self.skills_dir = Path(skills_dir)
        self._cache: Dict[str, str] = {}

    def load(self, skill_path: str, use_cache: bool = True) -> str:
        """Load the content of a skill file.

        Args:
            skill_path: path relative to the skills directory, e.g. "agents/phd_student/literature_review.md".
            use_cache: whether to use the in-memory cache.

        Returns:
            The markdown file content.

        Raises:
            FileNotFoundError: if the file does not exist.

        Example:
            loader = SkillsLoader()
            prompt = loader.load("agents/phd_student/literature_review.md")
        """
        # Check cache
        if use_cache and skill_path in self._cache:
            return self._cache[skill_path]

        # Build the full path
        full_path = self.skills_dir / skill_path

        if not full_path.exists():
            raise FileNotFoundError(
                f"Skill file not found: {full_path}\n"
                f"Skills directory: {self.skills_dir}\n"
                f"Looking for: {skill_path}"
            )

        # Read the file
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Cache the content
        if use_cache:
            self._cache[skill_path] = content

        return content

    def load_with_variables(self, skill_path: str, **variables) -> str:
        """Load a skill and substitute its variables.

        Two variable formats are supported:
        1. {variable_name} - Python format
        2. {{variable_name}} - double-brace format (avoids LaTeX conflicts)

        Args:
            skill_path: skill file path
            **variables: variables to substitute

        Returns:
            The content with variables substituted.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if a required variable is missing.

        Example:
            loader = SkillsLoader()
            prompt = loader.load_with_variables(
                "agents/phd_student/literature_review.md",
                reviewed_papers="paper1, paper2"
            )
        """
        content = self.load(skill_path)

        # Substitute variables
        try:
            # First handle the double-brace form {{var}}
            for key, value in variables.items():
                content = content.replace(f"{{{{{key}}}}}", str(value))

            # Then handle the single-brace form {var}
            # Use safe_substitute to avoid KeyError
            import string
            template = string.Template(content)
            # Convert {var} to $var
            content_with_dollar = content
            for key in variables.keys():
                content_with_dollar = content_with_dollar.replace(f"{{{key}}}", f"${key}")

            template = string.Template(content_with_dollar)
            content = template.safe_substitute(**variables)

        except Exception as e:
            raise ValueError(f"Error replacing variables in {skill_path}: {e}")

        return content

    def list_skills(self, category: Optional[str] = None) -> List[str]:
        """List all available skill files.

        Args:
            category: optional category (e.g. "agents", "code_generation").

        Returns:
            A list of skill file paths.

        Example:
            loader = SkillsLoader()
            all_skills = loader.list_skills()
            agent_skills = loader.list_skills("agents")
        """
        if category:
            search_dir = self.skills_dir / category
        else:
            search_dir = self.skills_dir

        if not search_dir.exists():
            return []

        # Recursively find all .md files (excluding README.md)
        md_files = []
        for md_file in search_dir.rglob("*.md"):
            if md_file.name != "README.md":
                # Return the path relative to the skills directory
                rel_path = md_file.relative_to(self.skills_dir)
                md_files.append(str(rel_path).replace(os.sep, '/'))  # Normalize to '/' separator

        return sorted(md_files)

    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()

    def reload(self, skill_path: str) -> str:
        """Reload a skill, ignoring the cache.

        Args:
            skill_path: skill file path

        Returns:
            The file content.
        """
        # Drop from cache
        if skill_path in self._cache:
            del self._cache[skill_path]

        # Reload
        return self.load(skill_path, use_cache=True)


# Global singleton
_global_loader: Optional[SkillsLoader] = None


def get_loader() -> SkillsLoader:
    """Return the global SkillsLoader instance.

    Returns:
        The global SkillsLoader instance.
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = SkillsLoader()
    return _global_loader


def load_skill(skill_path: str, **variables) -> str:
    """Convenience function to load a skill.

    This is the most commonly used entry point; import it directly.

    Args:
        skill_path: skill file path (relative to the skills directory).
        **variables: optional variables to substitute into the prompt.

    Returns:
        The loaded and processed prompt content.

    Example:
        from skills.prompt_loader import load_skill

        # Basic usage
        prompt = load_skill("agents/phd_student/literature_review.md")

        # With variables
        prompt = load_skill(
            "agents/phd_student/literature_review.md",
            reviewed_papers="paper1, paper2"
        )
    """
    loader = get_loader()
    if variables:
        return loader.load_with_variables(skill_path, **variables)
    return loader.load(skill_path)


def list_all_skills(category: Optional[str] = None) -> List[str]:
    """List all available skills.

    Args:
        category: optional category.

    Returns:
        A list of skill file paths.

    Example:
        from skills.prompt_loader import list_all_skills

        # List every skill
        all_skills = list_all_skills()

        # List a specific category
        agent_skills = list_all_skills("agents")
    """
    loader = get_loader()
    return loader.list_skills(category)


def reload_skill(skill_path: str) -> str:
    """Reload a skill (used for hot reload during development).

    Args:
        skill_path: skill file path.

    Returns:
        The file content.

    Example:
        from skills.prompt_loader import reload_skill

        # After editing a prompt file, reload it
        prompt = reload_skill("agents/phd_student/literature_review.md")
    """
    loader = get_loader()
    return loader.reload(skill_path)


# Convenient aliases
load = load_skill
list_skills = list_all_skills

