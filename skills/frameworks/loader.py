"""
Skills Framework loader - manages documentation and templates for deep-learning frameworks.
"""
import os
from pathlib import Path
from typing import Tuple, Optional


class FrameworkLoader:
    """Manages documentation and template configuration."""

    def __init__(self, base_dir: str = None):
        """
        Initialize the configuration.

        Args:
            base_dir: Camyla root directory. Auto-detected when None.
        """
        if base_dir is None:
            # Auto-detect the frameworks directory (directory containing loader.py)
            current_dir = Path(__file__).parent
            base_dir = str(current_dir)

        self.base_dir = Path(base_dir)
        self.frameworks_dir = self.base_dir

    def find_framework_file(self, extension: str, framework: str = "camylanet") -> Tuple[Optional[Path], Optional[str]]:
        """
        Find a framework file.

        Args:
            extension: file extension (e.g. .md, .py)
            framework: framework name (e.g. camylanet)

        Returns:
            (file path, source description) or (None, None)
        """
        # Make sure the extension starts with a dot
        if not extension.startswith('.'):
            extension = f".{extension}"

        # Strategy 1: look for a standard filename in frameworks/{framework}/
        framework_subdir = self.frameworks_dir / framework
        if framework_subdir.exists() and framework_subdir.is_dir():
            # Map extension to its standard filename
            if extension == ".py":
                standard_file = framework_subdir / "template.py"
                if standard_file.exists():
                    return standard_file, f"{framework} framework code template"
            elif extension == ".md":
                standard_file = framework_subdir / "documentation.md"
                if standard_file.exists():
                    return standard_file, f"{framework} framework documentation"

            # Fall back to any file matching the extension
            for file in framework_subdir.glob(f"*{extension}"):
                return file, f"{framework} framework file"

        # Strategy 2: look for {framework}{extension} directly under frameworks/
        framework_filename = f"{framework}{extension}"
        framework_file = self.frameworks_dir / framework_filename
        if framework_file.exists():
            return framework_file, "generic framework documentation"

        return None, None

    def find_code_template(self, framework: str = "camylanet") -> Tuple[Optional[Path], Optional[str]]:
        """
        Find the code template file.

        Args:
            framework: framework name

        Returns:
            (file path, source description) or (None, None)
        """
        return self.find_framework_file(".py", framework)

    def find_documentation(self, framework: str = "camylanet") -> Tuple[Optional[Path], Optional[str]]:
        """
        Find the documentation file.

        Args:
            framework: framework name

        Returns:
            (file path, source description) or (None, None)
        """
        return self.find_framework_file(".md", framework)

    def get_available_frameworks(self) -> list:
        """Return the list of available frameworks."""
        if not self.frameworks_dir.exists():
            return []

        frameworks = set()

        # Infer frameworks from filenames
        for file in self.frameworks_dir.glob("*.py"):
            frameworks.add(file.stem)
        for file in self.frameworks_dir.glob("*.md"):
            frameworks.add(file.stem)

        return sorted(list(frameworks))

    def get_framework_files(self, framework: str = "camylanet") -> dict:
        """
        Get all files for the given framework.

        Returns:
            {"docs": [list of doc files], "code": [list of code files]}
        """
        if not self.frameworks_dir.exists():
            return {"docs": [], "code": []}

        docs = []
        code = []

        # Look for framework.md and framework.py
        framework_md = self.frameworks_dir / f"{framework}.md"
        framework_py = self.frameworks_dir / f"{framework}.py"

        if framework_md.exists():
            docs.append(framework_md)
        if framework_py.exists():
            code.append(framework_py)

        return {"docs": docs, "code": code}


# Global configuration instance
framework_loader = FrameworkLoader()


def find_common_file(extension: str, framework: str = "camylanet") -> Tuple[Optional[str], Optional[str]]:
    """
    Convenience wrapper to find a common file.

    Args:
        extension: file extension (e.g. py, md)
        framework: framework name

    Returns:
        (file path string, source description) or (None, None)
    """
    file_path, source = framework_loader.find_framework_file(extension, framework)

    if file_path:
        return str(file_path), source
    return None, None


if __name__ == "__main__":
    # Test configuration
    config = FrameworkLoader()
    print(f"Base directory: {config.base_dir}")
    print(f"Available frameworks: {config.get_available_frameworks()}")

    # Test file lookup
    code_path, code_source = config.find_code_template()
    doc_path, doc_source = config.find_documentation()

    print(f"Code template: {code_path} from {code_source}")
    print(f"Documentation: {doc_path} from {doc_source}")
