"""
Skills System - Centralized management of prompts, frameworks, and tools

This system provides unified management of:
- Prompts: agent prompts, code generation, paper writing, review, etc.
- Frameworks: deep-learning framework documentation and templates
- Tools: LaTeX compilation and other utilities

Usage:
    from skills import load_skill, FrameworkLoader

    # Load a prompt
    prompt = load_skill("agents/phd_student/literature_review.md")

    # Load a framework
    framework_loader = FrameworkLoader()
    doc, _ = framework_loader.find_documentation("camylanet")
"""

from .prompt_loader import (
    load_skill,
    list_all_skills,
    reload_skill,
    get_loader,
    SkillsLoader
)

from .frameworks import FrameworkLoader, framework_loader

__all__ = [
    'load_skill',
    'list_all_skills', 
    'reload_skill',
    'get_loader',
    'SkillsLoader',
    'FrameworkLoader',
    'framework_loader'
]

__version__ = '2.0.0'
