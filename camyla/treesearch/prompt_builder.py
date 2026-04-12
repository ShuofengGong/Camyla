"""
OpenHands Prompt Builder - Assembles prompts from Skills templates

This module centralizes prompt construction for OpenHands code generation,
using the Skills system to load templates and inject variables.

Usage:
    from camyla.treesearch.prompt_builder import OpenHandsPromptBuilder
    
    builder = OpenHandsPromptBuilder(cfg)
    prompt = builder.build_prompt(
        stage=2,
        mode="improve",
        dataset_info={...},
        exp_name="...",
        innovation_description="..."
    )
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger("camyla")

# Try to import skills.prompt_loader
try:
    from skills.prompt_loader import load_skill
    SKILLS_AVAILABLE = True
except ImportError:
    SKILLS_AVAILABLE = False
    logger.warning("skills.prompt_loader not available, using fallback mode")


class OpenHandsPromptBuilder:
    """OpenHands Prompt Builder
    
    Responsible for assembling complete prompts from Skills templates.
    Uses load_skill() to load templates and inject variables.
    """
    
    BASE_PATH = "code_generation/openhands"
    
    # Default paths (can be overridden by config)
    DEFAULT_PYTHON_PATH = "/opt/conda/envs/py310/bin/python"
    DEFAULT_PYTEST_PATH = "/opt/conda/envs/py310/bin/pytest"
    
    def __init__(self, cfg):
        """Initialize the prompt builder
        
        Args:
            cfg: Configuration object with agent settings
        """
        self.cfg = cfg
        oh = cfg.experiment.openhands
        self.python_path = getattr(oh, 'python_path', self.DEFAULT_PYTHON_PATH)
        self.pytest_path = getattr(oh, 'pytest_path', self.DEFAULT_PYTEST_PATH)
        
        logger.debug(f"📝 OpenHandsPromptBuilder initialized")
    
    def _load_template(self, template_path: str, **variables) -> str:
        """Load a template file and substitute variables
        
        Args:
            template_path: Path relative to BASE_PATH
            **variables: Variables to substitute
            
        Returns:
            Template content with variables substituted
        """
        if not SKILLS_AVAILABLE:
            logger.warning(f"Skills not available, returning empty string for {template_path}")
            return ""
        
        full_path = f"{self.BASE_PATH}/{template_path}"
        try:
            content = load_skill(full_path, **variables)
            logger.debug(f"✅ Loaded template: {full_path}")
            return content
        except FileNotFoundError as e:
            logger.warning(f"Template not found: {full_path}")
            return ""
        except Exception as e:
            logger.error(f"Error loading template {full_path}: {e}")
            return ""
    
    # =========================================================================
    # Component Builders
    # =========================================================================
    
    def build_dataset_config(self, dataset_info: Dict[str, Any]) -> str:
        """Build dataset configuration block"""
        return self._load_template(
            "components/dataset_config.md",
            dataset_id=dataset_info.get('dataset_id', 'UNKNOWN'),
            dataset_name=dataset_info.get('name', 'Unknown Dataset'),
            configuration=dataset_info.get('configuration', '3d_fullres'),
            target_structure=dataset_info.get('target_structure', 'medical structure'),
            modality=dataset_info.get('modality', 'medical imaging'),
            patch_size=dataset_info.get('patch_size', 'Not specified')
        )
    
    def build_main_guard(self) -> str:
        """Build main guard requirement block"""
        return self._load_template("components/main_guard.md")
    
    def build_unit_testing(self) -> str:
        """Build unit testing requirements block"""
        return self._load_template(
            "components/unit_testing.md",
            python_path=self.python_path
        )
    
    def build_exp_name(self, exp_name: str) -> str:
        """Build experiment name configuration block"""
        return self._load_template(
            "components/exp_name.md",
            exp_name=exp_name
        )
    
    def build_impl_guideline(self, timeout_duration: str = "6 hours") -> str:
        """Build implementation guideline block"""
        return self._load_template(
            "components/impl_guideline.md",
            timeout_duration=timeout_duration
        )
    
    def build_environment(self) -> str:
        """Build environment information block"""
        return self._load_template("components/environment.md")
    
    def build_execution_control_summary(self) -> str:
        """Build execution control summary for end of prompt"""
        return self._load_template(
            "components/execution_control_summary.md",
            python_path=self.python_path,
            pytest_path=self.pytest_path
        )
    
    # =========================================================================
    # Skills Builders (for OpenHands AgentContext)
    # =========================================================================
    
    def build_execution_control_skill(self) -> str:
        """Build execution control skill (for OpenHands AgentContext)"""
        return self._load_template(
            "skills/execution_control.md",
            python_path=self.python_path,
            pytest_path=self.pytest_path
        )
    
    def build_code_generation_guidelines_skill(self) -> str:
        """Build code generation guidelines skill (for OpenHands AgentContext)"""
        return self._load_template(
            "skills/code_generation_guidelines.md",
            python_path=self.python_path
        )
    
    # =========================================================================
    # Stage-Specific Builders
    # =========================================================================
    
    def build_stage1_baseline(self) -> str:
        """Build Stage 1 baseline implementation block"""
        return self._load_template("stages/stage1_baseline.md")
    
    def build_stage2_innovation(
        self,
        innovation_description: str,
        baseline_info: str
    ) -> str:
        """Build Stage 2 innovation implementation block"""
        return self._load_template(
            "stages/stage2_innovation.md",
            innovation_description=innovation_description,
            baseline_info=baseline_info
        )
    
    def build_stage3_ablation(
        self,
        component_name: str,
        removal_description: str
    ) -> str:
        """Build Stage 3 ablation study block"""
        return self._load_template(
            "stages/stage3_ablation.md",
            component_name=component_name,
            removal_description=removal_description
        )
    
    # =========================================================================
    # Mode Builders
    # =========================================================================
    
    def build_draft_mode(self) -> str:
        """Build draft mode instructions (new code)"""
        return self._load_template("modes/draft.md")
    
    def build_improve_mode(
        self,
        code_length: int = 0,
        code_lines: int = 0
    ) -> str:
        """Build improve mode instructions"""
        return self._load_template(
            "modes/improve.md",
            code_length=code_length,
            code_lines=code_lines
        )
    
    def build_debug_mode(
        self,
        error_message: str,
        code_length: int = 0,
        code_lines: int = 0,
        stage: int = 1
    ) -> str:
        """Build debug mode instructions"""
        # Stage-specific constraints
        if stage == 1:
            stage_specific_constraints = """- Add optimizations or advanced features
- Implement novel techniques
- Change from baseline approach"""
        else:
            stage_specific_constraints = """- Remove or skip the core innovation
- Change the fundamental innovation concept"""
        
        return self._load_template(
            "modes/debug.md",
            error_message=error_message,
            code_length=code_length,
            code_lines=code_lines,
            stage_specific_constraints=stage_specific_constraints
        )
    
    # =========================================================================
    # Format Helpers
    # =========================================================================
    
    def format_innovation_references(self, innovations: List[Dict[str, Any]]) -> str:
        """Format innovation references (legacy, kept for compatibility)"""
        lines = []
        for i, innov in enumerate(innovations, 1):
            lines.append(f"**Innovation {i}: {innov.get('Name', 'Unknown')}**")
            lines.append(f"- 📁 Reference File: `reference_docs/innovations/{innov.get('Reference File', 'unknown')}`")
            lines.append(f"- 📝 Description: {innov.get('Description', 'No description')}")
            if "Performance Gains" in innov:
                lines.append("- 📈 Performance Gains:")
                for gain in innov['Performance Gains']:
                    lines.append(f"  - {gain}")
            lines.append("")
        return "\n".join(lines)
    
    # =========================================================================
    # Main Prompt Builder
    # =========================================================================
    
    def build_prompt(
        self,
        stage: int,
        mode: str,
        dataset_info: Dict[str, Any],
        exp_name: str,
        existing_code: Optional[str] = None,
        error_feedback: Optional[str] = None,
        innovation_description: Optional[str] = None,
        baseline_info: Optional[str] = None,
        component_name: Optional[str] = None,
        removal_description: Optional[str] = None,
        param_name: Optional[str] = None,  # Deprecated, kept for compatibility
        param_description: Optional[str] = None,  # Deprecated, kept for compatibility
    ) -> str:
        """Build complete prompt for any stage and mode
        
        Args:
            stage: Stage number (1-3)
            mode: Mode ('draft', 'improve', 'debug')
            dataset_info: Dataset configuration dictionary
            exp_name: Experiment name
            existing_code: Existing code (for improve/debug modes)
            error_feedback: Error message (for debug mode)
            innovation_description: For Stage 2
            baseline_info: For Stage 2
            component_name: For Stage 3 ablation studies
            removal_description: For Stage 3 ablation studies
            param_name: Deprecated, use component_name instead
            param_description: Deprecated, use removal_description instead
            
        Returns:
            Complete prompt string
        """
        # Handle backward compatibility
        if param_name and not component_name:
            component_name = param_name
        if param_description and not removal_description:
            removal_description = param_description
        parts = []
        
        # Header
        parts.append("# Camyla Experiment Code Generation Task\n")
        
        # Mode-specific intro
        code_length = len(existing_code) if existing_code else 0
        code_lines = (existing_code.count('\n') + 1) if existing_code else 0
        
        if mode == "draft":
            parts.append(self.build_draft_mode())
        elif mode == "debug" and error_feedback:
            parts.append(self.build_debug_mode(
                error_message=error_feedback,
                code_length=code_length,
                code_lines=code_lines,
                stage=stage
            ))
        elif mode == "improve" and existing_code:
            parts.append(self.build_improve_mode(code_length, code_lines))
        
        # Stage-specific content
        if stage == 1:
            parts.append(self.build_stage1_baseline())
        elif stage == 2 and innovation_description:
            parts.append(self.build_stage2_innovation(
                innovation_description=innovation_description,
                baseline_info=baseline_info or ""
            ))
        elif stage == 3 and component_name:
            # Stage 3 is ablation studies
            parts.append(self.build_stage3_ablation(
                component_name=component_name,
                removal_description=removal_description or ""
            ))
        
        # Dataset configuration (CRITICAL)
        parts.append(self.build_dataset_config(dataset_info))
        
        # Experiment name
        if exp_name:
            parts.append(self.build_exp_name(exp_name))
        
        # Main guard requirement - MOVED TO SKILL
        # parts.append(self.build_main_guard())
        
        # Unit testing requirements - MOVED TO SKILL
        # parts.append(self.build_unit_testing())
        
        # Execution control summary - MOVED TO SKILL
        # parts.append(self.build_execution_control_summary())
        
        # Add a reference to the guidelines in the system prompt
        parts.append("## 📝 Guidelines\nPlease refer to the 'Code Generation Guidelines' in the System Prompt for:\n- File Operation Rules\n- Main Guard Requirements\n- Unit Testing Workflow\n- Execution Control Restrictions")
        
        # Join with double newlines
        return "\n\n".join(filter(None, parts))
    
    # =========================================================================
    # Legacy Support
    # =========================================================================
    
    def build_stage2_prompt(
        self,
        innovation_description: str,
        baseline_info: str,
        dataset_info: Dict[str, Any],
        exp_name: str,
        existing_code: Optional[str] = None,
    ) -> str:
        """Build complete Stage 2 (Innovation) prompt - legacy method"""
        return self.build_prompt(
            stage=2,
            mode="improve" if existing_code else "draft",
            dataset_info=dataset_info,
            exp_name=exp_name,
            existing_code=existing_code,
            innovation_description=innovation_description,
            baseline_info=baseline_info
        )
    
    def build_skill_content(self, framework_doc: Optional[str] = None) -> str:
        """Build framework skill content for OpenHands AgentContext"""
        parts = []
        
        if framework_doc:
            parts.append("## CamylaNet Framework Documentation\n")
            parts.append(framework_doc)
            parts.append("\n---\n")
        
        return "\n".join(parts)


# Convenience function for testing
def test_builder():
    """Test the prompt builder with sample data"""
    class MockConfig:
        class _OH:
            python_path = "/usr/bin/python3"
            pytest_path = "/usr/bin/pytest"
        class _Exp:
            openhands = _OH()
        experiment = _Exp()
    
    builder = OpenHandsPromptBuilder(MockConfig())
    
    # Test Stage 2 prompt
    prompt = builder.build_prompt(
        stage=2,
        mode="improve",
        dataset_info={
            "dataset_id": 27,
            "name": "Pancreas",
            "configuration": "3d_fullres",
            "target_structure": "pancreas and tumors",
            "modality": "CT",
            "patch_size": "[64, 128, 128]"
        },
        exp_name="test_experiment_001",
        existing_code="# existing code here...\ndef main():\n    pass",
        innovation_description="Innovation: Attention-Guided Decoder\nDescription: Add attention mechanisms...",
        baseline_info="Dataset: Pancreas\nTask: Tumor segmentation"
    )
    
    print("=" * 60)
    print("GENERATED PROMPT (Stage 2 - Improve):")
    print("=" * 60)
    print(prompt)  # Print FULL prompt
    print("=" * 60)
    
    # Check for absence of static content
    static_markers = [
        "Main Guard Requirement",
        "Unit Testing Requirements",
        "Execution Control Summary",
        "Wrap all executable code",
        "timeout --foreground"
    ]
    
    print("\n🔍 VERIFICATION:")
    for marker in static_markers:
        if marker in prompt:
            print(f"❌ Found unexpected static marker: '{marker}'")
        else:
            print(f"✅ Static marker NOT found: '{marker}'")
            
    print(f"\n[Total: {len(prompt)} characters]")
    
    return prompt


if __name__ == "__main__":
    test_builder()
