import logging
import time
import json_repair
from pathlib import Path
from typing import Dict, List, Any, Optional

from camyla.paper_agent.agents.base_agent import BaseAgent
from camyla.paper_agent.func.config_resolver import load_qwbe_config, resolve_config_path
from camyla.paper_agent.func.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


def _find_qwbe_config() -> Optional[Path]:
    """Locate the active config file for plot generation."""
    return resolve_config_path(search_from=__file__)


class PlotAgent(BaseAgent):
    """
    Agent specialized in generating academic paper figures.
    Supports two types:
    1. Method Diagrams - AI-generated architecture diagrams via Gemini image model
    2. Result Plots - matplotlib code executed via OpenHands agent
    """
    
    def __init__(self, output_figure_dir: Path, experiment_dir: Optional[Path] = None, **kwargs):
        """
        Initialize Plot Agent.
        
        Args:
            output_figure_dir: Figure output root directory (typically outputs/figure/)
            experiment_dir: Optional experiment directory for accessing model_results
            **kwargs: Other parameters passed to BaseAgent
        """
        super().__init__(**kwargs)
        self.figure_dir = Path(output_figure_dir)
        self.method_dir = self.figure_dir / "method"
        self.result_dir = self.figure_dir / "result"
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        
        self.method_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        self.figure_metadata = []
        
        self.image_client = OpenRouterClient()
        self.image_config = self._load_image_config()
        
        logger.info(f"PlotAgent initialized. Figure directory: {self.figure_dir}")
    
    @staticmethod
    def _load_image_config() -> Dict[str, Any]:
        default = {"model": "google/gemini-3.1-flash-image-preview", "aspect_ratio": "16:9", "image_size": "2K"}
        try:
            from camyla.model_config import get_role
            role = get_role("image_generator", group="paper_writing")
            return {
                "model": role.get("model", default["model"]),
                "aspect_ratio": role.get("aspect_ratio", default["aspect_ratio"]),
                "image_size": role.get("image_size", default["image_size"]),
            }
        except Exception:
            return default
    
    def run(
        self, 
        research_idea: str,
        experimental_results: str,
        ablation_results: str = "",
        plot_plan: str = "",
    ) -> Dict[str, Any]:
        """
        Phase 1: generate **result** figures only (seg comparison + LLM plots).
        Method diagrams are generated later via ``generate_method_figures()``
        after the paper text is available.
        
        Returns:
            Dictionary containing information about generated figures
        """
        logger.info("=" * 60)
        logger.info("PlotAgent: Generating result figures (phase 1)...")
        logger.info("=" * 60)

        self._cleanup_result_artifacts(clear_workspace=True)
        
        # 1. Generate segmentation comparison visualization (deterministic)
        seg_figures = self._generate_segmentation_visualization()
        logger.info(f"  ✓ Generated {len(seg_figures)} segmentation visualization figures")
        
        # 2. Collect per-case data for richer analysis plots
        per_case_context = self._collect_per_case_context()
        
        # 3. Generate and execute result analysis plots (LLM-generated)
        result_figures = self._generate_result_plots(
            research_idea, 
            experimental_results, 
            ablation_results,
            plot_plan,
            per_case_data=per_case_context
        )
        logger.info(f"  ✓ Generated {len(result_figures)} result plots")
        
        all_figures = seg_figures + result_figures
        self.figure_metadata = all_figures
        
        return self._save_and_return(all_figures)

    def generate_method_figures(
        self,
        methods_latex: str,
        plot_plan: str = "",
    ) -> Dict[str, Any]:
        """
        Phase 2: generate **method diagrams** using the full Methods section
        LaTeX produced by PaperWritingAgent.  Call this *after* ``run()``.

        Args:
            methods_latex: Complete LaTeX of the Methods / Methodology section.
            plot_plan: Optional analysis report / plot plan for guidance.

        Returns:
            Dictionary with method figure metadata (same schema as ``run()``).
        """
        logger.info("=" * 60)
        logger.info("PlotAgent: Generating method figures (phase 2)...")
        logger.info("=" * 60)

        method_figures = self._generate_method_diagrams(
            methods_latex=methods_latex,
            plot_plan=plot_plan,
        )
        logger.info(f"  ✓ Generated {len(method_figures)} method diagrams")

        # Merge with any previously stored result figures
        all_figures = method_figures + self.figure_metadata
        self.figure_metadata = all_figures

        return self._save_and_return(all_figures)

    # ------------------------------------------------------------------

    _MAX_FIGURE_BYTES = 2 * 1024 * 1024  # 2 MB per figure
    _MIN_RESULT_PNG_BYTES = 12 * 1024
    _PLACEHOLDER_MARKERS = (
        "todo",
        "fixme",
        "placeholder",
        "add your data here",
        "add plotting code here",
    )
    _MAX_RESULT_REPAIR_ROUNDS = 2

    def _save_and_return(self, all_figures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Persist figure summary / LaTeX snippets and return the result dict."""
        self._enforce_figure_size_limit(all_figures)

        visible_figures = self._visible_figures(all_figures)
        figure_summary = self._generate_figure_summary(visible_figures)
        latex_figures = self._generate_latex_figures_file(visible_figures)

        latex_file_path = self.figure_dir / "latex_figures.tex"
        latex_file_path.write_text(latex_figures, encoding='utf-8')

        summary_path = self.figure_dir / "figure_summary.md"
        summary_path.write_text(figure_summary, encoding='utf-8')

        method_count = sum(1 for f in visible_figures if f.get("type") == "method")
        result_count = sum(1 for f in visible_figures if f.get("type") == "result")

        return {
            "figures": all_figures,
            "figure_summary": figure_summary,
            "latex_figures": latex_figures,
            "latex_file_path": str(latex_file_path),
            "method_count": method_count,
            "result_count": result_count,
            "summary_path": str(summary_path),
        }
    
    # Fixed prompts for method figure generation (do not modify)
    _MAIN_FIGURE_PROMPT = """You are an expert Scientific Illustrator for top-tier AI conferences (NeurIPS/CVPR/ICML).

Your task is to generate a professional Illustration (main figure for the paper) based on a research paper abstract and methodology.

**Visual Style Requirements:**

1.  **Style:** Flat vector illustration, clean lines, academic aesthetic. Similar to figures in DeepMind or OpenAI papers.

2.  **Layout:** Organized flow (Left-to-Right, Top-to-Bottom, Circular and other shapes). Group related components logically.

3.  **Color Palette:** Professional pastel tones. White background.

4.  **Text Rendering:** You MUST include legible text labels for key modules or equations mentioned in the methodology (e.g., "Encoder", "Loss", "Transformer").

5.  **Negative Constraints:** NO photorealistic photos, NO messy sketches, NO unreadable text, NO 3D shading artifacts.

**Generation Instruction:**

Highlight the core novelty. Ensure the connection logic makes sense."""

    _SUB_FIGURE_PROMPT_TEMPLATE = """You are an expert Scientific Illustrator for top-tier AI conferences (NeurIPS/CVPR/ICML).

**Your Task:** Generate a **zoomed-in, detailed diagram** showing ONLY the internal architecture and data flow of: {caption}

**CRITICAL SCOPE CONSTRAINT:**
- This figure must depict ONLY this specific module/block, NOT the overall network architecture.
- Do NOT include the encoder-decoder backbone, skip connections, or other modules that belong to the main overview figure.
- Think of this as a "magnified view" of one component — show its internal layers, operations, tensor shapes, and connections in detail.

**Visual Style Requirements:**
1.  **Style:** Flat vector illustration, clean lines, academic aesthetic. Use a similar color palette and line weight as the reference image for visual consistency.
2.  **Layout:** Organized flow (Left-to-Right or Top-to-Bottom). Group related sub-components logically.
3.  **Color Palette:** Professional pastel tones. White background.
4.  **Text Rendering:** You MUST include legible text labels for internal operations, tensor dimensions, and key components (e.g., "DWConv 7x1", "SE Attention", "Sigmoid", "Element-wise Multiply").
5.  **Negative Constraints:** NO photorealistic photos, NO messy sketches, NO unreadable text, NO 3D shading artifacts, NO reproduction of the overall network architecture.

**Generation Instruction:**
Focus on the internal data flow and component interactions within this single module. Highlight what makes this module novel compared to standard operations."""

    def _generate_method_diagrams(
        self,
        methods_latex: str,
        plot_plan: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Generate method architecture diagrams with a simplified pipeline:
          1. LLM plans which figures are needed (caption + subsection mapping)
          2. Generate main figure from full method LaTeX
          3. Generate sub-figures using main figure as style reference
        """
        import re as _re

        logger.info("  Step 1: Planning method figures...")

        plan_prompt = self.load_skill(
            "medical_segmentation/method_figure_plan.md",
            methods_latex=methods_latex,
        )
        plan_response = self.chat(messages=[{"role": "user", "content": plan_prompt}])
        plan = self._parse_json_response(plan_response)
        if plan is None or "figures" not in plan:
            logger.warning("  Failed to parse figure plan, using default (main only)")
            plan = {"figures": [{"figure_id": "method_fig1", "is_main": True,
                                 "target_subsection": None,
                                 "caption": "Overview of the proposed method."}]}

        figures_plan = plan["figures"]
        logger.info(f"  Planned {len(figures_plan)} method figures")

        # Helper: extract subsection LaTeX by title
        def _extract_subsection(title: str) -> str:
            pattern = _re.compile(
                rf"(\\subsection\{{{_re.escape(title)}\}}.*?)(?=\\subsection\{{|\\section\{{|\\end\{{document\}}|$)",
                _re.DOTALL,
            )
            m = pattern.search(methods_latex)
            return m.group(1).strip() if m else ""

        max_fig_retries = 2

        # Step 2: Generate main figure
        logger.info("  Step 2: Generating main figure...")
        main_fig = next((f for f in figures_plan if f.get("is_main")), figures_plan[0])
        main_fig_id = main_fig.get("figure_id", "method_fig1")
        main_image_path = self.figure_dir / f"{main_fig_id}.png"

        main_prompt_text = methods_latex + "\n\n" + self._MAIN_FIGURE_PROMPT

        # Save prompt for debugging
        (self.method_dir / f"{main_fig_id}_prompt.txt").write_text(main_prompt_text, encoding="utf-8")

        main_generated = False
        for attempt in range(max_fig_retries + 1):
            main_generated = self.image_client.generate_image(
                prompt=main_prompt_text,
                output_path=main_image_path,
                model=self.image_config.get("model", "google/gemini-3.1-flash-image-preview"),
                aspect_ratio=self.image_config.get("aspect_ratio", "16:9"),
                image_size=self.image_config.get("image_size", "2K"),
            )
            if main_generated:
                logger.info(f"    ✓ Main figure generated: {main_fig_id}.png")
                break
            if attempt < max_fig_retries:
                delay = 5 * (2 ** attempt)
                logger.warning(
                    f"    Main figure failed (attempt {attempt + 1}/{max_fig_retries + 1}), "
                    f"retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.warning(f"    ⚠ Main figure generation failed after {max_fig_retries + 1} attempts: {main_fig_id}")

        generated_figures = []

        main_latex_code = self._format_latex_figure(main_fig_id, {
            "type": "method", "caption": main_fig.get("caption", "")
        })
        generated_figures.append({
            "figure_id": main_fig_id,
            "type": "method",
            "is_main": True,
            "title": main_fig.get("caption", "Method Overview"),
            "file": str(main_image_path),
            "image_generated": main_generated,
            "caption": main_fig.get("caption", ""),
            "target_subsection": None,
            "placement": "Method section (before \\section{Method})",
            "latex_code": main_latex_code,
            "latex_label": f"fig:{main_fig_id}",
            "reference_example": f"Figure~\\ref{{fig:{main_fig_id}}}",
        })

        # Step 3: Generate sub-figures
        sub_figs = [f for f in figures_plan if not f.get("is_main")]
        if sub_figs:
            logger.info(f"  Step 3: Generating {len(sub_figs)} sub-figures...")

        for fig_spec in sub_figs:
            fig_id = fig_spec.get("figure_id", f"method_fig{len(generated_figures) + 1}")
            caption = fig_spec.get("caption", "")
            target_subsec = fig_spec.get("target_subsection", "")

            subsec_latex = _extract_subsection(target_subsec) if target_subsec else ""
            if not subsec_latex:
                logger.warning(f"    ⚠ Could not extract subsection '{target_subsec}', using full method")
                subsec_latex = methods_latex

            sub_prompt = self._SUB_FIGURE_PROMPT_TEMPLATE.format(caption=caption)
            full_sub_prompt = subsec_latex + "\n\n" + sub_prompt

            (self.method_dir / f"{fig_id}_prompt.txt").write_text(full_sub_prompt, encoding="utf-8")

            sub_image_path = self.figure_dir / f"{fig_id}.png"
            ref_image = main_image_path if main_generated else None

            sub_generated = False
            for attempt in range(max_fig_retries + 1):
                current_ref = ref_image
                if attempt == max_fig_retries and ref_image is not None:
                    logger.info(f"    Last attempt for {fig_id}: dropping reference image")
                    current_ref = None

                sub_generated = self.image_client.generate_image(
                    prompt=full_sub_prompt,
                    output_path=sub_image_path,
                    model=self.image_config.get("model", "google/gemini-3.1-flash-image-preview"),
                    aspect_ratio=self.image_config.get("aspect_ratio", "16:9"),
                    image_size=self.image_config.get("image_size", "2K"),
                    reference_image_path=current_ref,
                )
                if sub_generated:
                    logger.info(f"    ✓ Sub-figure generated: {fig_id}.png (subsection: {target_subsec})")
                    break

                if attempt < max_fig_retries:
                    delay = 5 * (2 ** attempt)
                    logger.warning(
                        f"    Sub-figure {fig_id} failed (attempt {attempt + 1}/{max_fig_retries + 1}), "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.warning(f"    ⚠ Sub-figure generation failed after {max_fig_retries + 1} attempts: {fig_id}")

            sub_latex_code = self._format_latex_figure(fig_id, {
                "type": "method", "caption": caption
            })
            generated_figures.append({
                "figure_id": fig_id,
                "type": "method",
                "is_main": False,
                "title": caption,
                "file": str(sub_image_path),
                "image_generated": sub_generated,
                "caption": caption,
                "target_subsection": target_subsec,
                "placement": f"Method section (before \\subsection{{{target_subsec}}})",
                "latex_code": sub_latex_code,
                "latex_label": f"fig:{fig_id}",
                "reference_example": f"Figure~\\ref{{fig:{fig_id}}}",
            })

        return generated_figures
    
    def _format_latex_figure(self, figure_id: str, fig_meta: Dict) -> str:
        """
        Generate complete LaTeX figure environment code.
        
        Args:
            figure_id: Figure ID (e.g., method_fig1, result_fig1)
            fig_meta: Figure metadata dictionary containing caption, type, etc.
            
        Returns:
            Complete LaTeX figure code block
        """
        # Select appropriate width parameter based on figure type
        fig_type = fig_meta.get('type', 'result')
        if fig_type == 'method':
            width_param = r"0.7\textwidth"  # Method figures use 0.7 page width
            file_extension = 'png'  # Use PNG for method figures
        else:
            width_param = r"0.7\textwidth"  # Result figures use 0.7 page width
            file_extension = 'pdf'  # Prefer PDF vector graphics for result figures
        
        # Generate LaTeX code
        latex_code = f"""\\begin{{figure}}[!t]
\\centering
\\includegraphics[width={width_param}]{{figure/{figure_id}.{file_extension}}}
\\caption{{{fig_meta.get('caption', 'Figure caption missing')}}}
\\label{{fig:{figure_id}}}
\\end{{figure}}"""
        
        return latex_code
    
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Extract and parse JSON from an LLM response that may be wrapped in markdown."""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            result = json_repair.loads(json_str)
            if isinstance(result, list):
                if result and isinstance(result[0], dict) and "is_main" in result[0]:
                    result = {"figures": result}
                else:
                    result = {"plots": result}
                logger.info(f"Wrapped JSON list into dict with key '{list(result.keys())[0]}'")
            if not isinstance(result, dict):
                logger.error(f"Parsed JSON is not a dictionary (got {type(result).__name__})")
                return None
            return result
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None
    
    def _generate_segmentation_visualization(self) -> List[Dict[str, Any]]:
        """Generate deterministic segmentation comparison figure using NIfTI data."""
        if not self.experiment_dir:
            logger.warning("No experiment_dir, skipping segmentation visualization")
            return []
        
        try:
            from func.segmentation_viz import generate_segmentation_comparison
        except ImportError:
            logger.warning("segmentation_viz module not available, skipping")
            return []
        
        output_path = self.figure_dir / "seg_comparison.png"
        result = generate_segmentation_comparison(
            experiment_dir=self.experiment_dir,
            output_path=output_path,
            max_rows=3,
            max_baselines=4,
        )
        
        if result is None:
            return []
        
        figure_id = "seg_comparison"
        latex_code = self._format_latex_figure(figure_id, {
            "type": "result",
            "caption": result["caption"],
        })
        # Override extension since it's a PNG
        latex_code = latex_code.replace(f"{figure_id}.pdf", f"{figure_id}.png")
        
        return [{
            "figure_id": figure_id,
            "type": "result",
            "title": result["title"],
            "file": result["file"],
            "image_generated": True,
            "caption": result["caption"],
            "placement": result["placement"],
            "plot_type": result["plot_type"],
            "latex_code": latex_code,
            "latex_label": f"fig:{figure_id}",
            "reference_example": f"Figure~\\ref{{fig:{figure_id}}}",
        }]
    
    def _collect_per_case_context(self) -> str:
        """Collect per-case metrics formatted for LLM prompt context."""
        if not self.experiment_dir:
            return ""
        try:
            from func.segmentation_viz import collect_per_case_data
            context = collect_per_case_data(self.experiment_dir)
            return context or ""
        except Exception as e:
            logger.warning(f"Failed to collect per-case data: {e}")
            return ""
    
    def _generate_result_plots(
        self,
        research_idea: str,
        experimental_results: str,
        ablation_results: str,
        plot_plan: str,
        per_case_data: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Generate matplotlib code for result visualization, then execute via OpenHands.
        
        Pipeline: LLM generates plot code JSON -> save .py files -> PlotCodeExecutor
                  runs scripts in sandbox -> collect generated PDF/PNG
        
        Falls back to code-only output if execution fails.
        """
        logger.info("  Generating result plot code...")
        
        prompt = self.load_skill(
            "medical_segmentation/result_plot_code.md",
            research_idea=research_idea,
            experimental_results=experimental_results,
            ablation_results=ablation_results,
            plot_plan=plot_plan,
            per_case_data=per_case_data
        )
        
        response = self.chat(messages=[{"role": "user", "content": prompt}])
        
        plot_specs = self._parse_json_response(response)
        if plot_specs is None:
            return []
        
        # Step 1: Generate and save all plot scripts
        generated_figures = []
        plot_scripts = {}
        
        for idx, spec in enumerate(plot_specs.get("plots", []), 1):
            figure_id = f"result_fig{idx}"
            filename = f"{figure_id}_plot.py"
            filepath = self.result_dir / filename
            spec_issues = self._collect_plot_spec_issues(spec)
            
            code_content = self._format_plot_code(figure_id, spec)
            filepath.write_text(code_content, encoding='utf-8')
            plot_scripts[filename] = code_content
            
            latex_code = self._format_latex_figure(figure_id, {
                "type": "result",
                "caption": spec.get('caption', '')
            })
            
            generated_figures.append({
                "figure_id": figure_id,
                "type": "result",
                "title": spec.get('title', 'Untitled'),
                "file": str(filepath),
                "image_generated": False,
                "caption": spec.get('caption', ''),
                "placement": spec.get('placement', 'Experiments section'),
                "plot_type": spec.get('plot_type', 'unknown'),
                "latex_code": latex_code,
                "latex_label": f"fig:{figure_id}",
                "reference_example": f"Figure~\\ref{{fig:{figure_id}}}",
                "validation_passed": False,
                "validation_errors": spec_issues[:],
            })
            
            logger.info(f"    ✓ Created {filename}")
        
        # Step 2: Execute scripts via PlotCodeExecutor (OpenHands)
        if plot_scripts:
            self._execute_plot_scripts(plot_scripts, generated_figures)
        
        return generated_figures
    
    def _execute_plot_scripts(
        self,
        plot_scripts: Dict[str, str],
        figures: List[Dict[str, Any]],
    ) -> None:
        """Execute plot scripts, validate outputs, and retry failed figures."""
        repair_feedback = self._build_result_repair_feedback(figures)

        for attempt in range(1, self._MAX_RESULT_REPAIR_ROUNDS + 2):
            logger.info(
                "  Result figure repair round %s/%s",
                attempt,
                self._MAX_RESULT_REPAIR_ROUNDS + 1,
            )

            self._cleanup_result_artifacts(clear_workspace=True)
            openhands_ok = self._try_openhands_execution(
                plot_scripts,
                figures,
                repair_feedback=repair_feedback,
            )
            invalid_figs = self._collect_invalid_result_figures(figures)

            if invalid_figs:
                if not openhands_ok:
                    logger.info("  OpenHands did not fully repair figures; trying local execution...")
                self._try_local_execution(plot_scripts, figures)
                self._rescue_result_figures(figures)
                invalid_figs = self._collect_invalid_result_figures(figures)

            if not invalid_figs:
                logger.info("  ✓ All result figures passed validation")
                return

            repair_feedback = self._build_result_repair_feedback(invalid_figs)
            logger.warning(
                "  Result figure validation still failing after round %s: %s",
                attempt,
                "; ".join(
                    f"{fig['figure_id']} -> {', '.join(fig.get('validation_errors', []))}"
                    for fig in invalid_figs
                ),
            )

        logger.warning(
            "  Some result figures remain invalid and will be excluded from the paper: %s",
            ", ".join(fig["figure_id"] for fig in self._collect_invalid_result_figures(figures)),
        )

    def _try_openhands_execution(
        self,
        plot_scripts: Dict[str, str],
        figures: List[Dict[str, Any]],
        repair_feedback: str = "",
    ) -> bool:
        """Try to execute via OpenHands PlotCodeExecutor.  Returns True on success."""
        try:
            from func.plot_executor import PlotCodeExecutor
        except ImportError:
            logger.info("PlotCodeExecutor not available, will use local fallback")
            return False
        
        executor = PlotCodeExecutor()
        workspace = self.result_dir / "openhands_workspace"
        
        if not executor.initialize(workspace):
            logger.info("PlotCodeExecutor initialization failed, will use local fallback")
            return False
        
        try:
            data_files = self._collect_data_files()
            success, generated_paths = executor.execute_plot_task(
                plot_scripts=plot_scripts,
                output_dir=self.figure_dir,
                data_files=data_files,
                repair_feedback=repair_feedback,
            )
            self._sync_workspace_scripts(workspace, plot_scripts.keys())
            self._update_figure_status(figures)
            if success:
                return True
            logger.info("PlotCodeExecutor did not produce all expected outputs")
            return False
        except Exception as e:
            logger.info(f"OpenHands execution failed ({e}), will use local fallback")
            return False
        finally:
            executor.cleanup()

    def _try_local_execution(
        self,
        plot_scripts: Dict[str, str],
        figures: List[Dict[str, Any]],
    ) -> None:
        """Fallback: run each plot script as a subprocess in the current environment."""
        import subprocess as _sp
        logger.info("  Running plot scripts locally (subprocess fallback)...")
        for filename, code in plot_scripts.items():
            script_path = self.result_dir / filename
            if not script_path.exists():
                script_path.write_text(code, encoding="utf-8")
            script_text = script_path.read_text(encoding="utf-8", errors="replace")
            if self._contains_placeholder_code(script_text):
                logger.warning(
                    "    ✗ Skipping local exec for %s because placeholder code remains",
                    filename,
                )
                continue
            try:
                result = _sp.run(
                    ["python", str(script_path)],
                    capture_output=True, text=True, timeout=120,
                    cwd=str(self.result_dir),
                )
                if result.returncode == 0:
                    logger.info(f"    ✓ Local exec OK: {filename}")
                else:
                    logger.warning(f"    ✗ {filename} failed (rc={result.returncode}): "
                                   f"{result.stderr[:300]}")
            except Exception as e:
                logger.warning(f"    ✗ {filename} local exec error: {e}")

        self._update_figure_status(figures)

    def _rescue_result_figures(self, figures: List[Dict[str, Any]]) -> None:
        """Copy result_fig files from result/ subdirectories to figure_dir if validation failed."""
        import shutil as _shutil
        search_dirs = [self.result_dir]
        oh_ws = self.result_dir / "openhands_workspace"
        if oh_ws.exists():
            search_dirs.append(oh_ws)

        for fig in figures:
            if fig.get("type") != "result":
                continue
            if fig.get("validation_passed"):
                continue
            fig_id = fig["figure_id"]
            for ext in [".pdf", ".png"]:
                dest = self.figure_dir / f"{fig_id}{ext}"
                for search_dir in search_dirs:
                    src = search_dir / f"{fig_id}{ext}"
                    if src.exists() and src.stat().st_size > 0:
                        _shutil.copy2(src, dest)
                        logger.info(f"    ✓ Rescued {fig_id}{ext} from {search_dir.name}/")
                        break

        self._update_figure_status(figures)

    def _enforce_figure_size_limit(self, figures: List[Dict[str, Any]]) -> None:
        """Compress any figure file in figure_dir that exceeds _MAX_FIGURE_BYTES."""
        from func.openrouter_client import OpenRouterClient

        checked = set()
        for fig in figures:
            fig_id = fig["figure_id"]
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = self.figure_dir / f"{fig_id}{ext}"
                if candidate.exists() and candidate not in checked:
                    checked.add(candidate)
                    if candidate.stat().st_size > self._MAX_FIGURE_BYTES:
                        logger.info(
                            f"  Compressing oversized figure {candidate.name} "
                            f"({candidate.stat().st_size / 1024 / 1024:.1f}MB > 2MB)"
                        )
                        OpenRouterClient._compress_saved_image(
                            candidate,
                            max_long_edge=2048,
                            max_bytes=self._MAX_FIGURE_BYTES,
                        )

    def _update_figure_status(self, figures: List[Dict[str, Any]]) -> None:
        """Scan figure_dir for generated images and update metadata."""
        for fig in figures:
            if fig.get("type") == "result":
                self._validate_result_figure(fig)
                if fig.get("validation_passed"):
                    logger.info(f"    ✓ Validated: {fig['figure_id']}")
                elif fig.get("validation_errors"):
                    logger.warning(
                        "    ✗ Invalid result figure %s: %s",
                        fig["figure_id"],
                        "; ".join(fig["validation_errors"]),
                    )
                continue
            fig_id = fig["figure_id"]
            for ext in [".pdf", ".png"]:
                candidate = self.figure_dir / f"{fig_id}{ext}"
                if candidate.exists():
                    fig["image_generated"] = True
                    fig["file"] = str(candidate)
                    logger.info(f"    ✓ Found: {fig_id}{ext}")
                    break
    
    def _collect_data_files(self) -> Optional[Dict[str, Path]]:
        """Collect data files from the best experiment node."""
        if not self.experiment_dir:
            return None
        
        data_files = {}
        results_dir = self.experiment_dir / "logs" / "0-run" / "results"
        
        if not results_dir.exists():
            return None
        
        try:
            from func.segmentation_viz import find_best_node
            best = find_best_node(results_dir)
            if not best:
                return None
            _, _, val_dir = best
            
            summary = val_dir / "summary.json"
            if summary.exists():
                data_files["summary.json"] = summary
        except Exception as e:
            logger.warning(f"Failed to find best node: {e}")
            for node_dir in results_dir.iterdir():
                if not node_dir.is_dir():
                    continue
                summary = node_dir / "model_results" / "validation" / "summary.json"
                if summary.exists():
                    data_files["summary.json"] = summary
                    break
        
        return data_files if data_files else None

    def _visible_figures(self, figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        visible = []
        for fig in figures:
            if fig.get("type") != "result":
                visible.append(fig)
                continue
            if fig.get("validation_passed", fig.get("image_generated", False)):
                visible.append(fig)
        return visible

    def _collect_plot_spec_issues(self, spec: Dict[str, Any]) -> List[str]:
        issues = []
        if not (spec.get("plot_code") or "").strip():
            issues.append("missing plot_code in plot spec")
        if not (spec.get("data_preparation") or "").strip():
            issues.append("missing data_preparation in plot spec")
        if self._contains_placeholder_code(spec.get("plot_code", "")):
            issues.append("plot_code still contains placeholders")
        if self._contains_placeholder_code(spec.get("data_preparation", "")):
            issues.append("data_preparation still contains placeholders")
        if not (spec.get("caption") or "").strip():
            issues.append("missing caption in plot spec")
        return issues

    def _contains_placeholder_code(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return any(marker in lowered for marker in self._PLACEHOLDER_MARKERS)

    def _cleanup_result_artifacts(self, clear_workspace: bool = False) -> None:
        import shutil as _shutil

        for base_dir in [self.figure_dir, self.result_dir]:
            for path in base_dir.glob("result_fig*.pdf"):
                path.unlink(missing_ok=True)
            for path in base_dir.glob("result_fig*.png"):
                path.unlink(missing_ok=True)

        if clear_workspace:
            workspace = self.result_dir / "openhands_workspace"
            if workspace.exists():
                _shutil.rmtree(workspace, ignore_errors=True)

    def _sync_workspace_scripts(self, workspace: Path, script_names) -> None:
        if not workspace.exists():
            return
        for filename in script_names:
            src = workspace / filename
            dest = self.result_dir / filename
            if src.exists():
                dest.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                logger.info(f"    ✓ Synced repaired script: {filename}")

    def _collect_invalid_result_figures(self, figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            fig for fig in figures
            if fig.get("type") == "result" and not fig.get("validation_passed", False)
        ]

    def _build_result_repair_feedback(self, figures: List[Dict[str, Any]]) -> str:
        issues = []
        for fig in figures:
            if fig.get("type") != "result":
                continue
            validation_errors = fig.get("validation_errors", [])
            if validation_errors:
                issues.append(f"- {fig['figure_id']}: " + "; ".join(validation_errors))
        if not issues:
            return ""
        return (
            "Repair the following figure issues before considering the task successful:\n"
            + "\n".join(issues)
        )

    def _validate_result_figure(self, fig: Dict[str, Any]) -> None:
        fig_id = fig["figure_id"]
        pdf_path = self.figure_dir / f"{fig_id}.pdf"
        png_path = self.figure_dir / f"{fig_id}.png"
        script_path = self.result_dir / f"{fig_id}_plot.py"

        errors = []
        if not pdf_path.exists():
            errors.append("missing expected PDF output")
        if not png_path.exists():
            errors.append("missing expected PNG output")

        if script_path.exists():
            script_text = script_path.read_text(encoding="utf-8", errors="replace")
            if self._contains_placeholder_code(script_text):
                errors.append("plot script still contains placeholders")
        else:
            errors.append("missing plot script in result directory")

        if png_path.exists():
            if png_path.stat().st_size < self._MIN_RESULT_PNG_BYTES:
                errors.append("PNG output is suspiciously small")
            png_ok, png_error = self._png_has_visual_content(png_path)
            if not png_ok and png_error:
                errors.append(png_error)

        fig["validation_errors"] = errors
        fig["validation_passed"] = not errors
        fig["image_generated"] = not errors
        if not errors:
            self._finalize_result_figure_metadata(fig)
            fig["file"] = str(pdf_path if pdf_path.exists() else png_path)

    def _finalize_result_figure_metadata(self, fig: Dict[str, Any]) -> None:
        title = (fig.get("title") or "").strip()
        caption = (fig.get("caption") or "").strip()
        plot_type = (fig.get("plot_type") or "").strip()

        if not title or title.lower() == "untitled":
            if caption:
                title = caption
            elif plot_type and plot_type.lower() != "unknown":
                title = plot_type.replace("_", " ").title()
            else:
                title = f"Result analysis for {fig['figure_id']}"

        if not caption:
            if title:
                caption = title
            elif plot_type and plot_type.lower() != "unknown":
                caption = f"Result analysis plot showing {plot_type.replace('_', ' ')}."
            else:
                caption = f"Result analysis plot for {fig['figure_id']}."

        if not plot_type or plot_type.lower() == "unknown":
            plot_type = "analysis_plot"

        fig["title"] = title
        fig["caption"] = caption
        fig["plot_type"] = plot_type
        fig["latex_code"] = self._format_latex_figure(
            fig["figure_id"],
            {"type": "result", "caption": caption},
        )

    def _png_has_visual_content(self, image_path: Path) -> tuple[bool, str]:
        try:
            from PIL import Image, ImageChops, ImageStat

            with Image.open(image_path) as img:
                rgb = img.convert("RGB")
                white_bg = Image.new("RGB", rgb.size, (255, 255, 255))
                diff = ImageChops.difference(rgb, white_bg)
                bbox = diff.getbbox()
                if bbox is None:
                    return False, "PNG output is blank/fully white"

                stat = ImageStat.Stat(diff.convert("L"))
                if stat.mean[0] < 0.5:
                    return False, "PNG output has near-zero visual content"
        except Exception as e:
            logger.warning(f"    Could not inspect PNG content for {image_path.name}: {e}")
        return True, ""
    
    def _format_plot_code(self, figure_id: str, spec: Dict) -> str:
        """
        Format result plot Python code.
        
        Args:
            figure_id: Figure ID
            spec: Figure specification dictionary
            
        Returns:
            Formatted Python code
        """
        caption_text = spec.get('caption', '').replace('"', '\\"')
        placement_text = spec.get('placement', 'Experiments section')
        abs_figure_dir = str(self.figure_dir.resolve())
        code_content = f'''"""
{spec.get('title', 'Untitled Plot')}
Figure ID: {figure_id}  |  Plot Type: {spec.get('plot_type', 'unknown')}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams.update({{
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 10, "font.family": "Arial",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
}})
sns.set_palette("Set2")

# Neutralise plt.show() so it does not clear the figure before saving
_original_show = plt.show
plt.show = lambda *a, **kw: None

FIGURE_ID = "{figure_id}"
OUTPUT_DIR = Path(r"{abs_figure_dir}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Generating {{FIGURE_ID}} -> {{OUTPUT_DIR}}")

# ========== Data Preparation ==========
{spec.get('data_preparation', '# TODO: Add your data here')}

# ========== Plot Code ==========
{spec.get('plot_code', '# TODO: Add plotting code here')}

# ========== Save (canonical paths for the pipeline) ==========
plt.tight_layout()
for ext in (".pdf", ".png"):
    plt.savefig(OUTPUT_DIR / f"{{FIGURE_ID}}{{ext}}", bbox_inches="tight", dpi=300)
print(f"Saved {{FIGURE_ID}}.pdf / .png to {{OUTPUT_DIR}}")

plt.show = _original_show
plt.close("all")
'''
        return code_content
    
    def _generate_latex_figures_file(self, figures: List[Dict]) -> str:
        """
        Generate file content containing LaTeX code for all figures.
        
        Args:
            figures: List of metadata for all figures
            
        Returns:
            Collection of figure codes in LaTeX format
        """
        latex_lines = ["% Auto-generated LaTeX figure environments",
                      "% Generated by PlotAgent",
                      "% DO NOT EDIT MANUALLY - Regenerate via PlotAgent.run()",
                      ""]
        
        # Group by type
        method_figs = [f for f in figures if f['type'] == 'method']
        result_figs = [f for f in figures if f['type'] == 'result']
        
        if method_figs:
            latex_lines.append("% ===== Method Diagrams =====")
            for fig in method_figs:
                latex_lines.append(f"% {fig['figure_id']}: {fig.get('title', '')}")
                latex_lines.append(f"% Placement: {fig.get('placement', 'Method section')}")
                latex_lines.append(fig['latex_code'])
                latex_lines.append("")

        if result_figs:
            latex_lines.append("% ===== Result Plots =====")
            for fig in result_figs:
                latex_lines.append(f"% {fig['figure_id']}: {fig.get('title', '')}")
                latex_lines.append(f"% Placement: {fig.get('placement', 'Experiments section')}")
                latex_lines.append(fig['latex_code'])
                latex_lines.append("")
        
        return "\n".join(latex_lines)
    
    def _generate_figure_summary(self, figures: List[Dict]) -> str:
        """
        Generate figure summary for PaperWritingAgent.

        The summary is written in English so the writing prompt (also English)
        can reference it without language-switching confusion.
        """
        summary_lines = ["# Figure Summary\n"]
        summary_lines.append(f"Total figures generated: {len(figures)}\n")

        method_figs = [f for f in figures if f.get('type') == 'method']
        if method_figs:
            summary_lines.append("## Method Diagrams (auto-inserted after writing)\n")
            summary_lines.append("Method figures are inserted automatically into the "
                                 "LaTeX after writing. Do NOT insert them yourself.\n")
            for fig in method_figs:
                generated = fig.get('image_generated', False)
                status = "generated" if generated else "pending"
                summary_lines.append(f"### {fig['figure_id']}: {fig.get('title', '')}\n")
                summary_lines.append(f"- **status**: {status}")
                summary_lines.append(f"- **placement**: {fig.get('placement', 'Method section')}")
                summary_lines.append(f"- **caption**: {fig.get('caption', '')}")
                summary_lines.append(f"- **latex_label**: `{fig.get('latex_label', '')}`")
                summary_lines.append(f"- **reference_example**: `{fig.get('reference_example', '')}`")
                summary_lines.append("")

        result_figs = [f for f in figures if f.get('type') == 'result']
        if result_figs:
            summary_lines.append("## Result Figures (type=result)\n")
            summary_lines.append("You MUST insert the latex_code below into the "
                                 "Experiments section and reference each figure.\n")
            for fig in result_figs:
                generated = fig.get('image_generated', False)
                status = "image ready" if generated else "code only"
                summary_lines.append(f"### {fig['figure_id']}: {fig.get('title', '')} "
                                     f"(type=result)\n")
                summary_lines.append(f"- **status**: {status}")
                summary_lines.append(f"- **plot_type**: {fig.get('plot_type', 'visualization')}")
                summary_lines.append(f"- **placement**: {fig.get('placement', 'Experiments section')}")
                summary_lines.append(f"- **caption**: {fig.get('caption', '')}")
                summary_lines.append(f"- **latex_label**: `{fig.get('latex_label', '')}`")
                summary_lines.append(f"- **reference_example**: `{fig.get('reference_example', '')}`")
                summary_lines.append("")
                summary_lines.append("**latex_code**:")
                summary_lines.append("```latex")
                summary_lines.append(fig.get('latex_code', ''))
                summary_lines.append("```\n")

        return "\n".join(summary_lines)
