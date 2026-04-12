
import sys
import logging
from pathlib import Path
import os

# Add project root to path (assuming this file is in paper_agent)
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from camyla.paper_agent.func.experiment_adapter import (
    ExperimentResultsAdapter,
    formalize_ablation_names,
    filter_baselines_in_report,
)
from camyla.paper_agent.agents.method_reconcile_agent import MethodReconcileAgent
from camyla.paper_agent.agents.part2_analysis import ResultAnalysisAgent
from camyla.paper_agent.agents.part3_writing import PaperWritingAgent, BibtexAgent
from camyla.paper_agent.agents.plot_agent import PlotAgent
from camyla.paper_agent.func.config_resolver import set_runtime_config_path
from camyla.paper_agent.func.latex_utils import (
    LatexCompiler,
    package_latex_project,
    deduplicate_bibtex_entries,
    deduplicate_latex,
)

# Configure logging
logger = logging.getLogger("paper_agent")

_ABSTRACT_CODE_AVAILABILITY_SENTENCE = (
    "The code and checkpoints are available at https://github.com/yifangao112/camyla"
)

# ---------------------------------------------------------------------------
# Prompts for the post-writing revision and polishing steps (OpenHands agent)
# ---------------------------------------------------------------------------

_REVISION_PROMPT = r"""You are a senior academic manuscript editor. Please review and fix
the file `main.tex` for the following two categories of issues.

### 1. Duplicate-module check
- Are there any \section or \subsection headings that appear more than once
  (e.g. the same subsection title duplicated)?
- Are there large paragraphs whose content is essentially repeated?
- If duplicates are found, keep the more complete version and remove the
  redundant one.

### 2. Cross-section consistency check
- Do the technical details in the Method section (architecture names,
  component names, loss function formulations, hyper-parameters, etc.)
  match what is described in the Experiments / Implementation Details
  section?
- If training recipe details (optimizer, scheduler, batch size, epochs,
  learning-rate schedule, hardware, train/test split) appear in the
  Method section, remove them from the Method section rather than
  propagating them.
- Keep the implementation-policy wording consistent with this rule:
  preprocessing and postprocessing follow nnU-Net, while non-nnU-Net
  baselines keep their original or author-recommended method-specific
  settings when available.

**Rules**
- Edit `main.tex` in-place. Do NOT create new files.
- All edits must keep the LaTeX syntactically correct.
- Preserve every math formula, \cite{}, \ref{}, \label{}, figure and
  table environment exactly as-is (unless the surrounding text is the
  part being fixed).
"""

_POLISH_PROMPT = r"""# Role
You are a senior academic editor in the field of artificial intelligence,
specialising in improving naturalness and readability. Your task is to
rewrite mechanically generated text so that it reads like prose written
by a native-speaking researcher and meets the standards of top-tier
journals (e.g. TMI, MedIA).

# Task
Perform a "de-AI" rewrite of the LaTeX paper in `main.tex` so that
the language style approaches that of a human native researcher.

# Constraints

1. **Vocabulary normalisation**
   - Prefer plain, precise academic vocabulary. Avoid overused complex
     words (e.g. do NOT use *leverage* — use *use*; do NOT use
     *delve into* — use *investigate*; do NOT use *tapestry* — use
     *context*; etc.) unless technically required.
   - Use terminology only when it conveys a specific technical meaning;
     never pile up fancy words for the sake of appearing sophisticated.

2. **Structural naturalisation**
   - **No bullet / numbered lists**: convert every \begin{itemize} or
     \begin{enumerate} block in running text into logically coherent
     prose paragraphs. (Lists inside tables are acceptable.)
   - **Remove mechanical connectors**: delete stiff transition phrases
     such as "First and foremost", "It is worth noting that",
     "Additionally", "Moreover", "Furthermore", "In summary". Instead
     let the logical progression of sentences provide the connection.
   - **Reduce em-dashes**: minimise the use of "---"; prefer commas,
     parentheses, or subordinate clauses.

3. **Typographic rules**
   - **No emphasis formatting**: do NOT use \textbf{} or \textit{} for
     emphasis in running text. Academic writing conveys importance
     through sentence structure, not bold/italic.
     (\textit{} is still fine for the first mention of a defined term
     or for variable names.)
   - Keep the LaTeX clean — do not introduce extraneous formatting.

4. **Modification threshold (critical)**
   - If a sentence or paragraph already reads naturally and shows no
     obvious AI artefacts, **leave it unchanged**. Do not rewrite for
     the sake of rewriting.

5. **Output format**
   - Edit `main.tex` directly.
   - The paper MUST remain entirely in English.
   - Escape special characters properly (%, _, &).
   - Keep all math formulas exactly as they are (preserve $ delimiters).

# Execution protocol — self-check
1. **Naturalness check**: confirm the text reads as if written by a
   human researcher.
2. **Necessity check**: is each edit genuinely improving readability?
   If a change is merely swapping one synonym for another, revert it.

# Banned word list
Remove or replace every occurrence of the following words (unless they
appear inside a math environment or a proper noun):

Accentuate, Ador, Amass, Ameliorate, Amplify, Alleviate, Ascertain,
Advocate, Articulate, Bear, Bolster, Bustling, Cherish, Conceptualize,
Conjecture, Consolidate, Convey, Culminate, Decipher, Demonstrate,
Depict, Devise, Delineate, Delve, Delve Into, Diverge, Disseminate,
Elucidate, Endeavor, Engage, Enumerate, Envision, Enduring, Exacerbate,
Expedite, Foster, Galvanize, Harmonize, Hone, Innovate, Inscription,
Integrate, Interpolate, Intricate, Lasting, Leverage, Manifest, Mediate,
Nurture, Nuance, Nuanced, Obscure, Opt, Originates, Perceive,
Perpetuate, Permeate, Pivotal, Ponder, Prescribe, Prevailing, Profound,
Recapitulate, Reconcile, Rectify, Rekindle, Reimagine, Scrutinize,
Substantiate, Tailor, Testament, Transcend, Traverse, Underscore,
Unveil, Vibrant
"""

import re as _re
from typing import Optional, Tuple


def _edit_and_compile_loop(
    tex_editor,
    content: str,
    task_prompt: str,
    output_base: Path,
    figure_dir: Path,
    bibtex: str,
    step_label: str,
    max_fix_attempts: int = 4,
) -> Tuple[str, bool]:
    """Run an OpenHands editing step followed by compile-fix retries.

    1. ``tex_editor.run_edit(content, task_prompt)``
    2. Compile with ``LatexCompiler``
    3. If compilation fails, send the error log to ``tex_editor.run_fix``
       and re-compile.  Repeat up to *max_fix_attempts* times.

    Returns ``(final_content, compile_success)``.
    """
    content = tex_editor.run_edit(content, task_prompt)

    bibtex = deduplicate_bibtex_entries(bibtex)
    (output_base / "references.bib").write_text(bibtex, encoding="utf-8")

    compiler = LatexCompiler()
    check_pdf = output_base / f"_compile_check_{step_label}.pdf"
    check_log = output_base / f"_compile_check_{step_label}.log"

    for attempt in range(1, max_fix_attempts + 1):
        result = compiler.compile_content(
            content,
            output_pdf=check_pdf,
            template_name="elsevier",
            log_path=check_log,
            figure_dir=figure_dir if figure_dir.exists() else None,
        )
        if result.success:
            logger.info(
                f"  [{step_label}] Compile OK (attempt {attempt})"
            )
            return content, True

        error_log = ""
        if check_log.exists():
            error_log = check_log.read_text(encoding="utf-8", errors="replace")
        if not error_log:
            error_log = (result.stderr or "") + "\n" + (result.stdout or "")

        logger.warning(
            f"  [{step_label}] Compile failed (attempt {attempt}/{max_fix_attempts}), "
            "asking agent to fix..."
        )

        fixed = tex_editor.run_fix(error_log)
        if fixed.strip():
            content = fixed
        else:
            logger.warning(
                f"  [{step_label}] Agent returned empty content during fix, "
                "keeping previous version"
            )
            break

    logger.warning(
        f"  [{step_label}] Could not compile after {max_fix_attempts} fix attempts"
    )
    return content, False


def _extract_methods_section(latex: str) -> str:
    """Extract the Methods / Method section body from a LaTeX document.
    
    Looks for ``\\section{Method...}`` and returns everything up to the next
    ``\\section{`` (or ``\\end{document}``).  Returns empty string on failure.
    """
    pattern = _re.compile(
        r"(\\section\{Method(?:s|ology)?\}.*?)(?=\\section\{|\\end\{document\})",
        _re.DOTALL | _re.IGNORECASE,
    )
    m = pattern.search(latex)
    return m.group(1).strip() if m else ""


def _insert_method_figures(latex: str, figures: list) -> str:
    """Insert method figure LaTeX code into the paper content.

    - Main figure (is_main=True): inserted right before ``\\section{Method...}``
    - Sub-figures (is_main=False): inserted right before the matching
      ``\\subsection{target_subsection}``

    A ``Figure~\\ref{fig:xxx}`` reference sentence is **not** injected
    automatically; that is left to the writing prompt or manual editing.
    """
    if not figures:
        return latex

    for fig in figures:
        if not fig.get("image_generated"):
            continue

        block = "\n\n" + fig["latex_code"] + "\n\n"

        if fig.get("is_main"):
            # Insert before \section{Method...}
            pattern = _re.compile(
                r"(\\section\{Method(?:s|ology)?\})",
                _re.IGNORECASE,
            )
            m = pattern.search(latex)
            if m:
                insert_pos = m.start()
                latex = latex[:insert_pos] + block + latex[insert_pos:]
                logger.info(f"  Inserted main figure {fig['figure_id']} before \\section{{Method}}")
            else:
                logger.warning(f"  Could not find \\section{{Method}} for main figure insertion")
        else:
            target = fig.get("target_subsection", "")
            if not target:
                continue
            # Insert before \subsection{target}
            pattern = _re.compile(
                rf"(\\subsection\{{{_re.escape(target)}\}})",
                _re.IGNORECASE,
            )
            m = pattern.search(latex)
            if m:
                insert_pos = m.start()
                latex = latex[:insert_pos] + block + latex[insert_pos:]
                logger.info(f"  Inserted sub-figure {fig['figure_id']} before \\subsection{{{target}}}")
            else:
                logger.warning(f"  Could not find \\subsection{{{target}}} for sub-figure insertion")

    return latex


def _ensure_code_sentence_in_abstract(latex: str) -> str:
    """Append the fixed code-availability sentence to the abstract."""
    if not latex or _ABSTRACT_CODE_AVAILABILITY_SENTENCE in latex:
        return latex

    pattern = _re.compile(
        r"(\\begin\{abstract\})(.*?)(\\end\{abstract\})",
        _re.DOTALL | _re.IGNORECASE,
    )
    match = pattern.search(latex)
    if not match:
        logger.warning("Could not locate abstract environment; skipping code sentence append")
        return latex

    prefix, abstract_body, suffix = match.groups()
    body = abstract_body.strip()
    if body:
        if not body.endswith((".", "!", "?")):
            body += "."
        body += " "
    body += _ABSTRACT_CODE_AVAILABILITY_SENTENCE

    replacement = f"{prefix}\n{body}\n{suffix}"
    return latex[:match.start()] + replacement + latex[match.end():]


def generate_paper(
    experiment_dir: str,
    debug_citations: bool = False,
    output_dir: str = None,
    config_path: str = None,
) -> bool:
    """
    Generate a paper from Camyla experiment results.
    
    Data loading strategy:
    - Reads from unified experiment_report.md (methodology, results, ablation)
    - Loads best node code from research_summary.json for one-time methodology reconciliation
    
    Args:
        experiment_dir: Path to the experiment directory (containing logs/0-run etc)
        debug_citations: If True, skip Semantic Scholar API calls
        output_dir: Optional custom output directory. If None, uses {experiment_dir}/paper_output
        config_path: Optional config file path (for example config1.yaml)
        
    Returns:
        bool: True if generation completed successfully (PDF compiled or at least TeX generated), False otherwise.
    """
    exp_dir = Path(experiment_dir).resolve()
    if not exp_dir.exists():
        logger.error(f"Experiment directory not found: {exp_dir}")
        return False

    logger.info("="*60)
    logger.info("Starting Paper Generation (API Mode)")
    logger.info(f"Source: {exp_dir}")
    logger.info("="*60)

    resolved_config = set_runtime_config_path(
        explicit_config_path=config_path,
        experiment_dir=exp_dir,
        search_from=__file__,
    )
    if resolved_config:
        logger.info(f"Paper Agent config: {resolved_config}")
    
    # Initialize Adapter
    try:
        adapter = ExperimentResultsAdapter(str(exp_dir))
        dataset_info = adapter.get_dataset_info()
        dataset_context = adapter.get_dataset_context_for_paper()
        
        # Log dataset info for debugging
        logger.info(f"Dataset: {dataset_info[0].get('full_name', 'Unknown')}")
        logger.info(f"Modality: {dataset_info[0].get('modality', 'Unknown')}")
        logger.info(f"Target: {dataset_info[0].get('target_structure', 'Unknown')}")
        
        # Determine output directory
        if output_dir:
            output_base = Path(output_dir)
        else:
            output_base = exp_dir / "paper_output"
        
        output_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_base}")
        
        # Save dataset context for reference
        (output_base / "1_dataset_context.txt").write_text(dataset_context, encoding='utf-8')
        logger.info("  ✓ Saved dataset context")
        
    except Exception as e:
        logger.error(f"Failed to initialize adapter: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ===== PRE-PROCESSING: Compute efficiency metrics =====
    try:
        from camyla.treesearch.compute_efficiency import compute_paper_efficiency_metrics
        run_dir = str(adapter.run_dir)
        logger.info("\n[PRE] Computing computational efficiency metrics...")
        compute_paper_efficiency_metrics(run_dir)
        logger.info("  ✓ Efficiency metrics computed")
    except Exception:
        logger.exception("Efficiency metrics computation failed; continuing without failed methods")

    try:
        # ===== LOAD DATA FROM UNIFIED REPORT =====
        logger.info("\n[LOAD] Loading Research Data from Unified Report...")
        
        # 1. Research Proposal (Methodology) - from experiment_report.md Section 2.1
        research_proposal = adapter.load_proposal()
        (output_base / "3_research_proposal.md").write_text(research_proposal, encoding='utf-8')
        logger.info("  ✓ Loaded Research Proposal (Methodology)")

        # 2. Experimental Results - from experiment_report.md Section 2.2
        experimental_results = adapter.load_experimental_results()
        
        # Filter out obviously poor baselines from the results table
        experimental_results = filter_baselines_in_report(
            str(adapter.run_dir), experimental_results
        )
        
        (output_base / "4_experimental_results.md").write_text(experimental_results, encoding='utf-8')
        logger.info("  ✓ Loaded Experimental Results")
        
        # 3. Ablation Results - from experiment_report.md Section 3
        ablation_results = adapter.load_ablation_results()
        
        # Formalize code-style ablation names into academic abbreviations
        if ablation_results:
            logger.info("  Formalizing ablation experiment names...")
            ablation_results = formalize_ablation_names(ablation_results)
        
        (output_base / "6_ablation_results.md").write_text(ablation_results, encoding='utf-8')
        logger.info("  ✓ Loaded Ablation Results")
        
        # 4. Best Node Code - from research_summary.json (for one-time method reconciliation)
        method_code = adapter.load_best_node_code()
        if method_code:
            (output_base / "2_method_code.py").write_text(method_code, encoding='utf-8')
            logger.info(f"  ✓ Loaded Best Node Code ({len(method_code)} chars)")
        else:
            logger.warning("  ⚠️ No method code available")
            method_code = ""

        # 5. Reconcile methodology details once, then stop using raw code downstream
        if method_code:
            reconcile_agent = MethodReconcileAgent()
            reconciled_research_proposal = reconcile_agent.run(research_proposal, method_code)
            if reconciled_research_proposal.strip():
                research_proposal = reconciled_research_proposal
            (output_base / "3b_reconciled_research_proposal.md").write_text(
                research_proposal,
                encoding="utf-8",
            )
            logger.info("  ✓ Reconciled methodology with implementation code")

        # 6. Training Configuration - from debug.json (paper-facing training params)
        training_config = adapter.load_training_config()
        (output_base / "2b_training_config.txt").write_text(training_config, encoding='utf-8')
        logger.info(f"  ✓ Loaded Training Config")

        baseline_training_policy = adapter.load_baseline_training_policy()
        (output_base / "2c_baseline_training_policy.txt").write_text(
            baseline_training_policy,
            encoding="utf-8",
        )
        logger.info("  ✓ Loaded Baseline Training Policy")

        # 7. Efficiency Metrics - keep for reference, conditionally expose to paper
        efficiency_payload = adapter.get_efficiency_section_for_paper()
        efficiency_text = efficiency_payload["markdown"]
        raw_efficiency_text = adapter.load_efficiency_metrics()
        if raw_efficiency_text:
            (output_base / "4b_efficiency_metrics.md").write_text(
                raw_efficiency_text, encoding="utf-8"
            )
        (output_base / "4c_efficiency_decision.txt").write_text(
            efficiency_payload["decision_reason"], encoding="utf-8"
        )
        if efficiency_text:
            experimental_results = experimental_results.rstrip() + "\n\n" + efficiency_text
            (output_base / "4_experimental_results.md").write_text(
                experimental_results, encoding='utf-8'
            )
            logger.info("  ✓ Appended Efficiency Metrics to Experimental Results")
        elif raw_efficiency_text:
            logger.info(
                "  ✓ Efficiency metrics kept for reference but omitted from paper prompt: %s",
                efficiency_payload["decision_reason"],
            )
        
        # ===== PART 2: Result Analysis =====
        logger.info("\n[PART 2] Analyzing Results...")
        
        analysis_agent = ResultAnalysisAgent()
        analysis_report = analysis_agent.run(research_proposal, experimental_results)
        (output_base / "5_analysis_report.md").write_text(analysis_report, encoding='utf-8')
        logger.info("  ✓ Analysis complete")

        # ===== PART 2.5: Result Figure Generation (phase 1) =====
        logger.info("\n[PART 2.5] Generating result figures (phase 1)...")
        
        figure_dir = output_base / "figure"
        plot_agent = PlotAgent(
            output_figure_dir=figure_dir,
            experiment_dir=exp_dir,
        )
        
        try:
            plot_result = plot_agent.run(
                research_idea=research_proposal,
                experimental_results=experimental_results,
                ablation_results=ablation_results,
                plot_plan=analysis_report,
            )
            figure_summary = plot_result["figure_summary"]
            logger.info(f"  ✓ Result figures generated: {plot_result['result_count']}")
        except Exception as e:
            logger.warning(f"  ⚠️ Result figure generation had issues: {e}")
            import traceback
            traceback.print_exc()
            figure_summary = "Figure generation skipped or failed."

        # ===== PART 3: Paper Writing =====
        logger.info("\n[PART 3] Writing Paper...")
        
        writing_agent = PaperWritingAgent()
        
        paper_content = writing_agent.run(
            research_idea=research_proposal,
            experimental_results=experimental_results,
            ablation_results=ablation_results,
            figures_description=figure_summary,
            dataset_context=dataset_context,
            training_config=training_config,
            baseline_training_policy=baseline_training_policy,
            template_name="elsevier",
            reference_style="" 
        )
        (output_base / "7_paper_draft.tex").write_text(paper_content, encoding='utf-8')
        
        # ===== PART 3.5: Method Figure Generation (phase 2) + Insertion =====
        logger.info("\n[PART 3.5] Generating method figures from paper text (phase 2)...")
        methods_latex = _extract_methods_section(paper_content)
        if methods_latex:
            logger.info(f"  Extracted Methods section ({len(methods_latex)} chars)")
            try:
                method_result = plot_agent.generate_method_figures(
                    methods_latex=methods_latex,
                    plot_plan=analysis_report,
                )
                figure_summary = method_result["figure_summary"]
                generated_count = sum(1 for f in method_result['figures']
                                      if f.get('type') == 'method' and f.get('image_generated'))
                logger.info(f"  ✓ Method diagrams: {method_result['method_count']} "
                           f"({generated_count} AI-generated)")

                # Insert method figures into paper content
                method_figs = [f for f in method_result["figures"] if f.get("type") == "method"]
                if method_figs:
                    logger.info("  Inserting method figures into paper...")
                    paper_content = _insert_method_figures(paper_content, method_figs)
                    (output_base / "7b_paper_with_figures.tex").write_text(paper_content, encoding='utf-8')
                    logger.info("  ✓ Method figures inserted into paper content")

            except Exception as e:
                logger.warning(f"  ⚠️ Method figure generation had issues: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning("  ⚠️ Could not extract Methods section from paper")
        
        # Citations
        logger.info("  Managing citations...")
        bibtex_agent = BibtexAgent(debug_mode=debug_citations)
        citation_result = bibtex_agent.run(paper_content)
        
        final_content = citation_result["updated_content"]
        bibtex = deduplicate_bibtex_entries(citation_result["bibtex"])
        
        # Post-process: remove duplicate figures, headings, etc.
        final_content = deduplicate_latex(final_content)

        # ===== PART 3.7: Article Revision (OpenHands) + compile check =====
        logger.info("\n[PART 3.7] Revising article (duplicate / inconsistency check)...")
        try:
            from func.tex_editor_executor import TexEditorExecutor

            tex_editor = TexEditorExecutor()
            revision_workspace = output_base / "_tex_revision_workspace"
            if tex_editor.initialize(revision_workspace):
                final_content, rev_compiled = _edit_and_compile_loop(
                    tex_editor=tex_editor,
                    content=final_content,
                    task_prompt=_REVISION_PROMPT,
                    output_base=output_base,
                    figure_dir=figure_dir,
                    bibtex=bibtex,
                    step_label="revision",
                )
                (output_base / "8a_paper_revised.tex").write_text(
                    final_content, encoding="utf-8"
                )
                logger.info(
                    f"  ✓ Revision complete (compiled={rev_compiled})"
                )
            tex_editor.cleanup()
        except Exception as e:
            logger.warning(f"  ⚠️ Revision step failed, using un-revised content: {e}")

        # ===== PART 3.8: Article Polishing (OpenHands) + compile check =====
        logger.info("\n[PART 3.8] Polishing article (de-AI rewriting)...")
        try:
            tex_editor2 = TexEditorExecutor()
            polish_workspace = output_base / "_tex_polish_workspace"
            if tex_editor2.initialize(polish_workspace):
                final_content, pol_compiled = _edit_and_compile_loop(
                    tex_editor=tex_editor2,
                    content=final_content,
                    task_prompt=_POLISH_PROMPT,
                    output_base=output_base,
                    figure_dir=figure_dir,
                    bibtex=bibtex,
                    step_label="polish",
                )
                (output_base / "8b_paper_polished.tex").write_text(
                    final_content, encoding="utf-8"
                )
                logger.info(
                    f"  ✓ Polishing complete (compiled={pol_compiled})"
                )
            tex_editor2.cleanup()
        except Exception as e:
            logger.warning(f"  ⚠️ Polishing step failed, using un-polished content: {e}")

        final_content = _ensure_code_sentence_in_abstract(final_content)
        (output_base / "8_paper_final.tex").write_text(final_content, encoding='utf-8')
        (output_base / "references.bib").write_text(bibtex, encoding='utf-8')
        
        # Compile
        logger.info("  Compiling PDF...")
        compile_success = False
        try:
            (output_base / "main.tex").write_text(final_content, encoding='utf-8')
            compiler = LatexCompiler()
            result = compiler.compile_content(
                final_content,
                output_pdf=output_base / "paper.pdf",
                template_name="elsevier",
                log_path=output_base / "compile.log",
                figure_dir=figure_dir if figure_dir.exists() else None,
            )
            if result.success:
                logger.info(f"  ✓ PDF compiled: {result.output_path}")
                compile_success = True
            else:
                logger.warning("  PDF compilation failed.")
        except Exception as e:
            logger.warning(f"  PDF compilation error: {e}")

        # ===== PART 4: Package =====
        logger.info("\n[PART 4] Packaging...")
        try:
            figure_path = figure_dir if figure_dir.exists() else None
            package_result = package_latex_project(
                latex_content=final_content,
                bibtex_content=bibtex,
                template_name="elsevier",
                output_dir=output_base / "overleaf_package",
                figure_dir=figure_path
            )
            if package_result:
                logger.info(f"  ✓ Packaged to: {package_result}")
        except Exception as e:
            logger.warning(f"  Packaging failed: {e}")
            
        logger.info("\nPaper Generation Complete.")
        return True
        
    except Exception as e:
        logger.error(f"Error during paper generation: {e}")
        import traceback
        traceback.print_exc()
        return False
