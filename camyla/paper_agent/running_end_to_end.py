#!/usr/bin/env python3
"""
End-to-End Test Script for Camyla Agent

This script runs the complete pipeline:
1. Part 1: Generate research idea from PDFs
2. Mock Experiment: Simulate experimental results
3. Part 2: Analyze results and plan ablations
4. Mock Ablation: Simulate ablation study results
5. Part 3: Write paper with citations and compile PDF

Usage:
    python running_end_to_end.py --task multi_dataset_example1 --debug-citations
"""

import sys
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from camyla.paper_agent.agents.part1_idea import PaperSummaryAgent, IdeaRefineAgent, IdeaGenerationAgent, IdeaVerificationAgent
from camyla.paper_agent.agents.mock_agents import MockExperimentAgent, MockAblationAgent
from camyla.paper_agent.agents.part2_analysis import ResultAnalysisAgent
from camyla.paper_agent.agents.part3_writing import PaperWritingAgent, BibtexAgent
from camyla.paper_agent.agents.dataset_metadata_agent import DatasetMetadataAgent
from camyla.paper_agent.func.latex_utils import LatexCompiler, deduplicate_bibtex_entries
from camyla.paper_agent.func.task_loader import TaskConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # python G:\paper_agent\running_end_to_end.py
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Camyla Agent - End-to-End Test")
    parser.add_argument("--task", type=str, default='hnt_2dataset',
                        help='Task name (e.g., MODID, TN3K, BaGLS)')
    parser.add_argument("--debug-citations", type=bool, default=False,
                        help='Enable debug mode: skip Semantic Scholar API and remove all citation placeholders')
    parser.add_argument("--part0-only", type=bool, default=False,
                        help='Run only Part 0 (Dataset Metadata Check) and exit')
    parser.add_argument("--dry-run", action="store_true",
                        help='Run without making actual expensive LLM calls (where supported)')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Camyla Agent - End-to-End Test")
    logger.info("="*60)
    
    # Load task configuration
    try:
        task_dir = Path("task") / args.task
        task = TaskConfig(task_dir)
        logger.info(f"Loaded task: {task.get_task_name()}")
        logger.info(f"Topic: {task.get_topic()}")
        
        # Display loaded datasets
        dataset_count = task.get_dataset_count()
        logger.info(f"Datasets: {dataset_count}")
        for i, ds in enumerate(task.dataset_info, 1):
            logger.info(f"  [{i}] {ds.get('name', 'Unknown')}")
    except Exception as e:
        logger.error(f"Failed to load task configuration: {e}")
        return
    
    # Get configuration from task
    pdf_files = [str(p) for p in task.get_pdf_paths()]
    topic = task.get_topic()
    dataset_info = task.get_dataset_info()
    style_path = task.get_style_path()
    output_base = task_dir / "outputs"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # ===== PART 0: Ensure Dataset Metadata =====
    logger.info("\n[PART 0] Checking Dataset Metadata...")
    
    if "datasets" in task.config:
        for ds_item in task.config["datasets"]:
            dataset_name = None
            if isinstance(ds_item, dict):
                dataset_name = ds_item.get("name")
            
            if dataset_name:
                dataset_dir = Path("dataset") / dataset_name
                metadata_path = dataset_dir / "metadata.json"
                
                # Always invoke the agent and let it handle incremental-update logic internally
                logger.info(f"  Processing {dataset_name}...")
                try:
                    agent = DatasetMetadataAgent()
                    metadata = agent.run(dataset_name, str(dataset_dir))
                    
                    # Save/update metadata.json
                    dataset_dir.mkdir(parents=True, exist_ok=True)
                    
                    import json
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                         json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"  ✓ Metadata updated: {metadata_path}")
                except Exception as e:
                    logger.error(f"  ❌ Failed to process {dataset_name}: {e}")
    
    # Reload TaskConfig to get the updated metadata
    task = TaskConfig(task_dir)
    pdf_files = [str(p) for p in task.get_pdf_paths()]
    topic = task.get_topic()
    dataset_info = task.get_dataset_info()
    
    logger.info("✓ Part 0 complete!")

    if args.part0_only:
        logger.info("Exiting after Part 0 as requested.")
        return

    # ===== PART 0.5: Research Challenge Extraction =====
    # Extract research challenges from baseline papers (not dataset PDFs)
    logger.info("\n[PART 0.5] Extracting Research Challenges from Papers...")
    
    from agents.challenge_extraction_agent import ChallengeExtractionAgent
    
    # Load custom instructions (will be used in challenge extraction)
    custom_instructions = task.get_custom_instructions()
    
    challenge_agent = ChallengeExtractionAgent()
    challenge_context = {}
    task_mode = task.get_task_mode()
    
    # Use dataset_info reloaded from Part 0 (to ensure updated dataset names are used)
    # Extract challenges for each dataset
    for ds in dataset_info:
        dataset_name = ds.get('name')
        if not dataset_name:
            continue
        dataset_dir = Path('dataset') / dataset_name
        
        # Check whether the baseline directory exists
        baseline_dir = dataset_dir / "baseline"
        if not baseline_dir.exists():
            logger.warning(f"  No baseline directory for {dataset_name}, skipping challenge extraction")
            continue
        
        # Call the agent directly and let it handle incremental logic internally
        logger.info(f"  Processing challenges for {dataset_name}...")
        try:
            result = challenge_agent.run(
                dataset_name=dataset_name,
                dataset_dir=str(dataset_dir),
                custom_instructions=custom_instructions  # Pass custom instructions
            )
            
            # Handle the result
            if result.get("status") == "success":
                challenge_file = dataset_dir / 'challenges' / f'{task_mode}.md'
                if challenge_file.exists():
                    challenge_context[dataset_name] = challenge_file.read_text(encoding='utf-8')
                    logger.info(f"    ✓ Challenges updated for {dataset_name}")
            elif result.get("status") == "skipped":
                # Even if processing is skipped, try reading the existing file
                logger.info(f"    ⏭️  {result.get('reason', 'Skipped')}")
                challenge_file = dataset_dir / 'challenges' / f'{task_mode}.md'
                if challenge_file.exists():
                    challenge_context[dataset_name] = challenge_file.read_text(encoding='utf-8')
                    logger.info(f"    ✓ Using existing challenges for {dataset_name}")
        except Exception as e:
            logger.error(f"  Error extracting challenges for {dataset_name}: {e}")
            continue
    
    # Merge challenge descriptions for downstream use
    if challenge_context:
        dataset_challenge_description = "\n\n---\n\n".join([
            f"# Challenges for {name}\n\n{content}" 
            for name, content in challenge_context.items()
        ])
        # Save into outputs
        (output_base / "0_research_challenges.md").write_text(
            dataset_challenge_description, 
            encoding='utf-8'
        )
        logger.info("✓ Research challenges extracted and saved!")
    else:
        dataset_challenge_description = None
        logger.info("  No challenges available")

    
    # ===== PART 1: Idea Generation =====
    logger.info("\n[PART 1] Generating Research Idea...")
    
    # custom_instructions already loaded in Part 0.5
    if custom_instructions:
        logger.info("  ✓ Using custom instructions for idea generation")
    
    # 1.1 Summary
    logger.info("  Step 1.1: Summarizing papers...")
    summary_agent = PaperSummaryAgent()
    summaries = summary_agent.run(pdf_files, topic=topic, dataset_info=dataset_info)
    
    # 1.2 Refine
    logger.info("  Step 1.2: Refining ideas...")
    refine_agent = IdeaRefineAgent()
    refined_ideas = refine_agent.run(summaries)
    (output_base / "1_refined_ideas.md").write_text(refined_ideas, encoding='utf-8')
    
    # 1.3 Generate - now with optional custom instructions
    logger.info("  Step 1.3: Generating new idea...")
    gen_agent = IdeaGenerationAgent()
    image_dimension = task.get_image_dimension()  # Get image dimension
    new_idea = gen_agent.run(
        refined_ideas, 
        dataset_info=dataset_info,
        dataset_challenges=dataset_challenge_description,  # Pass extracted challenges
        image_dimension=image_dimension,  # Pass image dimension
        custom_instructions=custom_instructions,  # Pass custom instructions
        task_mode=task_mode  # Pass task mode to differentiate fully_supervised vs domain_adaptation
    )
    (output_base / "2_new_idea.md").write_text(new_idea, encoding='utf-8')
    
    # 1.4 Verify - now with optional custom instructions
    logger.info("  Step 1.4: Verifying idea...")
    verify_agent = IdeaVerificationAgent()
    research_proposal = verify_agent.run(
        new_idea, 
        dataset_info=dataset_info,
        image_dimension=image_dimension,  # Pass image dimension
        custom_instructions=custom_instructions,  # Pass custom instructions
        task_mode=task_mode  # Pass task mode to differentiate fully_supervised vs domain_adaptation
    )
    (output_base / "3_research_proposal.md").write_text(research_proposal, encoding='utf-8')
    
    logger.info("✓ Part 1 complete!")
    
    # ===== MOCK: Experimental Results =====
    logger.info("\n[MOCK] Simulating Experimental Results...")
    
    mock_exp_agent = MockExperimentAgent()
    experimental_results = mock_exp_agent.run(research_proposal, dataset_info=dataset_info)
    (output_base / "4_experimental_results.md").write_text(experimental_results, encoding='utf-8')
    
    logger.info("✓ Mock experiments complete!")
    
    # ===== PART 2: Result Analysis =====
    logger.info("\n[PART 2] Analyzing Results...")
    
    analysis_agent = ResultAnalysisAgent()
    analysis_report = analysis_agent.run(research_proposal, experimental_results)
    (output_base / "5_analysis_report.md").write_text(analysis_report, encoding='utf-8')
    
    logger.info("✓ Part 2 complete!")
    
    # ===== MOCK: Ablation Study =====
    logger.info("\n[MOCK] Simulating Ablation Study...")
    
    mock_ablation_agent = MockAblationAgent()
    # Note: Ablation study focuses on primary (first) dataset only
    ablation_results = mock_ablation_agent.run(
        research_proposal, 
        analysis_report, 
        experimental_results,
        dataset_info=dataset_info
    )
    (output_base / "6_ablation_results.md").write_text(ablation_results, encoding='utf-8')
    
    logger.info("✓ Mock ablation complete!")
    
    # ===== PART 2.5: Figure Generation =====
    logger.info("\n[PART 2.5] Generating Figures...")
    
    from agents.plot_agent import PlotAgent
    
    figure_dir = output_base / "figure"
    plot_agent = PlotAgent(output_figure_dir=figure_dir)
    
    plot_result = plot_agent.run(
        research_idea=research_proposal,
        experimental_results=experimental_results,
        ablation_results=ablation_results,
        plot_plan=analysis_report  # includes the figure plan
    )
    
    figure_summary = plot_result["figure_summary"]
    logger.info(f"  ✓ Generated {plot_result['method_count']} method diagram prompts")
    logger.info(f"  ✓ Generated {plot_result['result_count']} result plot codes")
    logger.info(f"  ✓ Figure summary saved to: {plot_result['summary_path']}")
    
    # ===== PART 3: Paper Writing =====
    logger.info("\n[PART 3] Writing Paper...")
    
    # 3.1 Generate paper
    logger.info("  Step 3.1: Generating paper draft...")
    writing_agent = PaperWritingAgent()
    
    style_ref = ""
    if style_path:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(style_path))
            style_texts = []
            for page in reader.pages:
                style_texts.append(page.extract_text())
            style_ref = "\n".join(style_texts)
            logger.info(f"  Loaded style reference from {style_path}")
        except Exception as e:
            logger.warning(f"  Failed to read style PDF: {e}")
    
    paper_content = writing_agent.run(
        research_idea=research_proposal,
        experimental_results=experimental_results,
        ablation_results=ablation_results,
        figures_description=figure_summary,  # use the figure summary generated by PlotAgent
        template_name="elsevier",
        reference_style=style_ref
    )
    
    # Validate multi-dataset coverage in paper
    if len(dataset_info) > 1:
        logger.info(f"  Validating coverage of {len(dataset_info)} datasets in generated paper...")
        datasets_in_results = [ds.get('name', '') for ds in dataset_info]
        missing_datasets = []
        
        for ds_name in datasets_in_results:
            if ds_name and ds_name not in paper_content:
                missing_datasets.append(ds_name)
        
        if missing_datasets:
            logger.warning("  " + "="*60)
            logger.warning(f"  ⚠️  WARNING: {len(missing_datasets)} dataset(s) may be missing from SOTA comparison!")
            logger.warning(f"  Expected datasets: {', '.join(datasets_in_results)}")
            logger.warning(f"  Potentially missing: {', '.join(missing_datasets)}")
            logger.warning("  → Please check Section 'Experiments > State-of-the-Art Comparisons'")
            logger.warning("  → Each dataset should have its own subsection and table")
            logger.warning("  " + "="*60)
        else:
            logger.info(f"  ✓ All {len(dataset_info)} datasets found in paper content")
    
    (output_base / "7_paper_draft.tex").write_text(paper_content, encoding='utf-8')
    
    # 3.2 Manage citations
    logger.info("  Step 3.2: Managing citations...")
    if args.debug_citations:
        logger.info("  🔧 Debug mode enabled: citations will be removed without API calls")
    bibtex_agent = BibtexAgent(debug_mode=args.debug_citations)
    citation_result = bibtex_agent.run(paper_content)
    
    final_content = citation_result["updated_content"]
    bibtex = deduplicate_bibtex_entries(citation_result["bibtex"])
    
    (output_base / "8_paper_final.tex").write_text(final_content, encoding='utf-8')
    (output_base / "references.bib").write_text(bibtex, encoding='utf-8')
    
    logger.info(f"  Found {len(citation_result['citations'])} citations")
    
    # 3.3 Compile PDF
    logger.info("  Step 3.3: Compiling PDF (optional)...")
    try:
        # PaperWritingAgent now returns full LaTeX content
        full_latex = final_content
        
        (output_base / "main.tex").write_text(full_latex, encoding='utf-8')
        
        compiler = LatexCompiler()
        result = compiler.compile_content(
            full_latex,
            output_pdf=output_base / "paper.pdf",
            template_name="elsevier",  # Use elsevier template
            log_path=output_base / "compile.log"
        )
        
        if result.success:
            logger.info(f"  ✓ PDF compiled: {result.output_path}")
        else:
            logger.warning("  PDF compilation failed (check compile.log)")
            logger.warning("  You can manually compile with: pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex")
    except Exception as e:
        logger.warning(f"  PDF compilation skipped: {e}")
    
    logger.info("✓ Part 3 complete!")
    
    # ===== PART 4: Package Complete LaTeX Project =====
    logger.info("\n[PART 4] Packaging LaTeX project for Overleaf...")
    
    try:
        from func.latex_utils import package_latex_project
        
        # Determine the figure directory
        figure_path = figure_dir if figure_dir.exists() else None
        
        # Package the LaTeX project
        package_result = package_latex_project(
            latex_content=final_content,
            bibtex_content=bibtex,
            template_name="elsevier",
            output_dir=output_base / "overleaf_package",
            figure_dir=figure_path
        )
        
        if package_result:
            logger.info(f"  ✓ LaTeX project packaged: {package_result.absolute()}")
            logger.info("  → You can now upload this folder to Overleaf for compilation")
            logger.info("  → Or compress it to .zip and upload the zip file")
        else:
            logger.warning("  ✗ LaTeX packaging failed (check logs)")
    except Exception as e:
        logger.warning(f"  ✗ LaTeX packaging skipped: {e}")
    
    logger.info("✓ Part 4 complete!")
    
    # ===== Summary =====
    logger.info("\n" + "="*60)
    logger.info("END-TO-END TEST COMPLETE!")
    logger.info("="*60)
    logger.info(f"All outputs saved to: {output_base.absolute()}")
    logger.info("\nGenerated files:")
    for i, f in enumerate(sorted(output_base.glob("*.md")), 1):
        logger.info(f"  {i}. {f.name}")
    
    if (output_base / "paper.pdf").exists():
        logger.info(f"\n📄 Final PDF: {output_base / 'paper.pdf'}")
    
    if (output_base / "overleaf_package").exists():
        logger.info(f"📦 Overleaf Package: {output_base / 'overleaf_package'}")

if __name__ == "__main__":
    main()
