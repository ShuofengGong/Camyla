#!/usr/bin/env python3
"""
Script to recompile a LaTeX paper.

Usage:
    python recompile_latex.py [output_dir]

Default compile target: task/mia_example1/outputs
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from camyla.paper_agent.func.latex_utils import LatexCompiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def recompile_paper(output_dir: Path):
    """
    Recompile the LaTeX paper in the given directory.

    Args:
        output_dir: Output directory containing main.tex and references.bib.
    """
    # Check whether the directory exists
    if not output_dir.exists():
        logger.error(f"Directory does not exist: {output_dir}")
        return False

    # Check whether main.tex exists
    main_tex = output_dir / "main.tex"
    if not main_tex.exists():
        logger.error(f"main.tex not found: {main_tex}")
        return False

    logger.info(f"=" * 60)
    logger.info(f"Starting compilation: {output_dir}")
    logger.info(f"=" * 60)

    # Read LaTeX content
    logger.info("Reading main.tex...")
    latex_content = main_tex.read_text(encoding='utf-8')

    # Set output paths
    output_pdf = output_dir / "paper.pdf"
    log_path = output_dir / "compile.log"

    # Create the compiler (using the elsevier template)
    logger.info("Initializing LaTeX compiler...")
    compiler = LatexCompiler(timeout=120)  # Increase timeout to 120 seconds

    # Compile
    logger.info("Starting compilation (pdflatex → bibtex → pdflatex ×2)...")
    logger.info("This may take 1-2 minutes; please be patient...")

    # Find the figure directory
    figure_dir = output_dir / "figure"
    if not figure_dir.exists():
        figure_dir = None
        logger.info("No figure directory found; skipping figure copy")
    else:
        logger.info(f"Found figure directory: {figure_dir}")

    try:
        result = compiler.compile_content(
            latex_content=latex_content,
            output_pdf=output_pdf,
            template_name="elsevier",
            log_path=log_path,
            figure_dir=figure_dir,
        )

        logger.info(f"=" * 60)
        if result.success:
            logger.info("✅ Compilation succeeded!")
            logger.info(f"📄 PDF file: {result.output_path}")
            logger.info(f"📋 Log file: {result.log_path}")
            return True
        else:
            logger.error("❌ Compilation failed!")
            logger.error(f"See the log file: {log_path}")
            logger.error("\nLast 100 lines of output:")
            logger.error(result.stderr[-2000:] if result.stderr else "no stderr output")
            return False

    except Exception as e:
        logger.error(f"❌ Error during compilation: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        logger.info(f"=" * 60)


def main():
    """Main entry point."""
    # Parse command-line arguments
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        # Default path
        output_dir = Path("task/thyroid_nodule2/outputs")

    # Convert to absolute path
    output_dir = output_dir.resolve()

    logger.info(f"Target directory: {output_dir}")

    # Run compilation
    success = recompile_paper(output_dir)

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
