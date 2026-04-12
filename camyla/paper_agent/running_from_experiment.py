#!/usr/bin/env python3
"""
Paper Generation Script using Real Camyla Experiment Results
Wrapper around paper_generation_api.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add paper_agent to path (for local imports like func.*, agents.*)
project_root = Path(__file__).parent
sys.path.append(str(project_root))
# Add project root to path (for camyla.* package imports)
sys.path.append(str(project_root.parent.parent))

from paper_generation_api import generate_paper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Generate paper from Camyla experiment results")
    parser.add_argument("--experiment-dir", required=True, help='Path to Camyla experiment directory')
    parser.add_argument("--debug-citations", action="store_true", help='Skip API calls for citations')
    parser.add_argument("--output-dir", default=None, help='Custom output directory')
    parser.add_argument(
        "--config",
        default=str(project_root.parent.parent / "config1.yaml"),
        help="Path to the config file used by Paper Agent (default: repo-root config1.yaml)",
    )
    args = parser.parse_args()
    
    success = generate_paper(
        experiment_dir=args.experiment_dir,
        debug_citations=args.debug_citations,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
