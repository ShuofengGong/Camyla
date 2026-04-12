import os
import sys

# Parse --config early and expose it via QWBE_CONFIG_PATH so that any module
# imported below (e.g. log_summarization) resolves get_model_name at import
# time against the correct config file.
def _pre_set_config_env():
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            os.environ.setdefault("QWBE_CONFIG_PATH", os.path.abspath(sys.argv[i + 1]))
            return
        if arg.startswith("--config="):
            os.environ.setdefault("QWBE_CONFIG_PATH", os.path.abspath(arg.split("=", 1)[1]))
            return
    os.environ.setdefault("QWBE_CONFIG_PATH", os.path.abspath("config.yaml"))
_pre_set_config_env()

import os.path as osp
import json
import argparse
import shutil
import torch
import re
from datetime import datetime
from pathlib import Path
from camyla.llm import create_client
from camyla.model_config import get_model_name

from contextlib import contextmanager
from camyla.baseline import ensure_baseline
from camyla.treesearch.perform_experiments_qwbe_with_agentmanager import (
    perform_experiments_qwbe,
)
from camyla.treesearch.qwbe_utils import (
    idea_to_markdown,
    edit_qwbe_config_file,
)
from camyla.utils.token_tracker import token_tracker

# python running_from_experiment.py --experiment-dir <your_experiment_dir>

try:
    from camyla.paper_agent.paper_generation_api import generate_paper
    print("Successfully imported Paper Agent API")
except Exception as e:
    print(f"Warning: Failed to import Paper Agent API: {e}")
    generate_paper = None

import pickle
import os



def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def save_token_tracker(idea_dir):
    with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.get_interactions(), f)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Camyla experiments")
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="elsevier",
        choices=["normal", "elsevier"],
        help="Type of writeup to generate (normal=12 page)",
    )
    parser.add_argument(
        "--load_ideas",
        type=str,
        default="ideas/msd_1.json",
        help="Path to a JSON file containing pregenerated ideas",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        # default="<path_to_checkpoint.pkl>",
        help="Path to a checkpoint.pkl file to resume a previous experiment.",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to run",
    )
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="If set, add a HF dataset reference to the idea",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts to try",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID, used to distinguish same idea in different attempts in parallel runs",
    )
    # The model is now read from the llm_roles section in config.yaml
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation rounds to perform",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="If set, skip the writeup process",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="If set, skip the review process",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--innovation-queue",
        type=str,
        default="innovations.json",
        help="Path to JSON file containing initial innovation ideas",
    )
    parser.add_argument(
        "--debug-baseline",
        action="store_true",
        help="Debug mode: set all baseline metrics to dice=0, hd95=200 to easily pass all stages",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level). Default is INFO level.",
    )
    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def find_pdf_path_for_review(idea_dir):
    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    
    # First check if there are elsevier format PDFs
    elsevier_pdfs = [f for f in pdf_files if "elsevier" in f.lower()]
    if elsevier_pdfs:
        return osp.join(idea_dir, elsevier_pdfs[0])
    
    # Then check reflection PDFs
    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if reflection_pdfs:
        # First check if there's a final version
        final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
        if final_pdfs:
            # Use the final version if available
            return osp.join(idea_dir, final_pdfs[0])
        else:
            # Try to find numbered reflections
            reflection_nums = []
            for f in reflection_pdfs:
                match = re.search(r"reflection[_.]?(\d+)", f)
                if match:
                    reflection_nums.append((int(match.group(1)), f))

            if reflection_nums:
                # Get the file with the highest reflection number
                highest_reflection = max(reflection_nums, key=lambda x: x[0])
                return osp.join(idea_dir, highest_reflection[1])
            else:
                # Fall back to the first reflection PDF if no numbers found
                return osp.join(idea_dir, reflection_pdfs[0])
    
    # If no suitable PDF found, return any PDF file
    if pdf_files:
        return osp.join(idea_dir, pdf_files[0])
    
    # If no PDF files at all, return None
    print(f"Warning: No PDF files found in {idea_dir}")
    return None


@contextmanager
def redirect_stdout_stderr_to_file(log_file_path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log = open(log_file_path, "a")
    sys.stdout = log
    sys.stderr = log
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log.close()


def _print_success_banner(idea_dir: str | None) -> None:
    report_path = None
    if idea_dir:
        for candidate in ("logs/0-run/experiment_report.md", "experiment_report.md"):
            p = os.path.join(idea_dir, candidate)
            if os.path.exists(p):
                report_path = p
                break
    bar = "=" * 78
    print()
    print(bar)
    print(bar)
    print("✅  EXPERIMENT COMPLETED SUCCESSFULLY")
    print(bar)
    print(f"   Experiment dir : {idea_dir or '(unknown)'}")
    if report_path:
        print(f"   Report         : {report_path}")
    print("")
    print("   Note: any `OSError: libavutil.so.*` / `FFmpeg extension is not")
    print("   available` tracebacks below are harmless — torio tries to load")
    print("   ffmpeg on exit, the errors are swallowed internally, and do")
    print("   NOT affect the experiment. The pipeline finished OK.")
    print(bar)
    print(bar)
    print()


def _print_failure_banner(idea_dir: str | None, err: BaseException) -> None:
    import traceback as _tb
    bar = "=" * 78
    print()
    print(bar)
    print(bar)
    print("❌  EXPERIMENT FAILED")
    print(bar)
    print(f"   Experiment dir : {idea_dir or '(unknown)'}")
    print(f"   Error type     : {type(err).__name__}")
    print(f"   Error message  : {err}")
    print(bar)
    print(bar)
    _tb.print_exc()


def _run_main():
    args = parse_arguments()
    paper_config_path = args.config

    # Reconfigure log level based on --verbose flag
    import logging as _logging
    _log_level = _logging.DEBUG if args.verbose else _logging.INFO
    for _name in [None, "camyla", "camyla.treesearch", "camyla.llm"]:
        _l = _logging.getLogger(_name)
        _l.setLevel(_log_level)
    os.environ["CAMYLA_LOG_LEVEL"] = "DEBUG" if args.verbose else "INFO"

    # Load innovation queue path into environment for AgentManager
    os.environ["INNOVATION_QUEUE"] = args.innovation_queue

    # Set environment variables
    os.environ["CAMYLA_ROOT"] = os.path.dirname(os.path.abspath(__file__))
    print(f"CAMYLA_ROOT={os.environ['CAMYLA_ROOT']}")

    if args.debug_baseline:
        print("DEBUG MODE: Baseline metrics will be set to dice=0, hd95=200")

    available_gpus = get_available_gpus()
    print(f"Using GPUs: {available_gpus}")

    # Initialize idea_dir variable
    idea_dir = None

    # Load checkpoint
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        # Set environment variable so AgentManager.__setstate__ can get checkpoint path
        os.environ["QWBE_CHECKPOINT_PATH"] = args.resume_from_checkpoint
        checkpoint_path = Path(args.resume_from_checkpoint).resolve()
        # Search upward from checkpoint path until finding parent directory of "logs" directory
        current_path = checkpoint_path
        while current_path.name != 'logs' and current_path.parent != current_path:
            current_path = current_path.parent
        
        if current_path.name == 'logs':
            idea_dir = str(current_path.parent)
            print(f"Experiment directory: {idea_dir}")
        else:
            # If not found, fall back to the directory containing the checkpoint file
            idea_dir = str(checkpoint_path.parent)
            print(f"Warning: Could not determine experiment directory. Using: {idea_dir}")
        
        # Read exp_name from idea.json or extract it from the path
        idea_json_path = Path(idea_dir) / "idea.json"
        if idea_json_path.exists():
            with open(idea_json_path, "r") as f:
                idea_data = json.load(f)
                if "exp_name" in idea_data:
                    exp_name = idea_data["exp_name"]
                    print(f"exp_name={exp_name} (from idea.json)")
                else:
                    # Fallback: extract from the path
                    exp_name = Path(idea_dir).name
                    print(f"exp_name={exp_name} (from path)")
                    idea_data["exp_name"] = exp_name
                    with open(idea_json_path, "w") as f_write:
                        json.dump(idea_data, f_write, indent=4)
        else:
            # idea.json does not exist; fall back to extracting from the path
            exp_name = Path(idea_dir).name
            print(f"exp_name={exp_name} (idea.json not found, from path)")

        # Resume experiment from checkpoint
        perform_experiments_qwbe(
            config_path=args.config,
            resume_checkpoint_path=args.resume_from_checkpoint
        )
        resume_cfg = Path(idea_dir) / "config.yaml" if idea_dir else None
        if resume_cfg and resume_cfg.exists():
            paper_config_path = str(resume_cfg)
    else:
        # Load pre-generated ideas
        with open(args.load_ideas, "r") as f:
            loaded_data = json.load(f)

        # Only the new structure (v2.0) is supported
        if isinstance(loaded_data, list):
            ideas = loaded_data
            print(f"Loaded {len(ideas)} ideas from {args.load_ideas}")
        else:
            # Single idea object
            ideas = [loaded_data]
            print(f"Loaded 1 idea from {args.load_ideas}")

        idea = ideas[args.idea_idx]

        # Derive the idea name from the dataset name
        idea_name = idea["dataset"]["name"].replace(" ", "_").lower()

        # Create experiment directory
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        idea_dir = f"experiments/{date}_{idea_name}_attempt_{args.attempt_id}"
        print(f"Output dir: {idea_dir}")
        os.makedirs(idea_dir, exist_ok=True)

        exp_name = f"{date}_{idea_name}_attempt_{args.attempt_id}"
        print(f"exp_name={exp_name}")

        os.environ['CAMYLA_IDEA_NAME'] = idea_name

        # Process the code section of the idea
        idea_path_md = osp.join(idea_dir, "idea.md")
        code = None

        # Use the generic config system to locate files
        try:
            from skills import FrameworkLoader
            
            framework_loader = FrameworkLoader()
            
            # Locate the .py code file
            code_path, code_source = framework_loader.find_code_template()
            if code_path:
                with open(code_path, "r") as f:
                    code = f.read()
                print(f"Code template: {code_path}")
            else:
                print(f"Warning: No code file found for framework")

            doc_path, doc_source = framework_loader.find_documentation()
            if doc_path:
                print(f"Documentation: {doc_path}")
                    
        except ImportError as e:
            print(f"Error: docs.config module not available: {e}")
            sys.exit(1)
        
        # Convert idea to markdown
        idea_to_markdown(ideas[args.idea_idx], idea_path_md, code_path if code else None)

        # Add dataset info to idea to ensure it's passed to aider
        dataset_info = ideas[args.idea_idx].get("dataset", {})
        if dataset_info:
            dataset_id = dataset_info.get("dataset_id")
            configuration = dataset_info.get("configuration", "3d_fullres")
            dataset_name = dataset_info.get("name", f"Dataset {dataset_id}")
            task_description = dataset_info.get("description", "Medical segmentation task")
            
            # Add baseline info to idea
            baseline_requirements = f"""Dataset: {dataset_name}
Task: {task_description}
Baseline: nnUNet Baseline
Requirements: Use camylanet framework for nnUNet implementation; Follow standard preprocessing and training pipeline; Use dataset_id={dataset_id} and configuration='{configuration}'; Implement basic training, evaluation, and result reporting; Ensure reproducible results with proper metrics reporting"""
            
            ideas[args.idea_idx]["baseline_info"] = baseline_requirements
            print(f"Dataset: id={dataset_id}, config={configuration}")

        # Handle dataset reference code
        dataset_ref_code = None
        if args.add_dataset_ref:
            dataset_ref_path = "hf_dataset_reference.py"
            if os.path.exists(dataset_ref_path):
                with open(dataset_ref_path, "r") as f:
                    dataset_ref_code = f.read()
            else:
                print(f"Warning: Dataset reference file {dataset_ref_path} not found")

        # Merge code (if needed)
        if dataset_ref_code is not None or code is not None:
            added_code = ""
            if dataset_ref_code:
                added_code += dataset_ref_code + "\n"
            if code:
                added_code += code

            # Add code to idea for passing to experiment system
            ideas[args.idea_idx]["loaded_code"] = added_code

        ideas[args.idea_idx]["exp_name"] = exp_name
            
        # Save idea in JSON format
        idea_path_json = osp.join(idea_dir, "idea.json")
        with open(idea_path_json, "w") as f:
            json.dump(ideas[args.idea_idx], f, indent=4)

        # Prepare and run experiment
        config_path = args.config
        extra_config = {}
        if args.debug_baseline:
            extra_config["debug_baseline"] = True
        idea_config_path = edit_qwbe_config_file(
            config_path,
            idea_dir,
            idea_path_json,
            extra_config=extra_config if extra_config else None,
        )
        paper_config_path = idea_config_path

        ensure_baseline(dataset_id)

        perform_experiments_qwbe(idea_config_path)

        # Copy experiment results
        experiment_results_dir = osp.join(idea_dir, "logs/0-run/experiment_results")
        if os.path.exists(experiment_results_dir):
            shutil.copytree(
                experiment_results_dir,
                osp.join(idea_dir, "experiment_results"),
                dirs_exist_ok=True,
            )


        # shutil.rmtree(osp.join(idea_dir, "experiment_results"))

        save_token_tracker(idea_dir)

    '''
    # Old writeup and preview process - REMOVED/TRANSFERRED to Paper Agent
    if not args.skip_writeup:
        print("Legacy writeup skipped. Use Paper Agent instead.")
        
    save_token_tracker(idea_dir)
    
    if not args.skip_review and not args.skip_writeup:
        print("Legacy review skipped. Use Paper Agent instead.")
    '''
    
    # ===== NEW PAPER AGENT INTEGRATION =====
    if not args.skip_writeup and generate_paper:
        print("\nStarting Paper Agent...")
        try:
            # Determine debug citation mode (maybe reuse num_cite_rounds=0 logic or add flag?)
            # For robustness, we default to False unless specified via args (but args don't have it)
            # We can use skip_review as a proxy only if intended, but let's default to False (use API)
            
            success = generate_paper(
                experiment_dir=idea_dir,
                debug_citations=False, 
                output_dir=None, # Use default
                config_path=paper_config_path,
            )
            if success:
                print("Paper Agent: completed successfully")
            else:
                print("Paper Agent: failed (check logs)")
        except Exception as e:
            print(f"Paper Agent error: {e}")
            import traceback
            traceback.print_exc()

    return idea_dir


if __name__ == "__main__":
    _idea_dir = None
    try:
        _idea_dir = _run_main()
    except BaseException as _err:
        _print_failure_banner(_idea_dir, _err)
        sys.exit(1)
    else:
        _print_success_banner(_idea_dir)
