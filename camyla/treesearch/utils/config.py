"""configuration and setup utils"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, cast, Literal, Optional, Dict, Any, Union, List

import os
import sys
import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from . import tree_export
from . import copytree, serialize

shutup.mute_warnings()

def _build_log_handler():
    """Use RichHandler only in interactive terminals; plain StreamHandler when piped."""
    if sys.stdout.isatty():
        return RichHandler(rich_tracebacks=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    return handler

_log_level = os.environ.get("CAMYLA_LOG_LEVEL", "WARNING").upper()

logging.basicConfig(
    level=_log_level, format="%(message)s", datefmt="[%X]", handlers=[_build_log_handler()]
)
logger = logging.getLogger("camyla")
logger.setLevel(_log_level)


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class CodeConfig:
    """Code generation configuration for experiment stages.

    candidates: list of endpoint names drawn from llm_endpoints;
      - length 1 -> single model
      - length > 1 -> Stage 2 multi-way competition
    """
    candidates: List[str] = field(default_factory=list)
    max_tokens: int = 16384


@dataclass
class Stage2UCBConfig:
    """Stage 2 branch-level (hierarchical) PUCT tree search configuration.

    Phase 1 treats each baseline child-branch as an arm in a multi-armed
    bandit, with a special "open new branch" action whose cost grows with
    the number of existing branches.  Once any node beats baseline, Phase 2
    (depth-first on winning branch) takes over — unchanged from before.
    """
    enabled: bool = True
    c_puct: float = 1.5
    # Q normalization when metric < baseline: Q = -gap^q_below_exponent.
    # exponent < 1 strengthens penalty (e.g. 0.5); exponent == 1 is linear (legacy).
    q_below_exponent: float = 0.5
    # Buggy node Q: inherit nearest ancestor metric, then subtract this penalty.
    # 0 = same Q as ancestor; larger values make buggy nodes less attractive.
    buggy_q_penalty: float = 0.2
    # Prior power: P(Q) = max(0, 1+Q)^prior_power.  Higher values make the
    # algorithm more risk-averse (branches far below baseline get less
    # exploration).  Fixed at 3 by design — not intended for tuning.
    prior_power: int = 3


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int
    stage2_ucb: Stage2UCBConfig = field(default_factory=Stage2UCBConfig)


@dataclass
class ChallengeDiscoveryConfig:
    """Challenge-discovery configuration used for challenge-driven core_theme generation."""
    enabled: bool = True
    max_challenges: int = 3
    final_challenges: int = 3
    iterations: int = 6
    challenges_per_round: int = 3


@dataclass
class TargetPapersConfig:
    phase1: int = 6
    phase2: int = 6


@dataclass
class MinYearConfig:
    phase1: str = "2015-01-01"
    phase2: str = "2021-01-01"


@dataclass
class LiteratureSearchConfig:
    max_papers_per_search: int = 20
    target_papers: TargetPapersConfig = field(default_factory=TargetPapersConfig)
    max_iterations: int = 20
    min_year: MinYearConfig = field(default_factory=MinYearConfig)
    sources: List[str] = field(default_factory=lambda: ["semantic_scholar"])
    filter_open_access: bool = True
    enable_randomization: bool = True
    challenge_discovery: ChallengeDiscoveryConfig = field(default_factory=ChallengeDiscoveryConfig)


@dataclass 
class IdeaGeneratorConfig:
    name: str
    model: str
    temperature: float
    personality: str
    max_tokens: int = 4096


@dataclass
class EvaluationCriteriaConfig:
    weight: float
    description: str


@dataclass
class ScoringConfig:
    min: float = 0.0
    max: float = 10.0
    precision: int = 1


@dataclass
class AssessmentConfig:
    temperature: float = 1.0
    max_tokens: int = 8192
    criteria: Dict[str, EvaluationCriteriaConfig] = field(default_factory=dict)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


@dataclass
class IdeaGenConfig:
    """Top-level idea_generation section (Phase 1-3)."""
    enabled: bool = True
    initial_queue_size: int = 1
    literature_search: LiteratureSearchConfig = field(default_factory=LiteratureSearchConfig)
    idea_generators: list[IdeaGeneratorConfig] = field(default_factory=list)
    assessment: AssessmentConfig = field(default_factory=AssessmentConfig)
    research_proposal: "ResearchProposalConfig" = field(default_factory=lambda: ResearchProposalConfig())


@dataclass
class Stage2Config:
    max_iterations_per_innovation: int = 5


@dataclass
class Stage3Config:
    """Stage 3 Ablation Studies configuration"""
    max_ablations: int = 6
    prioritize_core_innovations: bool = True


@dataclass
class OpenHandsLLMConfig:
    timeout: int = 360
    max_output_tokens: int = 32768
    temperature: float = 0.9


@dataclass
class OpenHandsCondenserConfig:
    enabled: bool = True
    max_size: int = 80
    keep_first: int = 2


@dataclass
class OpenHandsConfig:
    python_path: str = "/opt/conda/envs/py310/bin/python"
    pytest_path: str = "/opt/conda/envs/py310/bin/pytest"
    max_iterations: int = 100
    llm: OpenHandsLLMConfig = field(default_factory=OpenHandsLLMConfig)
    condenser: OpenHandsCondenserConfig = field(default_factory=OpenHandsCondenserConfig)


@dataclass
class DebugConfig:
    stage3: bool  # ablation studies debug flag


@dataclass
class ResearchProposalConfig:
    """Research proposal generation configuration"""
    num_proposals: int = 5                    # Number of diverse proposals to generate
    modules_per_proposal: int = 3             # Target number of modules per proposal (2-4)
    duplicate_check: bool = True              # Enable LLM-based duplicate checking
    output_dir: str = "research_proposals"    # Directory to save proposal .md files


@dataclass
class ProposalRefinementConfig:
    """Proposal inline refinement configuration (on-the-fly tuning inside a substage)."""
    enabled: bool = True                      # whether to enable on-the-fly refinement inside a substage
    metric_threshold: float = 0.05            # Dice score threshold (how far below baseline triggers diagnosis)
    max_refinements_per_substage: int = 2     # max refinements per substage
    term_out_tail_chars: int = 1000           # term_out trimming length (tail characters)
    diagnostic_model: Optional[str] = None    # LLM model used for diagnosis (None = reuse the feedback model)
    diagnostic_temperature: float = 0.3       # temperature during diagnosis


@dataclass
class ExperimentConfig:
    """Top-level experiment section (Stage 1-3)."""
    num_workers: int = 1
    limit_workers_to_gpus: bool = True
    steps: int = 5

    stages: Dict[str, int] = field(default_factory=lambda: {
        "stage1_max_iters": 6, "stage2_max_iters": 3, "stage3_max_iters": 15
    })
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)

    code: CodeConfig = field(default_factory=CodeConfig)
    openhands: OpenHandsConfig = field(default_factory=OpenHandsConfig)
    search: Optional[SearchConfig] = None
    proposal_refinement: ProposalRefinementConfig = field(default_factory=ProposalRefinementConfig)


@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool
    # Conda environment configuration
    use_conda: bool = True
    conda_env: str = "py310"


@dataclass
class Config(Hashable):
    data_dir: Path
    desc_file: Optional[Path]

    log_dir: Optional[Path] = None
    workspace_dir: Optional[Path] = None

    copy_data: bool = True
    exp_name: str = "run"
    generate_report: bool = True
    metric_tiebreak_threshold: float = 0.005
    debug_baseline: bool = False

    exec: ExecConfig = field(default_factory=lambda: ExecConfig(
        timeout=7200, agent_file_name="runfile.py", format_tb_ipython=False
    ))
    debug: DebugConfig = field(default_factory=lambda: DebugConfig(stage3=False))

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    idea_generation: IdeaGenConfig = field(default_factory=IdeaGenConfig)

    # The following fields are consumed by model_config.py through get_endpoint / get_role / get_api_key.
    default_endpoint: Optional[str] = None
    llm_endpoints: Optional[Dict[str, Any]] = None
    llm_roles: Optional[Dict[str, Any]] = None
    api_keys: Optional[Dict[str, Any]] = None


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if (current_index := int(p.name.split("-")[0])) > max_index:
                max_index = current_index
        except ValueError:
            pass
    print("max_index: ", max_index)
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=False
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None:
        raise ValueError(
            "`desc_file` must be provided (path to the task description file)."
        )

    if cfg.log_dir is None:
        raise ValueError(
            "`log_dir` must be provided (top-level log output directory). "
            "The launcher normally sets this from the experiment dir — "
            "if you're calling prep_cfg directly, set cfg.log_dir first."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    # 🔧 Directory merge: workspace_dir and log_dir are now unified into the same directory.
    # A separate top_workspace_dir is no longer created.

    # generate experiment name and prefix with consecutive index
    ind = _get_next_logindex(top_log_dir)
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.exp_name = f"{ind}-{cfg.exp_name}"

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = cfg.log_dir  # 🔧 Merge workspace into log directory.

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from the desc_file."""
    if cfg.desc_file is None:
        raise ValueError("`desc_file` must be provided.")

    with open(cfg.desc_file) as f:
        return f.read()


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)


def save_run(cfg: Config, journal, stage_name: str = None):
    if stage_name is None:
        stage_name = "NoStageRun"
    save_dir = cfg.log_dir / stage_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    try:
        serialize.dump_json(journal, save_dir / "journal.json")
    except Exception as e:
        print(f"Error saving journal: {e}")
        raise
    # save config
    try:
        OmegaConf.save(config=cfg, f=save_dir / "config.yaml")
    except Exception as e:
        print(f"Error saving config: {e}")
        raise
    # create the tree + code visualization
    try:
        tree_export.generate(cfg, journal, save_dir / "tree_plot.html")
    except Exception as e:
        print(f"Error generating tree: {e}")
        raise
    # save the best found solution
    try:
        best_node = journal.get_best_node(only_good=False)
        if best_node is not None:
            for existing_file in save_dir.glob("best_solution_*.py"):
                existing_file.unlink()
            # Create new best solution file
            filename = f"best_solution_{best_node.id}.py"
            with open(save_dir / filename, "w") as f:
                f.write(best_node.code)
            # save best_node.id to a text file
            with open(save_dir / "best_node_id.txt", "w") as f:
                f.write(str(best_node.id))
        else:
            print("No best node found yet")
    except Exception as e:
        print(f"Error saving best solution: {e}")
