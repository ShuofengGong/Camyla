"""
Innovation queue + literature search integration module (refactored).
Integrates Research Proposal generation driven by literature search into AgentManager.

Key changes:
- Use the full Proposal format instead of the legacy innovation format.
- On failure, refine the Proposal (remove/modify modules) rather than generating a new innovation.
- Pass full Proposal data through to downstream consumers.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from .innovation_generator import InnovationGenerator
except ImportError:
    try:
        from innovation_generator import InnovationGenerator
    except ImportError:
        InnovationGenerator = None

logger = logging.getLogger(__name__)


class ProposalQueueManager:
    """
    Research Proposal queue manager.

    Responsibilities:
    - Initialize the proposal queue.
    - Refine a proposal on failure.
    - Save/load queue state.
    """

    def __init__(self, agent_manager, enable_literature_search: bool = True):
        """
        Initialize the Proposal queue manager.

        Args:
            agent_manager: AgentManager instance.
            enable_literature_search: whether to enable literature-search-driven innovation generation.
        """
        self.agent_manager = agent_manager
        self.enable_literature_search = enable_literature_search

        # Initialize the innovation generator
        if enable_literature_search and InnovationGenerator is not None:
            try:
                from camyla.model_config import get_role
                config = getattr(agent_manager, 'cfg', None)
                self.innovation_generator = InnovationGenerator(
                    model_backbone=get_role("literature_backbone")["model"],
                    openai_api_key=getattr(agent_manager, 'openai_api_key', None),
                    verbose=True,
                    config=config
                )
            except Exception as e:
                logger.warning(f"Failed to initialize innovation generator: {e}")
                self.innovation_generator = None
        else:
            self.innovation_generator = None

        # Store full proposal data
        self.proposals: List[Dict[str, Any]] = []
        self.research_context: Optional[Dict[str, str]] = None

    def initialize_proposal_queue(
        self,
        task_desc: Dict[str, Any],
        num_proposals: int = 5,
        output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Initialize the Proposal queue (literature-search driven).

        Args:
            task_desc: task description (idea.json content).
            num_proposals: number of proposals to generate.
            output_dir: directory to save proposal files.

        Returns:
            The full proposal list.

        Raises:
            RuntimeError: if proposal generation fails.
        """
        # Ensure literature search is enabled
        if not self.enable_literature_search:
            raise RuntimeError(
                "❌ Literature search is disabled (enable_literature_search=False). "
                "Proposal generation requires the literature-search feature."
            )

        if not self.innovation_generator:
            raise RuntimeError(
                "❌ InnovationGenerator failed to initialize. "
                "Check the model configuration and API key."
            )

        logger.info("🔬 Generating Research Proposals via literature search")

        # Resolve output directory — absolute path to avoid CWD issues.
        if output_dir is None:
            workspace = getattr(self.agent_manager, 'workspace_dir', None)
            if workspace:
                output_dir = Path(workspace).resolve() / "research_proposals"
            else:
                output_dir = Path.cwd().resolve() / "research_proposals"
        else:
            output_dir = Path(output_dir).resolve()

        print(f"🔍 Proposal output directory: {output_dir}")
        print(f"🔍 Current working directory: {Path.cwd()}")

        # Generate diverse proposals
        proposal_metadata = self.innovation_generator.generate_diverse_proposals(
            idea_data=task_desc,
            num_proposals=num_proposals,
            output_dir=output_dir
        )

        # Cache the research context
        self.research_context = self.innovation_generator._extract_research_context(task_desc)

        if not proposal_metadata:
            raise RuntimeError(
                "❌ Proposal generation failed: generate_diverse_proposals returned empty. "
                "Check literature search configuration and network connectivity."
            )

        # Rebuild the full proposal list from metadata
        self.proposals = self._load_proposals_from_metadata(proposal_metadata)

        if not self.proposals:
            raise RuntimeError(
                "❌ Proposal loading failed: all generated Proposal MD files are invalid. "
                "Check the proposal output directory and file permissions."
            )

        logger.info(f"✅ Successfully generated {len(self.proposals)} Research Proposals")
        return self.proposals

    def _load_proposals_from_metadata(
        self,
        metadata_list: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Load proposal data from metadata.

        🆕 Improvement: persists the full proposal data (modules, motivation, etc.)
        so that full content can be recovered even if the MD file is lost.

        Args:
            metadata_list: metadata returned by generate_diverse_proposals.

        Returns:
            Proposal list (includes full content for recovery).
        """
        proposals = []

        for i, meta in enumerate(metadata_list):
            md_file_path = meta.get("md_file_path", "")

            # Validate MD file path
            if not md_file_path:
                logger.warning(f"Proposal {i} missing md_file_path; skipped")
                continue

            if not os.path.exists(md_file_path):
                logger.warning(f"MD file for proposal {i} does not exist: {md_file_path}; skipped")
                continue

            # 🆕 Store the full proposal data (for recovery)
            proposal = {
                "title": meta.get("title", "Unknown Proposal"),
                "md_file": md_file_path,
                "challenge_theme": meta.get("core_theme", ""),
                "generator": meta.get("generator", ""),
                # 🆕 New: retain full data for recovery
                "full_data": {
                    "motivation": meta.get("motivation", {}),
                    "modules": meta.get("modules", []),
                    "integration": meta.get("integration", ""),
                    "contributions": meta.get("contributions", []),
                    "source_challenge": meta.get("source_challenge", ""),
                    "core_theme": meta.get("core_theme", ""),
                },
            }

            proposals.append(proposal)
            logger.info(f"✅ Loaded Proposal {i}: {proposal['title']} (full_data preserved)")

        return proposals

    def refine_proposal_on_failure(
        self,
        proposal_idx: int,
        failure_info: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """
        Refine a Proposal after it has failed.

        Reads the MD content directly and hands it to the LLM for refinement without parsing.

        Refinement policy:
        - Remove modules that caused errors.
        - Simplify overly complex modules.
        - Do not add new modules (to prevent regressions).

        Args:
            proposal_idx: index of the failed proposal.
            failure_info: failure details:
                - error_type: error type
                - error_message: error message
                - execution_time: execution time
                - performance_issues: description of performance issues

        Returns:
            The refined proposal (with updated md_file path), or None on failure.
        """
        if proposal_idx < 0 or proposal_idx >= len(self.proposals):
            logger.error(f"Invalid proposal index: {proposal_idx}")
            return None

        failed_proposal = self.proposals[proposal_idx]

        if not self.innovation_generator:
            logger.warning("InnovationGenerator unavailable; cannot refine proposal")
            return None

        # Get the MD file path
        md_file = failed_proposal.get("md_file", "")
        if not md_file:
            logger.warning(f"Proposal {proposal_idx} has no MD file path; cannot refine")
            return None

        # Read the MD file content
        if not os.path.exists(md_file):
            logger.warning(f"Proposal MD file does not exist: {md_file}")
            return None

        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()

            if not md_content or len(md_content.strip()) < 100:
                logger.warning(f"Proposal MD file content too short: {md_file}")
                return None

            logger.info(f"📄 Loaded proposal content from MD file: {md_file}")

            # Call the refinement API — pass the MD content directly
            refined_content = self.innovation_generator.refine_proposal_from_md_content(
                md_content=md_content,
                proposal_title=failed_proposal.get("title", "Unknown"),
                failure_info=failure_info
            )

            if refined_content:
                # Save the refined content to a new file
                refinement_count = len(failed_proposal.get("refinement_history", [])) + 1
                refined_path = md_file.replace(".md", f"_refined_v{refinement_count}.md")

                with open(refined_path, 'w', encoding='utf-8') as f:
                    f.write(refined_content)

                logger.info(f"✅ Refined Proposal saved to: {refined_path}")

                # 🆕 Update proposal info
                refined_proposal = {
                    "title": failed_proposal.get("title", "Unknown") + f" (Refined v{refinement_count})",
                    "md_file": refined_path,  # Point to the new file
                    "challenge_theme": failed_proposal.get("challenge_theme", ""),
                    "generator": f"{failed_proposal.get('generator', 'unknown')}_refined",
                    "refinement_history": failed_proposal.get("refinement_history", []) + [failure_info]
                }

                # 🆕 Append the refined proposal to the end of the queue rather than replacing in place.
                # The proposal list grows: proposal_1, proposal_2, proposal_3, proposal_4, ...
                refined_proposal_idx = len(self.proposals)  # new proposal index
                self.proposals.append(refined_proposal)  # append to the end
                logger.info(f"✅ Proposal {proposal_idx} refined successfully; appended as proposal_{refined_proposal_idx + 1}")

                return refined_proposal  # return the appended proposal

                return refined_proposal
            else:
                logger.warning(f"Proposal {proposal_idx} refinement failed")
                return None

        except Exception as e:
            logger.error(f"Error while refining proposal: {e}")
            return None

    def get_proposal(self, idx: int) -> Optional[Dict[str, Any]]:
        """Return the proposal at the given index."""
        if 0 <= idx < len(self.proposals):
            return self.proposals[idx]
        return None

    def get_proposal_for_prompt(self, idx: int) -> str:
        """
        Return a formatted proposal suitable for an OpenHands prompt.

        Reads the MD file directly with no fallback. Raises if the file is missing or unreadable.

        Returns:
            Full Markdown-formatted proposal content.

        Raises:
            ValueError: if the proposal index is invalid.
            FileNotFoundError: if the MD file does not exist.
            IOError: if the MD file cannot be read.
        """
        proposal = self.get_proposal(idx)
        if not proposal:
            raise ValueError(f"Proposal index {idx} is invalid or proposal not found")

        # Get the MD file path
        md_file = proposal.get("md_file", "")

        if not md_file:
            raise ValueError(
                f"Proposal {idx} does not have a valid MD file path. "
                f"Proposal data: title={proposal.get('title', 'Unknown')}"
            )

        # Check that the file exists
        if not os.path.exists(md_file):
            raise FileNotFoundError(
                f"MD file not found for proposal {idx}: {md_file}\n"
                f"Proposal title: {proposal.get('title', 'Unknown')}\n"
                f"Please ensure the proposal MD file exists."
            )

        # Read the MD file
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise IOError(
                f"Failed to read MD file for proposal {idx}: {md_file}\n"
                f"Error: {e}"
            )

        # Ensure the content is non-empty
        if not content or len(content.strip()) < 100:
            raise ValueError(
                f"MD file content is empty or too short for proposal {idx}: {md_file}\n"
                f"Content length: {len(content) if content else 0} characters"
            )

        logger.info(f"✅ Successfully loaded proposal {idx} from MD file: {md_file}")
        return content

    def get_statistics(self) -> Dict[str, Any]:
        """Return queue statistics."""
        return {
            "total_proposals": len(self.proposals),
            "literature_search_enabled": self.enable_literature_search,
            "has_research_context": self.research_context is not None,
            "innovation_generator_available": self.innovation_generator is not None,
            "refined_proposals": sum(1 for p in self.proposals if "_refined" in p.get("generator", ""))
        }

    def save_state(self, filepath: str):
        """Save queue state."""
        try:
            state = {
                "proposals": self.proposals,
                "research_context": self.research_context,
                "enable_literature_search": self.enable_literature_search,
                "statistics": self.get_statistics()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"✅ Proposal queue state saved to: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save queue state: {e}")

    def load_state(self, filepath: str) -> bool:
        """Load queue state."""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"State file does not exist: {filepath}")
                return False

            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)

            self.proposals = state.get("proposals", [])
            self.research_context = state.get("research_context")
            self.enable_literature_search = state.get("enable_literature_search", True)

            logger.info(f"✅ Proposal queue state restored from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load queue state: {e}")
            return False


# ============================================================================
# Backward-compatible alias and factory function
# ============================================================================

# Alias kept for backward compatibility
InnovationQueueManager = ProposalQueueManager


def integrate_proposal_queue_with_agent_manager(
    agent_manager,
    enable_literature_search: bool = True
) -> ProposalQueueManager:
    """
    Integrate the Proposal queue manager into an AgentManager.

    Args:
        agent_manager: AgentManager instance.
        enable_literature_search: whether to enable literature search.

    Returns:
        ProposalQueueManager instance.
    """
    proposal_manager = ProposalQueueManager(agent_manager, enable_literature_search)

    # Attach to the AgentManager
    agent_manager.proposal_queue_manager = proposal_manager

    logger.info(f"✅ Proposal queue manager integrated into AgentManager (literature search: {'enabled' if enable_literature_search else 'disabled'})")

    return proposal_manager

