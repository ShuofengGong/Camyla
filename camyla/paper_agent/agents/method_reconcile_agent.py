import logging
import re

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MethodReconcileAgent(BaseAgent):
    """Reconcile proposal methodology details against the implemented code once."""

    def run(self, research_idea: str, method_code: str) -> str:
        logger.info("Reconciling methodology details with implemented code...")

        prompt = self.load_skill(
            "medical_segmentation/reconcile_methodology.md",
            research_idea=research_idea,
            method_code=method_code,
        )
        response = self.chat(messages=[{"role": "user", "content": prompt}])

        cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", response.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()
