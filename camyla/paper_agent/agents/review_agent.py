from .base_agent import BaseAgent
import logging
import json_repair
import re

logger = logging.getLogger(__name__)


class ReviewAgent(BaseAgent):
    """
    Agent responsible for reviewing section quality and providing expansion suggestions.
    """
    
    def run(self, section_name: str, content: str) -> dict:
        """
        Evaluate section content quality based on depth rather than word count.
        
        Args:
            section_name: Name of the section
            content: LaTeX content of the section
            
        Returns:
            Dictionary with:
                - approved: bool
                - issues: list of identified problems
                - suggestions: str (detailed improvement suggestions)
        """
        logger.info(f"Reviewing section: {section_name}...")

        lower_name = section_name.lower()
        if "method" in lower_name:
            forbidden_patterns = [
                r"\boptimizer\b",
                r"\blearning rate\b",
                r"\bscheduler\b",
                r"\bbatch size\b",
                r"\bepochs?\b",
                r"\bmomentum\b",
                r"\bweight decay\b",
                r"\btrain/test split\b",
                r"\btrain[- ]test split\b",
                r"\bhardware\b",
                r"\bgpu\b",
            ]
            hits = []
            for pattern in forbidden_patterns:
                if re.search(pattern, content, flags=re.IGNORECASE):
                    hits.append(pattern.replace(r"\b", "").replace("\\", ""))
            if hits:
                return {
                    "approved": False,
                    "issues": [f"Method section contains training recipe details: {', '.join(hits)}"],
                    "suggestions": (
                        "Remove optimizer, scheduler, batch size, epochs, hardware, and other "
                        "training protocol details from the Method section. Keep only "
                        "architectural design, equations, module behavior, and data flow."
                    ),
                }
        
        # Use LLM to evaluate content quality
        prompt = self.load_skill(
            "common/review_section.md",
            section_name=section_name,
            content=content
        )
        
        review_response = self.chat(messages=[{"role": "user", "content": prompt}])
        
        # Parse LLM review result (expecting JSON format)
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', review_response, re.DOTALL)
            if json_match:
                result = json_repair.loads(json_match.group(1))
            else:
                result = json_repair.loads(review_response)

            assert isinstance(result, dict), "Parsed JSON is not a dictionary!"
            
            return {
                "approved": result.get("approved", False),
                "issues": result.get("issues", []),
                "suggestions": result.get("suggestions", "")
            }
        except Exception as e:
            logger.error(f"Failed to parse review result: {e}")
            # Conservative strategy: if parsing fails, assume improvement needed
            return {
                "approved": False,
                "issues": ["Review parsing failed"],
                "suggestions": review_response
            }
