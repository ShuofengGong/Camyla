from typing import Dict, Any
import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class MetadataAgent(BaseAgent):
    """
    Agent responsible for generating paper metadata (title, author, abstract, keywords).
    """
    
    def run(self, research_idea: str, template_config: Dict[str, Any], dataset_context: str = "") -> Dict[str, str]:
        """
        Generate metadata based on research idea and template configuration.
        
        Args:
            research_idea: The research idea/proposal text
            template_config: The template configuration dictionary (from template.json)
            dataset_context: Detailed dataset information to prevent LLM hallucination
            
        Returns:
            Dictionary containing generated metadata (title, abstract, keywords, etc.)
        """
        logger.info("Generating paper metadata...")
        
        # Extract metadata requirements from template config
        metadata_fields = template_config.get("metadata_fields", {})
        
        # Filter to only generate fields with auto_generate=true (or missing flag)
        fields_to_generate = {
            k: v for k, v in metadata_fields.items() 
            if v.get("auto_generate", True) is not False
        }
        
        logger.info(f"Fields to generate: {list(fields_to_generate.keys())}")
        logger.info(f"Skipping fields: {[k for k, v in metadata_fields.items() if not v.get('auto_generate', True)]}")
        
        prompt = self.load_skill(
            "medical_segmentation/metadata.md",
            research_idea=research_idea,
            metadata_requirements=str(fields_to_generate),
            dataset_context=dataset_context  # pass dataset context
        )
        
        response = self.chat(messages=[{"role": "user", "content": prompt}])
        
        # Parse the response (expecting JSON-like or structured output)
        # For now, we'll assume the LLM returns a JSON block or we parse it.
        # To keep it robust, let's ask for JSON in the prompt and parse it here.
        
        import json_repair
        import re
        
        try:
            # Try to find JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
                
            metadata = json_repair.loads(json_str)
            assert isinstance(metadata, dict), "Parsed JSON is not a dictionary!"
            
            # Post-process: convert literal \n to actual newlines in metadata fields
            # This handles cases where LLM outputs double-escaped \\n instead of \n
            for key in ['abstract', 'keywords', 'author', 'title']:
                if key in metadata and isinstance(metadata[key], str):
                    # Replace literal \n (backslash followed by n) with actual newline
                    # Use raw string replacement to avoid Python's own escape interpretation
                    original = metadata[key]
                    metadata[key] = metadata[key].replace('\\n', '\n')
                    if original != metadata[key]:
                        logger.info(f"Fixed literal \\n to newline in '{key}' field")
            
            return metadata
        except Exception as e:
            logger.error(f"Failed to parse metadata JSON: {e}")
            logger.debug(f"Raw response: {response}")
            # Fallback or return raw response if parsing fails (though this might break downstream)
            return {"raw_response": response}
