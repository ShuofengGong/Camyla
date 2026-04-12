import logging
import re
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from camyla.paper_agent.func.latex_utils import deduplicate_bibtex_entries
from camyla.paper_agent.func.openrouter_client import OpenRouterClient
from camyla.paper_agent.func.ssapi.semantic_scholar_client import SemanticScholarClient
from camyla.paper_agent.func.ssapi.base import Paper

logger = logging.getLogger(__name__)


def clean_latex_output(text: str) -> str:
    """
    Clean LLM output to remove markdown code blocks and other artifacts.
    
    Args:
        text: Raw LLM output
        
    Returns:
        Cleaned LaTeX content
    """
    # Remove markdown code block markers
    text = re.sub(r'^```latex\s*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n```\s*$', '', text)
    text = re.sub(r'^```\s*\n', '', text, flags=re.MULTILINE)
    
    # Remove any remaining triple backticks
    text = text.replace('```latex', '').replace('```', '')
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def fix_percentage_escaping(latex_content: str) -> str:
    """
    Post-processing: ensure all percentage signs are properly escaped with backslash.
    
    Converts "95%" to "95\\%" 
    Avoids double-escaping "95\\%" to "95\\\\%"
    
    Args:
        latex_content: LaTeX text that may contain unescaped percent signs
        
    Returns:
        LaTeX content with all percent signs properly escaped
    """
    # Match percent signs that are NOT already escaped
    # Negative lookbehind assertion ensures no backslash before %
    pattern = r'(?<!\\)%'
    fixed_content = re.sub(pattern, r'\\%', latex_content)
    
    return fixed_content


def clean_double_escaped_percent(latex_content: str) -> str:
    """
    Post-processing: clean up double-escaped percent signs.
    
    Converts "\\\\%" to "\\%" (double backslash percent to single backslash percent).
    This fixes cases where LLM generates \\\\% in JSON which becomes \\% after parsing.
    
    Args:
        latex_content: LaTeX text that may contain double-escaped percent signs
        
    Returns:
        LaTeX content with double-escaped percents fixed to single-escaped
    """
    # Replace \\% with \% 
    # In raw string: r'\\\\%' matches literal \\%
    # Replacement r'\\%' produces literal \%
    fixed_content = re.sub(r'\\\\%', r'\\%', latex_content)
    
    return fixed_content


def convert_markdown_to_latex(latex_content: str) -> str:
    """
    Post-processing: convert markdown formatting to LaTeX.
    
    Converts markdown bold (**text**) to LaTeX bold (\textbf{text})
    Converts markdown italic (*text*) to LaTeX italic (\textit{text})
    
    This is a safety net to catch cases where markdown formatting from
    research_proposal.md was copied into LaTeX output without conversion.
    
    Args:
        latex_content: LaTeX text that may contain markdown formatting
        
    Returns:
        LaTeX content with markdown formatting converted to proper LaTeX
    """
    # Convert **bold** to \textbf{bold}
    # Match ** followed by one or more non-* characters, followed by **
    # Use non-greedy match to avoid spanning across multiple bold sections
    latex_content = re.sub(r'\*\*([^*]+?)\*\*', r'\\textbf{\1}', latex_content)
    
    # Convert *italic* to \textit{italic}
    # Match single * followed by non-* characters, followed by single *
    # Negative lookbehind/lookahead to avoid matching ** (already handled above)
    latex_content = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'\\textit{\1}', latex_content)
    
    return latex_content


class PaperWritingAgent(BaseAgent):
    """
    Agent responsible for writing academic papers in LaTeX format.
    """
    def __init__(self, template_root: Optional[Path] = None):
        super().__init__()
        if template_root is None:
            # Default to func/latex_templates relative to this file
            # agents/part3_writing.py -> agents/ -> paper_agent/ -> func/latex_templates
            template_root = Path(__file__).parent.parent / "func" / "latex_templates"
        self.template_root = template_root

    def run(
        self,
        research_idea: str,
        experimental_results: str,
        ablation_results: str = "",
        figures_description: str = "",
        dataset_context: str = "",
        training_config: str = "",
        baseline_training_policy: str = "",
        template_name: str = "default",
        reference_style: str = ""
    ) -> str:
        """
        Generate a complete LaTeX paper draft using progressive generation.
        
        Args:
            research_idea: Research proposal content
            experimental_results: Experimental results markdown
            ablation_results: Ablation study results
            figures_description: Figure descriptions
            dataset_context: Detailed dataset information to prevent LLM hallucination
            training_config: nnUNet training configuration (epochs, optimizer, loss, etc.)
            baseline_training_policy: Paper-facing baseline training/fairness wording
            template_name: LaTeX template to use
            reference_style: Citation style
        """
        self._training_config = training_config
        self._baseline_training_policy = baseline_training_policy
        logger.info(f"Generating paper draft using template: {template_name}")
        
        # 1. Load template configuration
        template_config = self._load_template_config(template_name)
        
        # 2. Generate Metadata (Title, Author, Abstract, Keywords)
        from .metadata_agent import MetadataAgent
        metadata_agent = MetadataAgent()
        metadata = metadata_agent.run(research_idea, template_config, dataset_context=dataset_context)
        
        # 3. Generate Sections with iterative refinement
        from .section_agent import SectionAgent
        from .review_agent import ReviewAgent
        
        section_agent = SectionAgent()
        review_agent = ReviewAgent()
        
        sections_content = {}
        previous_context = ""
        
        # Get sections definition from config, fallback to default if not present
        sections_def = template_config.get("sections", [
            {"name": "Introduction", "description": "Introduction section"},
            {"name": "Method", "description": "Methodology section"},
            {"name": "Experiments", "description": "Experimental results"},
            {"name": "Conclusion", "description": "Conclusion"}
        ])
        
        for section in sections_def:
            sec_name = section["name"]
            generation_mode = section.get("generation_mode", "direct")
            
            # Dispatch to appropriate generation method based on mode
            if generation_mode == "structured_experiments":
                logger.info(f"Generating {sec_name} using structured experiments mode...")
                content = self._generate_structured_experiments(
                    section, research_idea, experimental_results,
                    ablation_results, previous_context,
                    figures_description,
                    dataset_context,
                    training_config,
                    baseline_training_policy
                )
            elif generation_mode == "hierarchical":
                logger.info(f"Generating {sec_name} using hierarchical mode...")
                content = self._generate_hierarchical_section(
                    section, research_idea, experimental_results,
                    ablation_results, previous_context, review_agent,
                    figures_description,
                    dataset_context,
                )
            else:
                logger.info(f"Generating {sec_name} using direct mode...")
                content = self._generate_direct_section(
                    section, research_idea, experimental_results,
                    ablation_results, previous_context, section_agent, review_agent,
                    figures_description,
                    dataset_context,
                )
            
            sections_content[sec_name] = content
            
            # Generate real summary and pass to next section
            section_summary = self._summarize_section(sec_name, content)
            previous_context += f"\n\n=== {sec_name} (Summary) ===\n{section_summary}"
            
        # 4. Assemble Paper
        full_latex = self._assemble_paper(metadata, sections_content, template_config, template_name)
        
        # 4.5 Fix percentage escaping (post-processing)
        full_latex = fix_percentage_escaping(full_latex)
        logger.info("Applied percentage escaping fix to paper content")
        
        # 4.6 Clean double-escaped percents (post-processing)
        full_latex = clean_double_escaped_percent(full_latex)
        logger.info("Cleaned double-escaped percent signs")
        
        # 4.7 Remove markdown formatting markers (post-processing)
        # ⚠️ MODIFIED: User requested NO bold/italic formatting in paper
        # Instead of converting **text** to \textbf{text}, we remove the markers
        # to keep plain text format throughout the paper
        
        # Remove **bold** markers (convert to plain text, not \textbf{})
        full_latex = re.sub(r'\*\*([^*]+?)\*\*', r'\1', full_latex)
        
        # Remove *italic* markers (convert to plain text, not \textit{})
        full_latex = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'\1', full_latex)
        
        logger.info("Removed markdown formatting markers (preserved plain text format)")
        
        return full_latex
    
    def _validate_outline(
        self, 
        outline: Dict, 
        config: Dict
    ) -> bool:
        """
        Validate outline structure and subsection count.
        
        Args:
            outline: Outline dictionary with subsections list
            config: Subsection configuration with min/max counts
            
        Returns:
            True if outline is valid, False otherwise
        """
        subsections = outline.get("subsections", [])
        
        # Check count range
        min_count = config.get("min_subsections", 1)
        max_count = config.get("max_subsections", 10)
        
        if not (min_count <= len(subsections) <= max_count):
            logger.warning(f"Subsection count {len(subsections)} not in range [{min_count}, {max_count}]")
            return False
        
        # Check required fields
        for subsec in subsections:
            if not all(key in subsec for key in ["name", "description"]):
                logger.warning("Subsection missing required fields (name, description)")
                return False
        
        return True
    
    def _generate_direct_section(
        self,
        section_config: Dict,
        research_idea: str,
        experimental_results: str,
        ablation_results: str,
        previous_context: str,
        section_agent,
        review_agent,
        figures_description: str = "",
        dataset_context: str = "",
    ) -> str:
        """
        Generate section content directly in one pass (original logic).
        
        Args:
            section_config: Section configuration dictionary
            research_idea: Research context
            experimental_results: Experimental results
            ablation_results: Ablation study results
            previous_context: Summary of previous sections
            section_agent: SectionAgent instance
            review_agent: ReviewAgent instance
            figures_description: Figure descriptions
            dataset_context: Detailed dataset information to prevent hallucination
            
        Returns:
            Complete LaTeX content for the section
        """
        sec_name = section_config["name"]
        sec_desc = section_config["description"]
        
        # Iterative refinement: up to 2 expansion attempts
        max_retries = 2
        content = ""
        
        for attempt in range(max_retries + 1):
            logger.info(f"  Generating {sec_name} (attempt {attempt + 1}/{max_retries + 1})...")
            
            content = section_agent.run(
                section_name=sec_name,
                section_description=sec_desc,
                research_idea=research_idea,
                experimental_results=experimental_results,
                ablation_results=ablation_results,
                previous_sections_summary=previous_context,
                figures_description=figures_description,
                dataset_context=dataset_context,
            )
            
            # Quality review
            review_result = review_agent.run(sec_name, content)
            
            if review_result["approved"]:
                logger.info(f"  ✓ {sec_name} approved")
                break
            
            if attempt < max_retries:
                logger.warning(f"  ⚠ {sec_name} needs expansion: {review_result['issues']}")
                sec_desc += f"\n\nEXPANSION NEEDED:\n{review_result['suggestions']}"
            else:
                logger.warning(f"  ⚠ {sec_name} still needs improvement after {max_retries} attempts")
        
        return content
    
    def _generate_hierarchical_section(
        self,
        section_config: Dict,
        research_idea: str,
        experimental_results: str,
        ablation_results: str,
        previous_context: str,
        review_agent,
        figures_description: str = "",
        dataset_context: str = "",
    ) -> str:
        """
        Generate section content hierarchically: outline first, then subsections.
        
        Args:
            section_config: Section configuration dictionary
            research_idea: Research context
            experimental_results: Experimental results
            ablation_results: Ablation study results
            previous_context: Summary of previous sections
            review_agent: ReviewAgent instance
            figures_description: Figure descriptions
            dataset_context: Detailed dataset information to prevent hallucination
            
        Returns:
            Complete LaTeX content for the section with subsections
        """
        from .outline_agent import OutlineAgent
        from .subsection_agent import SubsectionAgent
        
        sec_name = section_config["name"]
        sec_desc = section_config["description"]
        subsection_config = section_config.get("subsection_config", {})
        
        # Step 1: Generate subsection outline
        outline_agent = OutlineAgent()
        max_outline_retries = 2
        outline = None
        outline_valid = False
        
        for attempt in range(max_outline_retries + 1):
            logger.info(f"  Generating outline for {sec_name} (attempt {attempt+1})...")
            
            candidate_outline = outline_agent.run(
                section_name=sec_name,
                section_description=sec_desc,
                research_idea=research_idea,
                previous_sections_summary=previous_context,
                subsection_config=subsection_config
            )

            if (
                sec_name == "Related Work"
                and isinstance(candidate_outline, dict)
                and "subsections" in candidate_outline
            ):
                max_count = subsection_config.get("max_subsections")
                if (
                    isinstance(max_count, int)
                    and max_count > 0
                    and len(candidate_outline["subsections"]) > max_count
                ):
                    logger.warning(
                        f"  Related Work outline produced {len(candidate_outline['subsections'])} "
                        f"subsections; truncating to {max_count}"
                    )
                    candidate_outline["subsections"] = candidate_outline["subsections"][:max_count]

            outline = candidate_outline
            
            if self._validate_outline(outline, subsection_config):
                outline_valid = True
                logger.info(f"  ✓ Outline approved with {len(outline['subsections'])} subsections")
                break
            else:
                logger.warning(f"  ⚠ Outline validation failed (attempt {attempt+1})")
        
        if not outline_valid or not outline or "subsections" not in outline:
            logger.error(f"Failed to generate valid outline for {sec_name}, falling back to direct mode")
            # Fallback: use direct generation
            from .section_agent import SectionAgent
            section_agent = SectionAgent()
            return self._generate_direct_section(
                section_config, research_idea, experimental_results,
                ablation_results, previous_context, section_agent, review_agent,
                figures_description, dataset_context
            )
        
        # Step 2: Generate each subsection
        subsection_agent = SubsectionAgent()
        subsection_contents = []
        sibling_summaries = ""
        
        for i, subsec_def in enumerate(outline["subsections"]):
            subsec_name = subsec_def["name"]
            max_retries = 2
            content = ""  # Initialize to avoid unbound variable
            
            for attempt in range(max_retries + 1):
                logger.info(f"  Generating subsection {i+1}/{len(outline['subsections'])}: {subsec_name} (attempt {attempt+1})")
                
                content = subsection_agent.run(
                    section_name=sec_name,
                    subsection_name=subsec_name,
                    subsection_description=subsec_def["description"],
                    subsection_focus=subsec_def.get("focus", ""),
                    research_idea=research_idea,
                    experimental_results=experimental_results,
                    ablation_results=ablation_results,
                    sibling_subsections_summary=sibling_summaries,
                    previous_sections_summary=previous_context,
                    figures_description=figures_description,
                    dataset_context=dataset_context,
                )
                
                # Quality review
                review_result = review_agent.run(
                    f"{sec_name} > {subsec_name}", 
                    content
                )
                
                if review_result["approved"]:
                    logger.info(f"  ✓ Subsection {subsec_name} approved")
                    break
                elif attempt < max_retries:
                    logger.warning(f"  ⚠ Subsection needs expansion: {review_result['issues']}")
                    subsec_def["description"] += f"\n{review_result['suggestions']}"
            
            # Save subsection
            subsection_contents.append({
                "name": subsec_name,
                "content": content
            })
            
            # Generate summary for sibling context
            subsec_summary = self._summarize_section(subsec_name, content)
            sibling_summaries += f"\n- {subsec_name}: {subsec_summary}"
        
        # Step 3: Assemble section from subsections
        full_section_content = ""
        for subsec in subsection_contents:
            full_section_content += f"\\subsection{{{subsec['name']}}}\n\n"
            full_section_content += subsec["content"]
            full_section_content += "\n\n"
        
        return full_section_content.strip()
    
    def _generate_structured_experiments(
        self,
        section_config: Dict,
        research_idea: str,
        experimental_results: str,
        ablation_results: str,
        previous_context: str,
        figures_description: str = "",
        dataset_context: str = "",
        training_config: str = "",
        baseline_training_policy: str = ""
    ) -> str:
        """
        Generate Experiments section with fixed 3-subsection structure.
        
        Args:
            section_config: Section configuration with fixed_structure
            research_idea: Research context
            experimental_results: Experimental results
            ablation_results: Ablation study results
            previous_context: Summary of previous sections
            figures_description: Figure descriptions
            dataset_context: Detailed dataset information to prevent hallucination
            training_config: nnUNet training configuration (epochs, optimizer, loss, etc.)
            baseline_training_policy: Paper-facing baseline training/fairness wording
            
        Returns:
            Complete LaTeX content for Experiments section
        """
        from .section_agent import SectionAgent
        
        sec_name = section_config["name"]
        fixed_structure = section_config.get("fixed_structure", {})
        subsections_def = fixed_structure.get("subsections", [])
        
        logger.info(f"Generating {sec_name} with {len(subsections_def)} fixed subsections...")
        
        section_agent = SectionAgent()
        full_content = ""
        
        for i, subsec_config in enumerate(subsections_def):
            subsec_name = subsec_config["name"]
            skill_path = subsec_config.get("skill_path", "medical_segmentation/section.md")
            
            logger.info(f"  [{i+1}/{len(subsections_def)}] Generating {subsec_name}...")
            
            # Load specialized skill prompt
            prompt = self.load_skill(skill_path,
                section_name=sec_name,
                subsection_name=subsec_name,
                research_idea=research_idea,
                experimental_results=experimental_results,
                ablation_results=ablation_results,
                previous_context=previous_context,
                figures_description=figures_description,
                dataset_context=dataset_context,
                training_config=training_config,
                baseline_training_policy=baseline_training_policy,
            )
            
            # Generate content
            content = self.chat(messages=[{"role": "user", "content": prompt}])
            content = clean_latex_output(content)
            
            # Add subsection header
            # Note: Use single \n for actual newline, not \\n which creates literal '\n' string
            full_content += f"\\subsection{{{subsec_name}}}\n\n"
            full_content += content.strip()
            full_content += "\n\n"
        
        return full_content.strip()
    
    def _summarize_section(self, section_name: str, content: str) -> str:
        """
        Summarize a section's content using LLM for context propagation.
        
        Args:
            section_name: Name of the section
            content: Full LaTeX content of the section
            
        Returns:
            200-300 word summary in plain text
        """
        logger.info(f"Summarizing section: {section_name}...")
        
        prompt = self.load_skill(
            "common/summarize.md",
            section_name=section_name,
            content=content
        )
        
        summary = self.chat(messages=[{"role": "user", "content": prompt}])
        return summary.strip()

    def _load_template_config(self, template_name: str) -> Dict[str, Any]:
        """Load template.json configuration"""
        import json_repair
        
        template_path = self.template_root / template_name
        config_file = template_path / "template.json"
        
        if config_file.exists():
            try:
                config = json_repair.loads(config_file.read_text(encoding="utf-8"))
                assert isinstance(config, dict), "Parsed JSON is not a dictionary!"
                return config
            except Exception as e:
                logger.error(f"Failed to load template config: {e}")
        
        # Default config if not found
        return {
            "documentclass": "article",
            "sections": [],
            "metadata_fields": {}
        }

    def _assemble_paper(self, metadata: Dict[str, str], sections: Dict[str, str], config: Dict[str, Any], template_name: str) -> str:
        """Assemble the final LaTeX document using template-based approach"""
        import re
        
        logger.info(f"Assembling paper using template: {template_name}")
        
        # Read the template file
        template_path = self.template_root / template_name / config.get("main_file", "template.tex")
        
        if not template_path.exists():
            logger.error(f"Template file not found: {template_path}")
            # Fallback to simple assembly
            return self._simple_assemble(metadata, sections, config)
        
        template_content = template_path.read_text(encoding="utf-8")
        logger.info("Template file loaded successfully")
        
        # Replace title placeholder
        if "title" in metadata:
            title_text = metadata["title"]
            # Replace content between %%%%%%%%%TITLE%%%%%%%%% markers
            # Use lambda to avoid LaTeX backslashes being interpreted as regex escapes
            template_content = re.sub(
                r'%%%%%%%%%TITLE%%%%%%%%%.*?%%%%%%%%%TITLE%%%%%%%%%',
                lambda m: title_text,  # Directly replace with content, remove markers
                template_content,
                flags=re.DOTALL
            )
            logger.info("Title replaced in template")
        
        # Replace abstract placeholder
        if "abstract" in metadata:
            abstract_text = metadata["abstract"]
            # Use lambda to avoid LaTeX backslashes being interpreted as regex escapes
            template_content = re.sub(
                r'%%%%%%%%%ABSTRACT%%%%%%%%%.*?%%%%%%%%%ABSTRACT%%%%%%%%%',
                lambda m: abstract_text,  # Directly replace with content, remove markers
                template_content,
                flags=re.DOTALL
            )
            logger.info("Abstract replaced in template")
        
        # Replace keywords if present
        if "keywords" in metadata:
            keywords_text = metadata["keywords"]
            # For Elsevier template, keywords are inside \begin{keywords}...\end{keywords}
            # Use lambda to avoid LaTeX backslashes being interpreted as regex escapes
            template_content = re.sub(
                r'(\\begin\{keywords\}).*?(\\end\{keywords\})',
                lambda m: f'\\begin{{keywords}}\n{keywords_text}\n\\end{{keywords}}',
                template_content,
                flags=re.DOTALL
            )
            logger.info("Keywords replaced in template")
        
        # Insert section content after each \section{...} command
        for sec_name, sec_content in sections.items():
            # Match \section{sec_name} and insert content after it
            pattern = rf'(\\section\{{{re.escape(sec_name)}\}})'
            # Use lambda to avoid LaTeX backslashes in section content being interpreted as regex escapes
            template_content = re.sub(
                pattern,
                lambda m: f'{m.group(1)}\n\n{sec_content}',
                template_content
            )
            logger.info(f"Inserted content for section: {sec_name}")
        
        logger.info("Paper assembly completed")
        return template_content
    
    def _simple_assemble(self, metadata: Dict[str, str], sections: Dict[str, str], config: Dict[str, Any]) -> str:
        """Fallback: simple assembly when template file is not found"""
        logger.warning("Using fallback simple assembly")
        
        # Start with document class
        cls = config.get("documentclass", "article")
        latex = [f"\\documentclass{{{cls}}}"]
        
        # Add packages
        latex.append("\\usepackage{graphicx}")
        latex.append("\\usepackage{amsmath}")
        latex.append("\\usepackage{booktabs}")
        latex.append("\\usepackage{hyperref}")
        latex.append("\\begin{document}")
        
        if "title" in metadata:
            latex.append(metadata["title"])
        if "author" in metadata:
            latex.append(metadata["author"])
        
        latex.append("\\maketitle")
        
        if "abstract" in metadata:
            latex.append(metadata["abstract"])
            
        if "keywords" in metadata:
            latex.append(metadata["keywords"])
            
        # Sections
        sections_def = config.get("sections", [])
        for sec in sections_def:
            name = sec["name"]
            if name in sections:
                latex.append(f"\\section{{{name}}}")
                latex.append(sections[name])
                
        latex.append("\\end{document}")
        
        return "\n\n".join(latex)



class BibtexAgent(BaseAgent):
    """
    Agent responsible for managing citations using Semantic Scholar.
    """
    def __init__(self, debug_mode: bool = False, max_tokens: int = 5000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_mode = debug_mode  # Debug mode flag
        self.max_tokens = max_tokens  # LLM max generation tokens
        # Use one OpenRouter client per worker thread so citation verification
        # and retry-query generation can run concurrently without sharing a
        # single client instance.
        self._thread_local = threading.local()
        
        # Only initialize the S2 client outside debug mode
        if not self.debug_mode:
            # Allow searching for papers since 2000 to include classic deep-learning-era papers.
            # The default 2023-01-01 would exclude classics like U-Net (2015) and Swin Transformer (2021).
            # Disable the open-access requirement so that classic papers (often not open access) can be searched.
            self.s2_client = SemanticScholarClient(
                enable_venue_filter=False,
                min_year="2000-01-01",
                require_open_access=False  # Allow searching non-open-access classic papers
            )
        else:
            self.s2_client = None
            logger.info("🔧 BibtexAgent initialized in DEBUG MODE - API calls will be skipped")
        
        logger.info(f"BibtexAgent initialized with max_tokens={self.max_tokens}")
    
    def _safe_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Thread-local wrapper for LLM chat calls.
        
        Args:
            messages: Chat messages
            
        Returns:
            LLM response content
        """
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenRouterClient()
            self._thread_local.client = client

        response = client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if response.content is None:
            logger.warning("Received empty content from citation LLM.")
            return ""

        return response.content
    
    def _extract_citation_context(
        self, 
        paper_content: str, 
        keyword: str, 
        context_chars: int = 200
    ) -> str:
        """
        Extract context around a citation placeholder.
        
        Args:
            paper_content: Full paper LaTeX content
            keyword: Citation keyword (e.g., "unet")
            context_chars: Number of characters to extract before/after placeholder
            
        Returns:
            Context snippet with the target placeholder marked by **[TARGET]...[/TARGET]**
        """
        # Build patterns to find the placeholder
        patterns = [
            rf'\[CITE:{re.escape(keyword)}\]',
            rf'\\cite\{{CITE:{re.escape(keyword)}\}}',
            rf'\[CTA:{re.escape(keyword)}\]',
            rf'\\cite\{{missing_{re.escape(keyword)}\}}',
        ]
        
        # Find the first occurrence of this placeholder
        placeholder_pos = -1
        matched_placeholder = ""
        
        for pattern in patterns:
            match = re.search(pattern, paper_content)
            if match:
                placeholder_pos = match.start()
                matched_placeholder = match.group(0)
                break
        
        if placeholder_pos == -1:
            # Placeholder not found, return keyword as fallback
            logger.warning(f"Placeholder for keyword '{keyword}' not found in paper")
            return f"**[TARGET]**[CITE:{keyword}]**[/TARGET]**"
        
        # Extract context before and after
        start_pos = max(0, placeholder_pos - context_chars)
        end_pos = min(len(paper_content), placeholder_pos + len(matched_placeholder) + context_chars)
        
        # Get the context parts
        context_before = paper_content[start_pos:placeholder_pos]
        context_after = paper_content[placeholder_pos + len(matched_placeholder):end_pos]
        
        # Build context with target marker
        # Wrap the target citation with **[TARGET]** and **[/TARGET]** markers
        context = (
            ("..." if start_pos > 0 else "") +
            context_before +
            f"**[TARGET]**{matched_placeholder}**[/TARGET]**" +
            context_after +
            ("..." if end_pos < len(paper_content) else "")
        )
        
        return context
    
    def _verify_citation_with_llm(
        self,
        keyword: str,
        context: str,
        papers: List[Paper],
        max_candidates: int = 15
    ) -> Optional[Paper]:
        """
        Use LLM to select the most appropriate paper from search results.
        
        Args:
            keyword: Citation keyword
            context: Citation context snippet
            papers: List of candidate papers from search
            max_candidates: Maximum number of candidates to show to LLM
            
        Returns:
            Selected Paper object or None if all validation fails
        """
        if not papers:
            return None
        
        # If only one paper, return it directly
        if len(papers) == 1:
            return papers[0]
        
        # Limit candidates
        candidates = papers[:max_candidates]
        
        # Format candidates for LLM
        candidates_formatted = ""
        for i, paper in enumerate(candidates, 1):
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += " et al."
            
            candidates_formatted += f"""{i}. Title: {paper.title}
   Author: {authors_str}
   Year: {paper.year}
   Citation Count: {paper.metadata.get('citation_count', 0)}
   Venue: {paper.venue}

"""
        
        # Load verification prompt
        try:
            prompt = self.load_skill(
                "common/citation_verification.md",
                keyword=keyword,
                context=context,
                candidates=candidates_formatted.strip()
            )
            
            # Get LLM response
            response = self._safe_chat(messages=[{"role": "user", "content": prompt}])
            logger.info(f"  LLM verification response: {response[:200]}...")
            
            # Parse response
            import re
            
            # First check for NONE selection
            none_match = re.search(r'Selection[：:]\s*NONE', response, re.IGNORECASE)
            if none_match:
                # Extract reasoning
                reason_match = re.search(r'Reason[：:]\s*(.+)', response, re.IGNORECASE)
                reason = reason_match.group(1).strip() if reason_match else "No suitable candidate"
                
                logger.info(f"  ⚠️  LLM judged that none of the candidates are suitable")
                logger.info(f"    Reason: {reason}")
                return None  # Return None to trigger retry with new query
            
            # Then check for number selection
            match = re.search(r'Selection[：:]\s*(\d+)', response, re.IGNORECASE)
            if match:
                selected_idx = int(match.group(1)) - 1  # Convert to 0-indexed
                if 0 <= selected_idx < len(candidates):
                    selected_paper = candidates[selected_idx]
                    
                    # Extract reasoning if present (support both languages)
                    reason_match = re.search(r'Reason[：:]\s*(.+)', response, re.IGNORECASE)
                    reason = reason_match.group(1).strip() if reason_match else "No reason provided"
                    
                    logger.info(f"  ✓ LLM selected paper #{selected_idx + 1}: {selected_paper.title[:60]}...")
                    logger.info(f"    Reason: {reason}")
                    return selected_paper
                else:
                    logger.warning(f"  ✗ LLM returned invalid index: {selected_idx + 1}")
            else:
                logger.warning(f"  ✗ Could not parse LLM response")
                
        except Exception as e:
            logger.warning(f"  ✗ LLM verification failed: {e}")
        
        # If parsing failed or exception occurred, return None to trigger retry
        logger.info(f"  ⚠️  Verification failed; returning None to trigger a new search")
        return None
    
    def _generate_retry_query(
        self,
        keyword: str,
        context: str,
        failure_reason: str,
        previous_queries: List[str]
    ) -> Optional[str]:
        """
        Use the LLM to generate a new search keyword from the failure information.

        Args:
            keyword: Original keyword.
            context: Citation context.
            failure_reason: Failure reason ("no search results" or "none match").
            previous_queries: List of previously tried queries.

        Returns:
            A new search keyword, or None if generation fails.
        """
        logger.info(f"  → Calling LLM to generate retry keyword...")

        # Format attempted queries
        queries_formatted = "\n".join([f"  - {q}" for q in previous_queries])

        try:
            # Load the prompt
            prompt = self.load_skill(
                "common/citation_retry_query.md",
                original_keyword=keyword,
                context=context,
                failure_reason=failure_reason,
                previous_queries=queries_formatted
            )
            
            # Invoke the LLM
            response = self._safe_chat(messages=[{"role": "user", "content": prompt}])
            logger.info(f"  LLM retry-query response: {response[:200]}...")

            # Parse the response
            import re
            query_match = re.search(r'New Query[：:]\s*(.+)', response, re.IGNORECASE)
            reasoning_match = re.search(r'Reasoning[：:]\s*(.+)', response, re.IGNORECASE)

            if query_match:
                new_query = query_match.group(1).strip()
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

                logger.info(f"  ✓ Generated new query: '{new_query}'")
                logger.info(f"    Reasoning: {reasoning}")
                return new_query
            else:
                logger.warning(f"  ✗ Could not parse LLM response")
                return None

        except Exception as e:
            logger.warning(f"  ✗ LLM retry-query generation failed: {e}")
            return None
    
    def _process_single_citation(
        self,
        keyword: str,
        initial_query: str,
        context: str,
        max_attempts: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Full pipeline for processing a single citation (search + verify + retry).
        Thread-safe method, suitable for concurrent processing.

        Args:
            keyword: Citation keyword.
            initial_query: Initial search query.
            context: Citation context.
            max_attempts: Maximum retry attempts.

        Returns:
            On success: {'keyword': str, 'citation_key': str, 'bibtex': str}.
            On failure: None.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[Thread] Processing citation: '{keyword}'")
        
        current_query = initial_query
        previous_queries = []  # Track all attempted queries
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"  [Thread] Attempt {attempt}/{max_attempts}: Searching for '{current_query}'")
            previous_queries.append(current_query)
            
            try:
                # Search papers
                assert self.s2_client is not None, "s2_client should be initialized in normal mode"
                papers = self.s2_client.search(current_query, limit=20)
                
                if papers:
                    # LLM verification (thread-safe via _safe_chat)
                    paper = self._verify_citation_with_llm(
                        keyword=keyword,
                        context=context,
                        papers=papers,
                        max_candidates=15
                    )
                    
                    if paper:
                        # ✓ Verification success
                        logger.info(f"  [Thread] ✓ Selected: {paper.title[:80]}... -> {paper.citation_key}")
                        logger.info(f"    ({paper.year}, {paper.metadata.get('citation_count', 0)} citations)")
                        return {
                            'keyword': keyword,
                            'citation_key': paper.citation_key,
                            'bibtex': paper.bibtex
                        }
                    else:
                        # ✗ Verification failed (LLM returned NONE)
                        failure_reason = f"Attempt {attempt}: found {len(papers)} candidates, but LLM judged none to match"
                        logger.warning(f"  [Thread] ✗ {failure_reason}")
                        
                        # If not at max attempts, let LLM generate new query
                        if attempt < max_attempts:
                            new_query = self._generate_retry_query(
                                keyword=keyword,
                                context=context,
                                failure_reason="None match: no paper in the search results matches the citation intent",
                                previous_queries=previous_queries
                            )
                            
                            if new_query and new_query not in previous_queries:
                                current_query = new_query
                                logger.info(f"  [Thread] → Will retry with the new query")
                            else:
                                logger.warning(f"  [Thread] ✗ LLM failed to generate a valid new query; aborting retries")
                                break
                        else:
                            # Reached max attempts
                            logger.warning(f"  [Thread] ⚠️  Reached {max_attempts} verification failures")
                            break
                else:
                    # ✗ No search results
                    failure_reason = f"Attempt {attempt}: no search results"
                    logger.warning(f"  [Thread] ✗ {failure_reason}")
                    
                    # If not at max attempts, let LLM generate new query
                    if attempt < max_attempts:
                        new_query = self._generate_retry_query(
                            keyword=keyword,
                            context=context,
                            failure_reason="No search results: this query returns no papers on Semantic Scholar",
                            previous_queries=previous_queries
                        )
                        
                        if new_query and new_query not in previous_queries:
                            current_query = new_query
                            logger.info(f"  [Thread] → Will retry with the new query")
                        else:
                            logger.warning(f"  [Thread] ✗ LLM failed to generate a valid new query; aborting retries")
                            break
                    else:
                        # Reached max attempts
                        logger.warning(f"  [Thread] ⚠️  Reached {max_attempts} search failures")
                        break
                        
            except Exception as e:
                logger.warning(f"  [Thread] ✗ Search error: {e}")
                # Also try to retry on exception
                if attempt < max_attempts:
                    new_query = self._generate_retry_query(
                        keyword=keyword,
                        context=context,
                        failure_reason=f"Search exception: {str(e)}",
                        previous_queries=previous_queries
                    )
                    if new_query and new_query not in previous_queries:
                        current_query = new_query
                    else:
                        break
                else:
                    break
        
        # Complete failure
        logger.warning(f"  [Thread] ✗ Final failure: could not find a citation for '{keyword}' (tried {len(previous_queries)} times)")
        logger.info(f"    Attempted queries: {previous_queries}")
        return None
    
    def run(self, paper_content: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Extract citation placeholders and search for papers on Semantic Scholar.
        
        Args:
            paper_content: The LaTeX paper content with citation placeholders
            max_retries: Maximum retry attempts for each citation
            
        Returns:
            Dictionary with:
                - updated_content: Paper with citations replaced or removed
                - bibtex: Combined BibTeX entries
                - citations: Dict of found citations
                - removed_citations: List of citations that couldn't be found
        """
        
        # ========== DEBUG MODE: strip all citations directly ==========
        if self.debug_mode:
            logger.info("="*60)
            logger.info("🔧 DEBUG MODE ENABLED - Skipping Semantic Scholar API")
            logger.info("="*60)
            
            # Step 1: find all citation placeholders
            pattern1 = r'\[CITE:([^\]]+)\]'
            pattern2 = r'\\cite\{CITE:([^\}]+)\}'
            pattern3 = r'\[CTA:([^\]]+)\]'
            pattern4 = r'\\cite\{missing_([^\}]+)\}'
            
            matches1 = re.findall(pattern1, paper_content)
            matches2 = re.findall(pattern2, paper_content)
            matches3 = re.findall(pattern3, paper_content)
            matches4 = re.findall(pattern4, paper_content)
            
            all_keywords = list(set(matches1 + matches2 + matches3 + matches4))
            logger.info(f"Found {len(all_keywords)} citation placeholders to remove:")
            for kw in all_keywords:
                logger.info(f"  - {kw}")
            
            # Step 2: remove all placeholders directly
            updated_content = paper_content
            
            for keyword in all_keywords:
                patterns_to_remove = [
                    rf'\[CITE:{re.escape(keyword)}\]',
                    rf'\\cite\{{CITE:{re.escape(keyword)}\}}',
                    rf'\[CTA:{re.escape(keyword)}\]',
                    rf'\\cite\{{missing_{re.escape(keyword)}\}}',
                ]
                
                for pattern in patterns_to_remove:
                    updated_content = re.sub(pattern, '', updated_content)
            
            # Step 3: clean up extra whitespace
            updated_content = re.sub(r' +', ' ', updated_content)
            updated_content = re.sub(r' +([.,;:])', r'\1', updated_content)
            
            logger.info(f"✓ Removed {len(all_keywords)} citation placeholders")
            logger.info("✓ Generated empty references.bib (debug mode)")
            
            return {
                "updated_content": updated_content,
                "bibtex": "",  # Empty BibTeX
                "citations": {},
                "removed_citations": all_keywords
            }
        
        # ========== Normal mode: original logic ==========
        logger.info("Extracting citation placeholders...")
        
        # Step 0: Pre-processing - Clean up any malformed citations from LLM output
        logger.info("Pre-processing: cleaning malformed citations...")
        
        # Remove double \cite\cite
        paper_content = re.sub(r'\\cite\\cite\{', r'\\cite{', paper_content)
        
        # Fix bracket mismatches: \cite[xxx} -> \cite{xxx}
        paper_content = re.sub(r'\\cite\[([^\]]+)\}', r'\\cite{\1}', paper_content)
        paper_content = re.sub(r'\\cite\{([^\}]+)\]', r'\\cite{\1}', paper_content)
        
        # Remove orphan \cite} (cite without content)
        paper_content = re.sub(r'\\cite\}', '', paper_content)
        
        logger.info("Pre-processing complete")
        
        # Step 0.5: Detect academic-style citations (should use placeholder format instead)
        academic_pattern = r'\\cite\{([A-Z][a-z]+\d{4}[a-zA-Z]*)\}'
        academic_citations = re.findall(academic_pattern, paper_content)
        
        if academic_citations:
            logger.warning(f"⚠️ Found {len(academic_citations)} academic-style citations: {academic_citations[:5]}...")
            logger.warning("These citations use standard format like \\cite{Author2023}")
            logger.warning("They should have been generated as [CITE:keyword] placeholders!")
            logger.warning("Check the SectionAgent prompt to ensure it enforces placeholder format.")
            logger.warning("These citations will NOT be processed and will appear as undefined in the PDF.")
        
        # Step 1: Find all citation placeholders in the text
        # Support multiple formats: [CITE:xxx], \cite{CITE:xxx}, [CTA:xxx], etc.
        pattern1 = r'\[CITE:([^\]]+)\]'
        pattern2 = r'\\cite\{CITE:([^\}]+)\}'
        pattern3 = r'\[CTA:([^\]]+)\]'  # Alternative format
        pattern4 = r'\\cite\{missing_([^\}]+)\}'  # Catch any existing missing_ ones
        
        matches1 = re.findall(pattern1, paper_content)
        matches2 = re.findall(pattern2, paper_content)
        matches3 = re.findall(pattern3, paper_content)
        matches4 = re.findall(pattern4, paper_content)
        
        all_keywords = list(set(matches1 + matches2 + matches3 + matches4))
        logger.info(f"Found {len(all_keywords)} unique citation keywords: {all_keywords}")
        
        if not all_keywords:
            logger.warning("No citation placeholders found in paper")
            return {
                "updated_content": paper_content,
                "bibtex": "",
                "citations": {},
                "removed_citations": []
            }
        
        # Step 2: Extract context for each citation and use LLM to generate precise search queries
        logger.info("Extracting context for each citation...")
        citation_contexts = {}
        for keyword in all_keywords:
            context = self._extract_citation_context(paper_content, keyword)
            citation_contexts[keyword] = context
            logger.info(f"  Context for '{keyword}': {context[:100]}...")
        
        # Format contexts for LLM prompt
        contexts_formatted = "\n\n".join([
            f"{i+1}. Keyword: \"{kw}\"\n   Context: {ctx}"
            for i, (kw, ctx) in enumerate(citation_contexts.items())
        ])
        
        prompt = self.load_skill(
            "common/citation_extraction_v2.md",
            citations_with_context=contexts_formatted
        )
        
        extraction_response = self._safe_chat(messages=[{"role": "user", "content": prompt}])
        logger.info(f"Citation extraction response:\n{extraction_response[:500]}...")
        
        # Step 3: Parse the LLM response to get search queries
        search_queries = self._parse_search_queries(extraction_response, all_keywords, citation_contexts)
        
        # Step 4: Process citations concurrently
        import concurrent.futures
        import time as time_module
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting parallel citation processing...")
        logger.info(f"Total citations to process: {len(search_queries)}")
        logger.info(f"Max concurrent threads: 3 (respecting Semantic Scholar 1req/s limit)")
        logger.info(f"{'='*60}\n")
        
        citations = {}
        bibtex_entries = []
        removed_citations = []
        
        start_time = time_module.time()
        
        # Use ThreadPoolExecutor for parallel processing
        # max_workers=5 ensures ~5 requests/second (each request has 1s delay)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all citation processing tasks
            future_to_keyword = {
                executor.submit(
                    self._process_single_citation,
                    keyword=keyword,
                    initial_query=query,
                    context=citation_contexts.get(keyword, ""),
                    max_attempts=3
                ): keyword
                for keyword, query in search_queries.items()
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_keyword):
                keyword = future_to_keyword[future]
                try:
                    result = future.result()
                    if result:
                        # Success
                        citations[result['keyword']] = result['citation_key']
                        bibtex_entries.append(result['bibtex'])
                        logger.info(f"✓ Completed: {result['keyword']} -> {result['citation_key']}")
                    else:
                        # Failed after all retries
                        removed_citations.append(keyword)
                        logger.warning(f"✗ Failed: {keyword}")
                except Exception as e:
                    logger.error(f"Exception processing '{keyword}': {e}")
                    removed_citations.append(keyword)
        
        elapsed_time = time_module.time() - start_time
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Parallel citation processing complete!")
        logger.info(f"  Total time: {elapsed_time:.1f}s")
        logger.info(f"  Average: {elapsed_time/len(search_queries):.1f}s per citation")
        logger.info(f"  Found: {len(citations)} citations")
        logger.info(f"  Failed: {len(removed_citations)} citations")
        logger.info(f"{'='*60}\n")
        
        # Step 5: Replace found citations and remove unfound ones
        updated_content = paper_content
        
        # Replace found citations
        for keyword, cite_key in citations.items():
            patterns = [
                (rf'\[CITE:{re.escape(keyword)}\]', f"\\\\cite{{{cite_key}}}"),
                (rf'\\cite\{{CITE:{re.escape(keyword)}\}}', f"\\\\cite{{{cite_key}}}"),
                (rf'\[CTA:{re.escape(keyword)}\]', f"\\\\cite{{{cite_key}}}"),
                (rf'\\cite\{{missing_{re.escape(keyword)}\}}', f"\\\\cite{{{cite_key}}}"),
            ]
            
            for pattern, replacement in patterns:
                updated_content = re.sub(pattern, replacement, updated_content)
        
        # Remove unfound citation placeholders silently (no marker left in text)
        for keyword in removed_citations:
            patterns_to_replace = [
                rf'\[CITE:{re.escape(keyword)}\]',
                rf'\\cite\{{CITE:{re.escape(keyword)}\}}',
                rf'\[CTA:{re.escape(keyword)}\]',
                rf'\\cite\{{missing_{re.escape(keyword)}\}}',
            ]
            
            for pattern in patterns_to_replace:
                updated_content = re.sub(pattern, '', updated_content)
            logger.info(f"  Removed unfound citation placeholder '{keyword}'")
        
        # --- Previous approach: insert red [Citation needed] marker (kept for future use) ---
        # citation_needed_marker = r'\\textcolor{red}{[Citation needed]}'
        # for keyword in removed_citations:
        #     patterns_to_replace = [
        #         rf'\[CITE:{re.escape(keyword)}\]',
        #         rf'\\cite\{{CITE:{re.escape(keyword)}\}}',
        #         rf'\[CTA:{re.escape(keyword)}\]',
        #         rf'\\cite\{{missing_{re.escape(keyword)}\}}',
        #     ]
        #     for pattern in patterns_to_replace:
        #         updated_content = re.sub(pattern, citation_needed_marker, updated_content)
        #     logger.info(f"  Replaced unfound citation '{keyword}' with red [Citation needed] marker")
        # --- End of previous approach ---
        
        # Clean up any double spaces or spaces before punctuation caused by removing citations
        updated_content = re.sub(r' +', ' ', updated_content)  # Multiple spaces -> single
        updated_content = re.sub(r' +([.,;:])', r'\1', updated_content)  # Space before punctuation
        
        # Post-processing: Final cleanup of any remaining malformed citations
        logger.info("Post-processing: final cleanup...")
        
        # Remove any remaining double \cite\cite
        updated_content = re.sub(r'\\cite\\cite\{', r'\\cite{', updated_content)
        
        # Fix any remaining bracket mismatches
        updated_content = re.sub(r'\\cite\[([^\]]+)\}', r'\\cite{\1}', updated_content)
        updated_content = re.sub(r'\\cite\{([^\}]+)\]', r'\\cite{\1}', updated_content)
        
        # Remove orphan \cite}
        updated_content = re.sub(r'\\cite\}', '', updated_content)
        
        # Remove any \cite with empty braces
        updated_content = re.sub(r'\\cite\{\}', '', updated_content)
        
        # Apply percentage escaping fix to citation-processed content
        updated_content = fix_percentage_escaping(updated_content)
        logger.info("Applied percentage escaping fix to citation-processed content")
        
        logger.info("Post-processing complete")
        logger.info(f"\nReplaced {len(citations)} citations successfully")
        logger.info(f"Removed {len(removed_citations)} unfound citations")
        
        # Validate BibTeX content
        bibtex_content = deduplicate_bibtex_entries("\n\n".join(bibtex_entries))
        unique_bibtex_count = len(
            re.findall(r'@\w+\{\s*([^,\s]+)\s*,', bibtex_content)
        )
        if not bibtex_content or not bibtex_content.strip():
            logger.warning("⚠️  No BibTeX entries generated! The references.bib file will be empty.")
            logger.warning("This means no citations were successfully found.")
        else:
            logger.info(
                f"✓ Generated BibTeX with {unique_bibtex_count} unique entries"
            )
        
        return {
            "updated_content": updated_content,
            "bibtex": bibtex_content,
            "citations": citations,
            "removed_citations": removed_citations
        }
    
    def _parse_search_queries(
        self, 
        llm_response: str, 
        fallback_keywords: List[str],
        citation_contexts: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Parse LLM response to extract keyword -> search query mapping.
        
        Args:
            llm_response: LLM response with citation analysis
            fallback_keywords: List of keywords to ensure coverage
            citation_contexts: Dictionary of {keyword: context} for fallback
            
        Returns:
            Dictionary mapping keyword to search query
        """
        queries = {}
        lines = llm_response.split('\n')
        
        current_keyword = None
        current_analysis = None
        
        for line in lines:
            line = line.strip()
            
            # Look for "Keyword: xxx" or numbered format "1. Keyword: xxx"
            if "Keyword:" in line or "keyword:" in line.lower():
                match = re.search(r'[Kk]eyword:\s*["\']?([^"\']+)["\']?', line)
                if match:
                    current_keyword = match.group(1).strip()
                    logger.debug(f"Found keyword: {current_keyword}")
            
            # Look for "Analysis: xxx" (optional, for logging)
            elif ("Analysis:" in line or "analysis:" in line.lower()) and current_keyword:
                match = re.search(r'[Aa]nalysis:\s*(.+)', line)
                if match:
                    current_analysis = match.group(1).strip()
                    logger.info(f"  Analysis for '{current_keyword}': {current_analysis}")
            
            # Look for "Search Query: xxx"
            elif ("Search Query:" in line or "query:" in line.lower()) and current_keyword:
                match = re.search(r'[Qq]uery:\s*["\']?([^"\']+)["\']?', line)
                if match:
                    queries[current_keyword] = match.group(1).strip()
                    logger.info(f"  Search query for '{current_keyword}': {queries[current_keyword]}")
                    current_keyword = None
                    current_analysis = None
        
        # Fallback: if LLM didn't provide queries, infer from context (not keyword)
        _MARKER_STOPWORDS = {
            'CITE', 'CTA', 'TARGET', 'TARGET]', '[TARGET]',
            'The', 'This', 'That', 'These', 'Those',
            'Our', 'We', 'Its', 'Their', 'His', 'Her',
            'For', 'With', 'From', 'Into', 'More', 'Most',
            'And', 'But', 'Not', 'Also', 'Such', 'Each',
            'However', 'Moreover', 'Furthermore', 'Therefore',
            'Section', 'Figure', 'Table', 'Equation',
        }
        
        for kw in fallback_keywords:
            if kw not in queries:
                context = citation_contexts.get(kw, "")
                
                # Strip citation/target markers before extracting entities
                clean_ctx = re.sub(r'\*\*\[/?TARGET\]\*\*', ' ', context)
                clean_ctx = re.sub(r'\[CITE:[^\]]*\]', ' ', clean_ctx)
                clean_ctx = re.sub(r'\\cite\{CITE:[^\}]*\}', ' ', clean_ctx)
                clean_ctx = re.sub(r'\[CTA:[^\]]*\]', ' ', clean_ctx)
                clean_ctx = re.sub(r'\\cite\{missing_[^\}]*\}', ' ', clean_ctx)
                clean_ctx = re.sub(r'\s+', ' ', clean_ctx)
                
                # Extract capitalized multi-word phrases (likely method/model names)
                named_entities = re.findall(
                    r'\b[A-Z][a-zA-Z]*(?:[-][A-Z][a-zA-Z]*)*(?:\s+[A-Z][a-zA-Z]*)*',
                    clean_ctx
                )
                # Deduplicate, filter stopwords
                seen = set()
                unique_entities = []
                for e in named_entities:
                    e_stripped = e.strip()
                    if (e_stripped
                            and e_stripped not in seen
                            and e_stripped not in _MARKER_STOPWORDS
                            and len(e_stripped) > 2):
                        seen.add(e_stripped)
                        unique_entities.append(e_stripped)
                
                if unique_entities:
                    fallback_query = ' '.join(unique_entities[:3])
                    words = fallback_query.split()
                    if len(words) > 6:
                        fallback_query = ' '.join(words[:6])
                else:
                    fallback_query = kw.replace('_', ' ').replace('-', ' ')
                    words = fallback_query.split()
                    if len(words) > 6:
                        fallback_query = ' '.join(words[:6])
                
                # Remove any year numbers from fallback query
                fallback_query = re.sub(r'\b(19|20)\d{2}\b', '', fallback_query).strip()
                fallback_query = re.sub(r'\s+', ' ', fallback_query)
                
                queries[kw] = fallback_query
                logger.info(f"Using context-based fallback query for '{kw}': {queries[kw]}")
        
        return queries
