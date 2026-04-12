"""
Literature-search-driven innovation queue generator.
Ports AgentArxiv's literature-search techniques into innovation-queue initialization and updates.

Features:
- Automatically saves detailed per-LLM-query logs to markdown files.
- Log files are saved under <experiment_dir>/logs/ by default.
- Captures the full prompt, response, and metadata.
- Toggle logging via the global ENABLE_QUERY_LOGGING variable.
"""

import os
import sys
import json
import time
import logging
import random
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from .backend import backend_openai

from camyla.treesearch.backend.utils import FunctionSpec

import json_repair
from camyla.infrastructure.literature import MultiSourceLiteratureSearch
from camyla.treesearch.utils.text_manager import clean_unicode_text_global, \
estimate_token_count, format_token_count, save_query_log, extract_prompt
from skills import load_skill

logger = logging.getLogger(__name__)

# Global toggle: whether to enable query logging
ENABLE_QUERY_LOGGING = True  # Set to False to disable query logging

# ============================================================================
# Function Specifications for Literature Search
# ============================================================================
# LITERATURE_REVIEW function specification - contains every available function
LITERATURE_REVIEW_FUNCTION_SPEC = FunctionSpec(
    name="literature_review_actions",
    description="Available actions for literature review: search papers or get full text (which automatically adds the paper to your review)",
    json_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search_papers", "get_full_text"],
                "description": "The action to perform. Note: get_full_text will automatically retrieve, analyze, and ADD the paper to your review."
            },
            "query": {
                "type": "string",
                "description": "Search query (required for search_papers action)"
            },
            "paper_id": {
                "type": "string",
                "description": "Paper ID (required for get_full_text action). WARNING: Calling get_full_text will automatically add this paper to your review, so only use it on papers you want to include."
            }
        },
        "required": ["action"]
    }
)

# ============================================================================
# Function calling utilities
# ============================================================================

def query_model(prompt, system_prompt, model_str="deepseek/deepseek-v3.2", temp=None, func_spec=None, **kwargs):
    # Sanitize Unicode in input text
    if prompt:
        prompt = clean_unicode_text_global(prompt)
    if system_prompt:
        system_prompt = clean_unicode_text_global(system_prompt)

    # Count input tokens
    system_tokens = estimate_token_count(system_prompt) if system_prompt else 0
    prompt_tokens = estimate_token_count(prompt)
    total_input_tokens = system_tokens + prompt_tokens

    '''
    print('=' * 80)
    print(f'🤖 LLM Query - Model: {model_str}')
    print(f'📊 Input Token Analysis:')
    print(f'   System Prompt: {format_token_count(system_tokens):>6s} tokens')
    print(f'   User Prompt:   {format_token_count(prompt_tokens):>6s} tokens')
    print(f'   Total Input:   {format_token_count(total_input_tokens):>6s} tokens')
    '''
    if func_spec:
        logger.debug(f'Function Calling: {func_spec.name}')

    try:
        # Prepare model parameters
        model_kwargs = {
            "model": model_str,
            "temperature": temp or 0.7,
            **kwargs
        }

        # Call backend_openai.query
        output, req_time, in_tokens, out_tokens, info = backend_openai.query(
            system_message=system_prompt,
            user_message=prompt,
            func_spec=func_spec,
            **model_kwargs
        )

        # Compute output token count (estimate when backend does not report it)
        actual_output_tokens = out_tokens if out_tokens else estimate_token_count(str(output))

        '''
        print(f'📈 LLM Response:')
        print(f'   Request Time:  {req_time:.2f}s')
        print(f'   Input Tokens:  {format_token_count(in_tokens if in_tokens else total_input_tokens)}')
        print(f'   Output Tokens: {format_token_count(actual_output_tokens)}')
        print(f'   Total Tokens:  {format_token_count((in_tokens if in_tokens else total_input_tokens) + actual_output_tokens)}')
        print('=' * 80)
        '''

        logger.info(f"LLM query completed in {req_time:.2f}s, tokens: {in_tokens}/{out_tokens}")

        # Save the query log to a markdown file
        save_query_log(
            prompt=prompt,
            system_prompt=system_prompt,
            output=output,
            model_str=model_str,
            func_spec=func_spec,
            req_time=req_time,
            in_tokens=in_tokens if in_tokens else total_input_tokens,
            out_tokens=actual_output_tokens
        )

        # Return the output (preserving the original interface)
        return output

    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        raise

# ============================================================================
# Simplified Agent class (ported from agentarxiv)
# ============================================================================

class BaseAgent:
    """Base agent class."""

    def __init__(self, model="deepseek/deepseek-v3.2", notes=None, max_steps=100):
        self.notes = notes or []
        self.max_steps = max_steps
        self.model = model
        self.phases = []
        self.plan = ""
        self.report = ""
        self.history = []
        self.prev_comm = ""
        self.lit_review_sum = ""
        self.lit_review = []

    def role_description(self):
        """Role description."""
        raise NotImplementedError("Subclasses should implement this method.")

    def phase_prompt(self, phase):
        """Stage prompt."""
        raise NotImplementedError("Subclasses should implement this method.")

    def command_descriptions(self, phase):
        """Command description."""
        raise NotImplementedError("Subclasses should implement this method.")

    def context(self, phase):
        """Context info."""
        return ""

    @staticmethod
    def clean_text(text):
        """Clean up text."""
        # If text is a tuple, take the first element
        if isinstance(text, tuple):
            text = text[0]
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("```\n", "```")
        return text

    def inference(self, research_topic, phase, step=0, feedback="", temp=None):
        """Modern inference generation using function calling."""
        sys_prompt = f"You are {self.role_description()}\n[Objective] Your goal is to perform research on the following topic: {research_topic}\n{self.phase_prompt(phase)}"
        context = self.context(phase)

        # Build the history and compute token counts
        history_parts = []
        total_history_tokens = 0

        logger.debug(f"History Token Analysis ({len(self.history)} records total):")
        for i, record in enumerate(self.history):
            record_tokens = estimate_token_count(record)
            total_history_tokens += record_tokens
            history_parts.append(record)

            record_preview = record[:200].replace('\n', ' ') + "..." if len(record) > 200 else record.replace('\n', ' ')
            logger.debug(f"  {i+1:2d}. {format_token_count(record_tokens):>6s} tokens - {record_preview}")

        history_str = "\n".join(history_parts)
        logger.debug(f"Total History Tokens: {format_token_count(total_history_tokens)}")

        phase_notes = [_note for _note in self.notes if phase in _note.get("phases", [])]
        notes_str = f"Notes for the task objective: {phase_notes}\n" if len(phase_notes) > 0 else ""

        complete_str = ""
        if step/(self.max_steps-1) > 0.7:
            complete_str = "You must finish this task and submit as soon as possible!"

        # Add a mandatory function-calling reminder for the literature-review stage
        function_reminder = ""
        if phase == "literature review":
            function_reminder = (
                "\n🚨 **MANDATORY FUNCTION CALLING REMINDER** 🚨\n"
                "- You MUST call the literature_review_actions function\n"
                "- DO NOT provide text responses or summaries\n"
                "- ALWAYS use function calls for your actions\n"
                "- If you provide text without function call, you are FAILING the task\n\n"
            )

        prompt = (
            f"{context}\n{'~' * 10}\nHistory: {history_str}\n{'~' * 10}\n"
            f"Current Step #{step}, Phase: {phase}\n{complete_str}\n"
            f"Feedback: {feedback}\nNotes: {notes_str}\n"
            f"Your previous command was: {self.prev_comm}. Make sure your new output is very different.\n"
            f"{function_reminder}"
        )

        # Expose all available functions for the literature-review stage so the agent picks autonomously
        func_spec = self._get_available_functions(phase)
        
        search_indices = [i for i, record in enumerate(self.history) if "Search completed successfully" in record]
        # If multiple matches are found
        if len(search_indices) > 1:
            logger.debug(f"More than one search record; truncating intelligently...")
            # Iterate over every index except the last (most recent) one
            for i in search_indices[:-1]:
                
                # old_record = self.history[i]
                # simplified_record = self._simplify_search_record(old_record)
                # self.history[i] = simplified_record
                self.history[i] = "Detailed list simplified"
                
                history_parts = []
                logger.debug(f"Updated History Token Analysis ({len(self.history)} records total):")
                for i, record in enumerate(self.history):
                    record_tokens = estimate_token_count(record)
                    total_history_tokens += record_tokens
                    history_parts.append(record)

                    record_preview = record[:200].replace('\n', ' ') + "..." if len(record) > 200 else record.replace('\n', ' ')
                    logger.debug(f"  {i+1:2d}. {format_token_count(record_tokens):>6s} tokens - {record_preview}")

                history_str = "\n".join(history_parts)
                
            # Rebuild the prompt
            prompt = (
                f"{context}\n{'~' * 10}\nHistory: {history_parts}\n{'~' * 10}\n"
                f"Current Step #{step}, Phase: {phase}\n{complete_str}\n"
                f"Feedback: {feedback}\nNotes: {notes_str}\n"
                f"Your previous command was: {self.prev_comm}. Make sure your new output is very different.\n"
            )

        # Use modern function calling
        model_resp = query_model(
            prompt=prompt,
            system_prompt=sys_prompt,
            model_str=self.model,
            temp=temp,
            func_spec=func_spec  # important: pass the function spec
        )

        # Process the response (may be text or structured data)
        if isinstance(model_resp, dict):
            # Function-call response; make sure function_name is present
            function_name = func_spec.name if func_spec else "unknown"
            model_resp['function_name'] = function_name  # attach function_name to the response
            self.prev_comm = f"{function_name}({model_resp})"
            response_text = f"Function call: {function_name} with parameters: {model_resp}"
        else:
            # Plain text response
            model_resp = self.clean_text(model_resp)
            self.prev_comm = model_resp
            response_text = model_resp

        # History handling — remove the EXPIRATION mechanism
        # Strip EXPIRATION markers from feedback (if present)
        if feedback is not None and "```EXPIRATION" in feedback:
            feedback = extract_prompt(feedback, "EXPIRATION")

        # Simplified history storage — no expiration mechanism
        self.history.append(f"Step #{step}, Phase: {phase}, Feedback: {feedback}, Your response: {response_text}")

        return model_resp

    def _smart_truncate_history(self):
        """Smart history truncation: drop token-heavy search results first, keep other valuable operation records."""
        if len(self.history) <= 3:  # keep at least 3 records
            return

        # Analyze the history and identify search-result records (usually long)
        search_indices = []
        other_indices = []
        total_tokens_before = 0

        logger.debug(f"Pre-truncation history token analysis:")
        for i, record in enumerate(self.history):
            record_tokens = estimate_token_count(record)
            total_tokens_before += record_tokens

            if ("Search completed successfully" in record and
                "papers from 2023 to 2025:" in record and
                len(record) > 2000):
                search_indices.append(i)
                record_type = "search result"
            else:
                other_indices.append(i)
                record_type = "other operation"

            record_preview = record[:50].replace('\n', ' ') + "..." if len(record) > 50 else record.replace('\n', ' ')
            logger.debug(f"  {i+1:2d}. {format_token_count(record_tokens):>6s} tokens [{record_type}] - {record_preview}")

        logger.debug(f"Pre-truncation totals: {len(self.history)} records, {format_token_count(total_tokens_before)} tokens, "
                     f"search results: {len(search_indices)}, other ops: {len(other_indices)}")

        # Drop the oldest search results first
        removed_count = 0
        while len(self.history) > 5 and search_indices:
            oldest_search_idx = search_indices.pop(0)  # remove the oldest search result

            # Capture original record info
            original_record = self.history[oldest_search_idx - removed_count]

            # Simplify the search-result record
            simplified_record = self._simplify_search_record(original_record)

            # Replace with the simplified version
            self.history[oldest_search_idx - removed_count] = simplified_record

            logger.debug(f"Simplified search record #{oldest_search_idx}: {len(original_record)} -> {len(simplified_record)} characters")

            # If the history is still too long, drop the oldest search records entirely
            if len(self.history) > 8:
                removed_record = self.history.pop(oldest_search_idx - removed_count)
                removed_count += 1
                logger.debug(f"Removed search record #{oldest_search_idx}")

                # Update the remaining indices
                search_indices = [idx - 1 for idx in search_indices if idx > oldest_search_idx]
                other_indices = [idx - 1 for idx in other_indices if idx > oldest_search_idx]

        # Only remove other records if the history is still too long (keep where possible)
        while len(self.history) > 6 and other_indices:
            oldest_other_idx = other_indices.pop(0)
            removed_record = self.history.pop(oldest_other_idx - removed_count)
            removed_count += 1
            logger.debug(f"Removed other record #{oldest_other_idx}")

            # Update the remaining indices
            other_indices = [idx - 1 for idx in other_indices if idx > oldest_other_idx]

        # Compute post-truncation token stats
        total_tokens_after = 0
        for record in self.history:
            total_tokens_after += estimate_token_count(record)

        tokens_saved = total_tokens_before - total_tokens_after
        logger.debug(f"Truncation complete: kept {len(self.history)} records, "
                     f"tokens saved: {format_token_count(tokens_saved)} ({format_token_count(total_tokens_before)} -> {format_token_count(total_tokens_after)})")

    def _simplify_search_record(self, original_record):
        """Simplify a search-result record: keep key info while reducing token usage."""
        lines = original_record.split('\n')

        # Extract key fields
        step_info = ""
        paper_count = 0
        query_info = ""

        for line in lines:
            if line.startswith("Step #"):
                step_info = line.split(", Your response:")[0]  # keep only the step info; drop the detailed response
            elif "Search completed successfully" in line:
                # Extract paper count
                if "Found" in line and "papers from 2023 to 2025" in line:
                    import re
                    match = re.search(r'Found (\d+) papers', line)
                    if match:
                        paper_count = int(match.group(1))
                break
            elif "'query':" in line and not query_info:
                # Extract query info (simplified)
                import re
                query_match = re.search(r"'query':\s*'([^']+)'", line)
                if query_match:
                    query_info = query_match.group(1)[:50]  # cap query length

        # Build a minimal record
        simplified = f"{step_info}, Your response: search_papers(query='{query_info}')\n"
        simplified += f"Found {paper_count} papers (detailed list simplified)"

        return simplified

    def _get_available_functions(self, phase):
        """Return the function spec available for the current stage so the agent can choose autonomously."""
        if phase == "literature review":
            return LITERATURE_REVIEW_FUNCTION_SPEC
        else:
            return None


class CitationNetworkAgent(BaseAgent):
    """Citation-network agent — extracts referenced papers from full text and generates search keywords."""

    def __init__(self, model="deepseek/deepseek-v3.2", notes=None, max_steps=5):
        super().__init__(model, notes, max_steps)
        self.phases = ["citation_analysis"]
        self.extracted_keywords = []

    def role_description(self):
        return "an expert AI researcher specializing in analyzing academic papers and extracting citation networks to identify research trends and generate search keywords."

    def phase_prompt(self, phase):
        if phase == "citation_analysis":
            return load_skill("agents/citation_network_agent.md")
        else:
            return "Your goal is to analyze citation networks and generate search keywords."

    def command_descriptions(self, phase):
        if phase == "citation_analysis":
            return (
                "Analyze the provided full paper text and perform citation network analysis in JSON format.\n\n"
                "**ANALYSIS GUIDELINES:**\n"
                "1. Thoroughly scan the references/bibliography section\n"
                "2. Extract complete citation information for each reference\n"
                "3. Apply filtering criteria: 2023+, good quality venues, application-focused architecture innovations\n"
                "4. Avoid overly famous works, prioritize emerging and specialized research\n"
                "5. Include domain-specific conferences and journals relevant to practical applications\n"
                "6. Generate NOVEL, CUTTING-EDGE keywords from THIS PAPER'S citations (use the STYLE of: 'Dynamic Aggregation', 'Token-driven', 'Decoupled', but extract YOUR OWN terms)\n"
                "7. AVOID mature baseline methods (e.g., 'Shifted Windows', 'Cross-Attention', 'Dilated Convolution', 'Gated mechanism')\n"
                "8. DO NOT copy the example keywords directly - they are STYLE REFERENCES only\n"
                "9. DO NOT use generic terms like 'architecture', 'network', 'design' or year numbers\n\n"
                "**JSON OUTPUT REQUIREMENTS:**\n"
                "- 'all_citations': Complete list of all found citations\n"
                "- 'filtered_citations': High-quality recent application-focused architecture papers only\n"
                "- 'search_keywords': 5 EMERGING, DISTINCTIVE technical innovations from 2023-2025 top-tier venues (NOT mature baseline methods)\n\n"
                "**CRITICAL:** Output must be valid JSON only. No markdown, no explanations, no additional text."
            )
        else:
            return ""

    def context(self, phase):
        return ""

    def analyze_citations_from_full_text(self, paper_id: str, full_text: str, paper_title: str = "", core_theme: str = "") -> Dict[str, Any]:
        """
        Analyze the citation network from full paper text and generate search keywords.

        Args:
            paper_id: paper ID.
            full_text: full paper text.
            paper_title: paper title (optional).
            core_theme: core theme (optional) used to constrain keyword extraction.

        Returns:
            Dict containing citation-analysis results and search keywords.
        """
        logger.debug(f"Starting citation-network analysis: {paper_id}, core theme: {core_theme}")

        if not full_text or len(full_text.strip()) < 100:
            logger.error(f"Paper full text is empty or too short: {paper_id}")
            raise RuntimeError(f"Paper full text is empty or too short: {paper_id}, length: {len(full_text) if full_text else 0}")

        try:
            # Clear history so each analysis is independent
            self.history = []

            # Build the analysis prompt
            analysis_prompt = self._build_citation_analysis_prompt(paper_id, full_text, paper_title, core_theme)

            # Use the agent to run citation analysis
            citation_analysis = self._direct_citation_analysis(analysis_prompt)

            # Handle the analysis result
            if isinstance(citation_analysis, str) and len(citation_analysis.strip()) > 50:
                analyzed_info = self._parse_citation_analysis(citation_analysis)
                analyzed_info["paper_id"] = paper_id
                analyzed_info["paper_title"] = paper_title
                analyzed_info["core_theme"] = core_theme

                # Append to the list of analyzed entries
                keywords = analyzed_info.get('search_keywords', [])
                self.extracted_keywords.extend(keywords)

                logger.debug(f"Citation analysis complete: {paper_id}, citations: {len(analyzed_info.get('all_citations', []))}, "
                            f"filtered citations: {len(analyzed_info.get('filtered_citations', []))}, keywords: {len(keywords)}")

                return analyzed_info
            else:
                logger.error(f"Citation analysis result too short: {paper_id}, length: {len(citation_analysis) if citation_analysis else 0}")
                raise RuntimeError(f"Citation analysis result too short: {paper_id}, response length: {len(citation_analysis) if citation_analysis else 0}")

        except Exception as e:
            logger.error(f"Error during citation analysis: {paper_id}, error: {e}")
            raise RuntimeError(f"Citation analysis failed: {paper_id}, error: {e}")

    def _build_citation_analysis_prompt(self, paper_id: str, full_text: str, paper_title: str, core_theme: str = "") -> str:
        """Build the citation-analysis prompt."""
        # Report raw-text token info
        original_tokens = estimate_token_count(full_text)
        logger.debug(f"Raw paper full text: {format_token_count(original_tokens)} tokens ({len(full_text)} characters)")

        # Build architecture-component constraints depending on whether a core theme is provided
        theme_constraint = ""
        if core_theme:
            theme_constraint = f"""
🎯 CORE RESEARCH ARCHITECTURE: {core_theme}

KEYWORD EXTRACTION STRATEGY:
The overall research is building a '{core_theme}'-based system. We need to find COMPLEMENTARY COMPONENTS and BUILDING BLOCKS.

Extract keywords that represent:

1. **Functional Components** (50% of keywords): Technical modules/mechanisms that COULD work within '{core_theme}' framework
   - Focus on FUNCTION and PURPOSE, not the core theme name itself
   - Examples: "Dynamic Routing", "Hierarchical Aggregation", "Attention Distillation"
   - Think: What functional components would a '{core_theme}'-based system need?

2. **Related Techniques** (30% of keywords): Emerging methods that naturally align with '{core_theme}' principles
   - Can include domain-specific terms, but avoid forcing '{core_theme}' prefix
   - Examples: "Multi-scale Learning", "Token-driven Processing", "Structure-Aware Encoding"
   - Think: What complementary techniques enhance '{core_theme}'?

3. **Synergistic Concepts** (20% of keywords): Complementary technical directions
   - Should enhance '{core_theme}' when combined, but standalone valuable
   - Examples: "Adaptive Keyframe Sampling", "Geometric Embedding", "Adaptive Calibration"
   - Think: What concepts work well with '{core_theme}' but bring new perspectives?

KEYWORD DIVERSITY REQUIREMENTS:
✅ Extract keywords representing DIFFERENT aspects of potential components
✅ Mix architectural patterns, attention mechanisms, aggregation/fusion methods
✅ Ensure at least 3 out of 5 keywords are functionally descriptive (not theme-prefixed)
✅ Think: "What components would I need to build a complete '{core_theme}'-based architecture?"

CRITICAL RULES:
❌ AVOID all 5 keywords being '{core_theme}-XXX' format
❌ AVOID repetitive or overly similar keywords
❌ AVOID keywords that are just variations of '{core_theme}' itself
✅ PREFER descriptive functional terms over theme-prefixed terms

GOAL: Find diverse building blocks for a '{core_theme}'-based architecture, not just '{core_theme}' variations.

"""

        instruction_text = """
CITATION NETWORK ANALYSIS TASK:
Paper ID: {paper_id}
Paper Title: {paper_title}

FULL PAPER TEXT:
{truncated_text}

ANALYSIS INSTRUCTIONS:
Analyze this paper and perform citation network analysis. Complete ALL THREE tasks in a single JSON response:

TASK 1: Extract ALL citations from the references/bibliography section
TASK 2: Filter citations (2023+, good quality venues, application-focused architecture innovations)
TASK 3: Generate 5 HIGHLY SPECIFIC search keywords for finding related technical innovations

""" + theme_constraint + """
⚠️ CRITICAL KEYWORD REQUIREMENTS:
Your search keywords MUST be:
✅ SPECIFIC technical terms, module names, or mechanisms (2-4 words each)
✅ Focused on NOVEL, CUTTING-EDGE architectural components from RECENT top-tier papers (CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR)
✅ Extract EMERGING techniques that represent NEW research directions (2023-2025)
✅ DO NOT include generic words like "architecture", "network", "design", "model", "framework"
✅ DO NOT include year numbers (2023, 2024, 2025, etc.)
✅ AVOID mature, widely-adopted techniques from 2-3 years ago (they are now baseline methods)

"STRICTLY AVOID these topics (DO NOT search for):\n"
"- Data augmentation, input enhancement, preprocessing techniques\n"
"- Unsupervised learning, self-supervised learning, semi-supervised learning\n"
"- Pre-training strategies, transfer learning, domain adaptation\n"
"- Training techniques, optimization methods, loss functions\n"
"- Simple attention mechanisms (unless very novel architecture)\n"
"- Data-related innovations, dataset construction, labeling strategies\n\n"
"- Diffusion models, diffusion-based methods\n"

📋 GOOD KEYWORD STYLE REFERENCE (DO NOT use these exact terms, extract YOUR OWN from the paper):
✓ "Dynamic Aggregation" style - recent innovative aggregation approach
✓ "Token-driven" style - emerging token-based paradigm
✓ "Decoupled" style - novel decoupling strategy
✓ "Global local learning" style - new learning paradigm
✓ "Structure-Aware" style - emerging xxx-aware methods
✓ "Motion-guidance Fuser" style - innovative fusion mechanism

⚠️ CRITICAL: Extract NEW keywords from THIS PAPER'S citations, NOT the examples above!

❌ BAD KEYWORD EXAMPLES (AVOID - generic, mature, or outdated):
✗ "novel network architecture" - too generic
✗ "Shifted Windows" - mature Swin technique from 2021, now baseline
✗ "Cross-Attention" - too common, widely used, not distinctive
✗ "Dilated Convolution" - old technique from 2015, not novel
✗ "Gated mechanism" - mature technique, widely adopted
✗ "Dual-Dilated Convolution" - variant of old technique
✗ "efficient CNN design" - includes generic "design"
✗ "transformer-based model 2024" - includes year and "model"

🎯 EXTRACTION STRATEGY:
- Carefully READ the paper's references and extract NOVEL technical terms used in recent citations
- Focus on DISTINCTIVE innovations from the MOST RECENT papers (2023-2025)
- Look for terms that represent EMERGING research directions, not incremental improvements
- Extract techniques that appear in TOP-TIER conferences (CVPR, ICCV, ECCV, NeurIPS, ICML)
- AVOID mature baseline methods that are already widely adopted in the field
- DO NOT simply copy the example keywords - extract YOUR OWN from the actual paper citations

CRITICAL OUTPUT REQUIREMENT:
You MUST respond with ONLY a valid JSON object in this exact format:

{{
  "all_citations": [
    {{
      "title": "Paper Title",
      "authors": "Author names",
      "year": "2024",
      "venue": "Conference/Journal name"
    }}
  ],
  "filtered_citations": [
    {{
      "title": "Application-Focused Architecture Paper",
      "authors": "Author names",
      "year": "2023",
      "venue": "Domain-Specific Conference",
      "relevance_reason": "Why this paper is relevant to practical architecture innovations"
    }}
  ],
  "search_keywords": [
    "keyword1",
    "keyword2",
    "keyword3",
    "keyword4",
    "keyword5"
  ]
}}

STRICT RULES:
- Output ONLY valid JSON, no markdown, no explanations
- Complete all three tasks in one response
- Focus on practical application-oriented network architecture innovations
- Avoid overly famous works, prioritize emerging and specialized research
- Include domain-specific conferences and journals, not limited to top-tier venues
- Extract SPECIFIC technical terms that can be searched effectively across CV/AI domains
"""

        prompt = instruction_text.format(
            paper_id=paper_id,
            paper_title=paper_title,
            truncated_text=full_text
        )

        # Report final prompt token info
        final_prompt_tokens = estimate_token_count(prompt)
        logger.debug(f"Final analysis prompt: {format_token_count(final_prompt_tokens)} tokens")

        return prompt

    def _direct_citation_analysis(self, analysis_prompt: str, max_retries: int = 3) -> str:
        """Directly call the LLM for citation analysis, with retries."""
        system_prompt = f"You are {self.role_description()}\n{self.phase_prompt('citation_analysis')}"

        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.debug(f"LLM citation-analysis attempt {attempt + 1}/{max_retries}")

                # Call query_model directly instead of using inference
                response = query_model(
                    prompt=analysis_prompt,
                    system_prompt=system_prompt,
                    model_str=self.model,
                    temp=0.3
                )

                if response and isinstance(response, str) and len(response.strip()) > 20:
                    logger.debug(f"LLM citation analysis succeeded; response length: {len(response)} characters")
                    return response
                else:
                    raise RuntimeError(f"LLM response empty or too short: {len(response) if response else 0} characters")

            except Exception as e:
                last_exception = e
                logger.warning(f"LLM citation-analysis attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    logger.debug(f"Waiting before retry...")
                    import time
                    time.sleep(2 ** attempt)  # exponential backoff

        # All retries failed
        logger.error(f"All LLM citation-analysis retries failed; last error: {last_exception}")
        raise RuntimeError(f"LLM citation analysis failed after {max_retries} retries; last error: {last_exception}")

    def _parse_citation_analysis(self, citation_analysis: str) -> Dict[str, Any]:
        """Parse the citation-analysis result; must be valid JSON."""
        try:
            # Clean response text; strip possible markdown code-block markers
            cleaned_text = citation_analysis.strip()

            # If the response wraps JSON in a markdown code block, extract it
            if "```json" in cleaned_text:
                import re
                json_pattern = r'```json\s*(.*?)\s*```'
                json_match = re.search(json_pattern, cleaned_text, re.DOTALL)
                if json_match:
                    cleaned_text = json_match.group(1).strip()
                else:
                    # When no full code block is found, try to extract starting from ```json
                    start_idx = cleaned_text.find('```json')
                    if start_idx != -1:
                        cleaned_text = cleaned_text[start_idx + 7:]  # skip the ```json marker
                        # Remove the trailing ```
                        if cleaned_text.endswith('```'):
                            cleaned_text = cleaned_text[:-3]
                        cleaned_text = cleaned_text.strip()

            # Try to parse as JSON
            json_data = json_repair.loads(cleaned_text)

            if not isinstance(json_data, dict):
                raise ValueError(f"Response is not a valid JSON object: {type(json_data)}")

            # Validate required fields
            required_fields = ["all_citations", "filtered_citations", "search_keywords"]
            for field in required_fields:
                if field not in json_data:
                    raise ValueError(f"Missing '{field}' in the JSON response")

            # Validate field types
            if not isinstance(json_data["all_citations"], list):
                raise ValueError(f"'all_citations' must be an array; got: {type(json_data['all_citations'])}")

            if not isinstance(json_data["filtered_citations"], list):
                raise ValueError(f"'filtered_citations' must be an array; got: {type(json_data['filtered_citations'])}")

            if not isinstance(json_data["search_keywords"], list):
                raise ValueError(f"'search_keywords' must be an array; got: {type(json_data['search_keywords'])}")

            # Validate keyword count
            keywords = json_data["search_keywords"]
            if len(keywords) == 0:
                raise ValueError("'search_keywords' is empty — no keywords were generated")

            # JSON parsed successfully
            parsed_info = {
                "analysis_result": citation_analysis,
                "json_data": json_data,
                "all_citations": json_data["all_citations"],
                "filtered_citations": json_data["filtered_citations"],
                "search_keywords": keywords
            }

            logger.debug(f"Citation-analysis JSON parsed successfully; citations: {len(json_data['all_citations'])}, "
                        f"filtered citations: {len(json_data['filtered_citations'])}, keywords: {len(keywords)}")
            return parsed_info

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Cleaned text: {cleaned_text[:500]}...")
            raise RuntimeError(f"LLM response is not valid JSON: {e}")

        except ValueError as e:
            logger.error(f"JSON validation failed: {e}")
            raise RuntimeError(f"JSON validation failed: {e}")

        except Exception as e:
            logger.error(f"Unknown error while parsing citation analysis: {e}")
            raise RuntimeError(f"Failed to parse citation analysis: {e}")

    def get_all_extracted_keywords(self) -> List[str]:
        """Return all extracted keywords."""
        return self.extracted_keywords

    def format_citations_for_prompt(self) -> str:
        """Format extracted citation info as text suitable for prompts."""
        # Use this to forward citation info to other agents
        return f"Extracted {len(self.extracted_keywords)} search keywords from citation analysis."


class PaperSummaryAgent(BaseAgent):
    """Paper-methods summarization agent — extracts network structure and module-level innovations from papers."""

    def __init__(self, model="moonshotai/kimi-k2", notes=None, max_steps=10):
        super().__init__(model, notes, max_steps)
        self.phases = ["method_extraction"]
        self.extracted_methods = []

    def role_description(self):
        return "an expert AI researcher specializing in analyzing deep learning papers and extracting technical methodologies, network architectures, and novel modules."

    def phase_prompt(self, phase):
        if phase == "method_extraction":
            return load_skill("agents/paper_summary_agent.md")
        else:
            return "Your goal is to extract technical methodologies from research papers in JSON format."

    def command_descriptions(self, phase):
        if phase == "method_extraction":
            return (
                "Analyze the provided paper text to extract its core architectural innovations as self-contained modules "
                "in JSON format.\n\n"
                "**EXTRACTION GUIDELINES:**\n"
                "1. **Identify Complete Contributions:** Focus on whole modules or blocks (e.g., 'Cross-Attention Fusion Block'), not their individual layers.\n"
                "2. **Detail Each Contribution:** For each innovation, describe its purpose, internal architecture, data flow, and mathematical formulas.\n"
                "3. **Provide Pseudocode:** Include a minimal, clear pseudocode or code implementation for each module.\n"
                "4. **Architectural Focus:** Extract only network architecture details.\n"
                "5. **Strictly Ignore Training:** Omit all details about training, loss, optimizers, and experiments.\n\n"
                "**JSON OUTPUT REQUIREMENTS:**\n"
                "- Each innovation is a separate object in the 'innovations' array.\n"
                "- 'name': The official name of the module.\n"
                "- 'description': A structured, comprehensive technical explanation of the module.\n"
                "- 'implementation': Minimal code or clear pseudocode for the module.\n\n"
                "**CRITICAL:** Output must be a single, valid JSON object only. Do not include any explanatory text or markdown formatting outside the JSON structure."
            )
        else:
            return ""

    def context(self, phase):
        return ""

    def extract_methods_from_full_text(self, paper_id: str, full_text: str, paper_title: str = "") -> Dict[str, str]:
        """
        Extract method content from paper full text.

        Args:
            paper_id: paper ID.
            full_text: full paper text.
            paper_title: paper title (optional).

        Returns:
            Dict containing the extracted method content.
        """
        logger.debug(f"Starting method extraction: {paper_id}")

        if not full_text or len(full_text.strip()) < 100:
            logger.error(f"Paper full text is empty or too short: {paper_id}")
            raise RuntimeError(f"Paper full text is empty or too short: {paper_id}, length: {len(full_text) if full_text else 0}")

        try:
            # Clear history so each extraction is independent
            self.history = []

            # Build the extraction prompt
            extraction_prompt = self._build_extraction_prompt(paper_id, full_text, paper_title)

            # Run method extraction via the agent (calling the LLM directly instead of using inference)
            method_summary = self._direct_method_extraction(extraction_prompt)

            # Handle the extraction result
            if isinstance(method_summary, str) and len(method_summary.strip()) > 50:
                extracted_info = self._parse_method_summary(method_summary)
                extracted_info["paper_id"] = paper_id
                extracted_info["paper_title"] = paper_title

                # Append to the list of extracted methods
                self.extracted_methods.append(extracted_info)

                logger.debug(f"Method extraction complete: {paper_id}, length: {len(extracted_info.get('summary', ''))} characters")

                return extracted_info
            else:
                logger.error(f"Method extraction result too short: {paper_id}, length: {len(method_summary) if method_summary else 0}")
                raise RuntimeError(f"Method extraction result too short: {paper_id}, response length: {len(method_summary) if method_summary else 0}")

        except Exception as e:
            logger.error(f"Error during method extraction: {paper_id}, error: {e}")
            raise RuntimeError(f"Method extraction failed: {paper_id}, error: {e}")

    def _build_extraction_prompt(self, paper_id: str, full_text: str, paper_title: str) -> str:
        """Build the method-extraction prompt."""
        # Report raw-text token info
        original_tokens = estimate_token_count(full_text)
        logger.debug(f"Raw paper full text: {format_token_count(original_tokens)} tokens ({len(full_text)} characters)")

        instruction_text = """
PAPER ANALYSIS TASK:
Paper ID: {paper_id}
Paper Title: {paper_title}

FULL PAPER TEXT:
{truncated_text}

EXTRACTION INSTRUCTIONS:
Analyze this paper and extract ONLY the technical methodology and network architecture innovations.
Focus exclusively on implementable architectural components that can be directly coded.

CRITICAL OUTPUT REQUIREMENT:
You MUST respond with ONLY a valid JSON object in this exact format:

{{
  "innovations": [
    {{
      "name": "Specific Innovation Name",
      "description": "Describe the technical method following the structure of a top-tier computer science conference paper with mathematical formulations",
      "implementation": "Minimal code implementation or clear pseudocode"
    }}
  ]
}}

STRICT RULES:
- Output ONLY valid JSON, no markdown, no explanations
- Each distinct innovation must be a separate object
- Focus on architecture/modules, ignore training methods completely
- Include precise implementation details for direct coding
- Provide exact mathematical formulations where applicable
"""

        prompt = instruction_text.format(
            paper_id=paper_id,
            paper_title=paper_title,
            truncated_text=full_text
        )

        # Report final prompt token info
        final_prompt_tokens = estimate_token_count(prompt)
        logger.debug(f"Final extraction prompt: {format_token_count(final_prompt_tokens)} tokens")

        return prompt

    def _direct_method_extraction(self, extraction_prompt: str, max_retries: int = 3) -> str:
        """Directly call the LLM for method extraction, with retries."""
        system_prompt = f"You are {self.role_description()}\n{self.phase_prompt('method_extraction')}"

        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.debug(f"LLM method-extraction attempt {attempt + 1}/{max_retries}")

                # Call query_model directly instead of using inference
                response = query_model(
                    prompt=extraction_prompt,
                    system_prompt=system_prompt,
                    model_str=self.model,
                    temp=0.3
                )

                if response and isinstance(response, str) and len(response.strip()) > 20:
                    logger.debug(f"LLM method extraction succeeded; response length: {len(response)} characters")
                    return response
                else:
                    raise RuntimeError(f"LLM response empty or too short: {len(response) if response else 0} characters")

            except Exception as e:
                last_exception = e
                logger.warning(f"LLM method-extraction attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    logger.debug(f"Waiting before retry...")
                    import time
                    time.sleep(2 ** attempt)  # exponential backoff

        # All retries failed
        logger.error(f"All LLM method-extraction retries failed; last error: {last_exception}")
        raise RuntimeError(f"LLM method extraction failed after {max_retries} retries; last error: {last_exception}")

    def _parse_method_summary(self, method_summary: str) -> Dict[str, str]:
        """Parse the method summary; must be valid JSON."""
        try:
            # Try to parse as JSON
            json_data = json_repair.loads(method_summary.strip())

            if not isinstance(json_data, dict):
                raise ValueError(f"Response is not a valid JSON object: {type(json_data)}")

            if "innovations" not in json_data:
                raise ValueError("Missing 'innovations' in the JSON response")

            innovations = json_data.get("innovations", [])
            if not isinstance(innovations, list):
                raise ValueError(f"'innovations' must be an array; got: {type(innovations)}")

            if len(innovations) == 0:
                raise ValueError("'innovations' is empty — no innovations were extracted")

            # Validate required fields on each innovation
            for i, innovation in enumerate(innovations):
                if not isinstance(innovation, dict):
                    raise ValueError(f"Innovation {i+1} is not a valid object: {type(innovation)}")

                required_fields = ["name", "description", "implementation"]
                for field in required_fields:
                    if field not in innovation:
                        raise ValueError(f"Innovation {i+1} is missing required field: {field}")

                    if not innovation[field] or not innovation[field].strip():
                        raise ValueError(f"Innovation {i+1} field {field} is empty")

            # JSON parsed successfully — keep only the necessary fields
            parsed_info = {
                "summary": method_summary,
                "json_data": json_data,
                "innovations": innovations
            }

            logger.debug(f"Method summary JSON parsed successfully; contains {len(innovations)} innovations")
            return parsed_info

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            raise RuntimeError(f"LLM response is not valid JSON: {e}")

        except ValueError as e:
            logger.error(f"JSON validation failed: {e}")
            raise RuntimeError(f"JSON validation failed: {e}")

        except Exception as e:
            logger.error(f"Unknown error while parsing method summary: {e}")
            raise RuntimeError(f"Failed to parse method summary: {e}")



    def get_all_extracted_methods(self) -> List[Dict[str, str]]:
        """Return all extracted methods."""
        return self.extracted_methods

    def format_methods_for_prompt(self) -> str:
        """Format extracted methods as prompt-ready text, preferring the detailed JSON form."""
        if not self.extracted_methods:
            return "No technical methods extracted yet."

        formatted_methods = []
        for i, method in enumerate(self.extracted_methods, 1):
            paper_title = method.get('paper_title', 'Unknown Title')
            paper_id = method.get('paper_id', 'Unknown')

            method_text = f"\nPaper {i}: {paper_title} (ID: {paper_id})\n"

            # Prefer the detailed JSON info when available
            if method.get('json_data') and method['json_data'].get('innovations'):
                method_text += "\n🔧 INNOVATIONS:\n"

                for j, innovation in enumerate(method['json_data']['innovations'], 1):
                    method_text += f"\n  {j}. {innovation.get('name', 'Unknown Innovation')}\n"
                    method_text += f"     Description: {innovation.get('description', 'No description')}\n"

                    if innovation.get('implementation'):
                        method_text += f"     Implementation: {innovation['implementation']}\n"

            else:
                # Fall back to the raw summary
                method_text += f"\nExtracted Content: {method.get('summary', 'No method content extracted.')}\n"

            formatted_methods.append(method_text)

        return "\n" + "="*80 + "\n".join(formatted_methods) + "\n" + "="*80



class PhDStudentAgent(BaseAgent):
    """PhD-student agent."""

    def __init__(self, model="deepseek/deepseek-v3.2", notes=None, max_steps=100):
        super().__init__(model, notes, max_steps)
        self.phases = [
            "literature review",
            "plan formulation",
            "running experiments",
            "results interpretation",
            "report writing",
            "report refinement",
        ]
        self.lit_review = []

    def role_description(self):
        return "a computer science PhD student at a top university."

    def phase_prompt(self, phase):
        if phase == "literature review":
            # Literature-review prompt focused on network architecture and module-level innovation
            phase_str = (
                "🔧 **CRITICAL: YOU MUST ALWAYS USE FUNCTION CALLS**\n"
                "- You MUST use the literature_review_actions function for ALL actions\n"
                "- NEVER provide text summaries or responses without calling a function\n"
                "- If you want to search, call literature_review_actions with action='search_papers'\n"
                "- If you want to get full text AND ADD a paper, call literature_review_actions with action='get_full_text'\n"
                "  get_full_text will AUTOMATICALLY ADD the paper to your review!\n"
                "  Only use get_full_text on papers you're confident are relevant and high-quality.\n"
                "- ALWAYS respond with a function call, not plain text\n"
                "- Even if you think the review is complete, you must still call a function\n\n"
                "Your goal is to perform a comprehensive literature review to find innovative NETWORK ARCHITECTURES and ARCHITECTURAL MODULES in image segmentation.\n"
                "STRICT FOCUS: ONLY search for papers about:\n"
                "- Novel network architectures\n"
                "- Innovative architectural modules and components \n"
                "- Advanced network design principles and architectural innovations\n\n"
                "⚠️ IMPORTANT SEARCH QUERY GUIDELINES:\n"
                "- DO NOT include year numbers (like 2023, 2024, 2025) in your search queries. The system automatically filters papers by publication date.\n"
                "- DO NOT include task-specific terms (like 'Medical Image Segmentation', 'Semantic Segmentation') - search BROADLY across computer vision\n"
                "- Use EMERGING, CUTTING-EDGE technical terms ALONE (e.g., 'Dynamic Aggregation' NOT 'Dynamic Aggregation Medical Segmentation')\n"
                "- Good query style (NOVEL techniques, BROAD search): 'Dynamic Aggregation', 'Token-driven', 'Decoupled', 'Global local learning', 'Structure-Aware' (DO NOT use these terms, extract YOUR OWN unique and high-influenced query)\n"
                "- Bad query examples: 'Shifted Windows', 'Cross-Attention', 'Dilated Convolution', 'Dynamic Aggregation Medical Image Segmentation' (too narrow)\n"
                "- AVOID generic terms like 'novel architecture', 'efficient network design', 'xxx model'\n"
                "- Search queries are automatically limited to cs.CV, cs.AI, and eess.IV categories\n"
                "- Your queries should find papers across ALL CV domains, not just your specific task\n\n"
                "- ⚠️ DO NOT copy these examples directly - extract YOUR OWN technical terms from the actual paper\n"
                "SEARCH DOMAINS: Look across ALL domains of computer vision and deep learning:\n"
                "- Any domain with architectural innovations that can be adapted\n\n"
                "ADDITIONAL GUIDELINES:\n"
                "- Prefer application-oriented architectural innovations with practical impact and implementable designs\n"
                "- Avoid overly famous or mainstream works; prioritize emerging, specialized, and domain-specific research\n"
                "STRICTLY AVOID these topics (DO NOT search for):\n"
                "- Data augmentation, input enhancement, preprocessing techniques\n"
                "- Unsupervised learning, self-supervised learning, semi-supervised learning\n"
                "- Pre-training strategies, transfer learning, domain adaptation\n"
                "- Training techniques, optimization methods, loss functions\n"
                "- Simple attention mechanisms (unless very novel architecture)\n"
                "- Data-related innovations, dataset construction, labeling strategies\n\n"
                "- Diffusion models, diffusion-based methods\n"
                "You have access to arXiv and can perform literature review actions using the literature_review_actions function.\n\n"
                "RECOMMENDED WORKFLOW:\n"
                "1. Use literature_review_actions(action='search_papers', query='...') to find paper lists\n"
                "2. CAREFULLY evaluate titles and abstracts from search results\n"
                "3. Choose ONE most promising paper with clear architectural innovations from top-tier venues\n"
                "4. Use literature_review_actions(action='get_full_text', paper_id='...') to retrieve and ADD it\n"
                "   → This will automatically analyze and add the paper to your review\n"
                "5. Search for more papers with different keywords and repeat\n\n"
                "DO NOT search multiple times without getting full texts first!\n"
                "REMEMBER: Quality over quantity - only add papers with significant architectural innovations.\n"
                "REMEMBER: You MUST always call a function - never provide plain text responses!"
            )
            # Dynamically append info about already-reviewed papers
            rev_papers = "Papers in your review so far: " + " ".join([_paper["arxiv_id"] for _paper in self.lit_review])
            phase_str += f"\n{rev_papers}" if len(self.lit_review) > 0 else ""
            return phase_str
        elif phase == "plan formulation":
            return load_skill("agents/phd_student/plan_formulation.md")
        elif phase == "running experiments":
            return load_skill("agents/phd_student/running_experiments.md")
        elif phase == "results interpretation":
            return load_skill("agents/phd_student/results_interpretation.md")
        elif phase == "report writing":
            return load_skill("agents/phd_student/report_writing.md")
        elif phase == "report refinement":
            return load_skill("agents/phd_student/report_refinement.md")
        else:
            return "Your goal is to assist in the research process."

    def command_descriptions(self, phase):
        if phase == "literature review":
            return (
                "You have access to the literature_review_actions function. Use this function to perform literature review tasks.\n\n"
                "**FUNCTION USAGE:**\n"
                "Call literature_review_actions with the appropriate action parameter:\n"
                "- action: 'search_papers' - Search for papers using a query string (requires 'query' parameter)\n"
                "- action: 'get_full_text' - Retrieve full text of a specific paper (requires 'paper_id' parameter)\n"
                "- action: 'add_paper_to_review' - Add a relevant paper to your review (requires 'paper_id' and 'summary' parameters)\n\n"
                "**AUTONOMOUS WORKFLOW:**\n"
                "You decide when and how to use this function based on:\n"
                "- Your current progress and needs\n"
                "- The quality and relevance of papers found\n"
                "- Whether you have enough architectural innovations in your review\n\n"
                "**FOCUS AREAS:**\n"
                "- ONLY search for and add papers about NETWORK ARCHITECTURES and ARCHITECTURAL MODULES\n"
                "- Focus on structural innovations, novel layers, and architectural designs\n"
                "- AVOID papers about training techniques, data processing, or learning strategies\n"
                "- Quality over quantity - understand each architecture deeply\n"
                "- Make intelligent decisions about which papers to investigate further\n"
                "- Extensively discuss architectural innovations and structural designs in your summaries\n\n"
                "**DECISION MAKING:**\n"
                "- Start by searching if you need more papers\n"
                "- Get full text only for promising architectural papers\n"
                "- Add papers only if they contain significant architectural innovations\n"
                "- Continue until you have a comprehensive review of architectural advances\n"
            )
        elif phase == "plan formulation":
            return "You can communicate through dialogue to collaborate on research proposals."
        else:
            return ""

    def context(self, phase):
        if phase == "plan formulation":
            return f"Current Literature Review: {self.lit_review_sum}"
        elif phase == "literature review":
            return ""
        else:
            return ""

    def add_review(self, review, arx_eng, agentrxiv=False, GLOBAL_AGENTRXIV=None):
        """Add a paper to the literature review (follows the agentarxiv implementation exactly)."""
        try:
            if agentrxiv:
                arxiv_id = review.split("\n")[0]
                review_text = "\n".join(review.split("\n")[1:])
                full_text = GLOBAL_AGENTRXIV.retrieve_full_text(arxiv_id,)
            else:
                arxiv_id, review_text = review.strip().split("\n", 1)
                full_text = arx_eng.retrieve_full_paper_text(arxiv_id)
            review_entry = {
                "arxiv_id": arxiv_id,
                "full_text": full_text,
                "summary": review_text,
            }
            self.lit_review.append(review_entry)

            # Update the lit-review summary
            self.lit_review_sum = "Provided here is a literature review on this topic:\n" + "\n".join(
                f"arXiv ID: {_l['arxiv_id']}, Summary: {_l['summary']}"
                for _l in self.lit_review)

            return f"Successfully added paper {arxiv_id}", full_text
        except Exception as e:
            return f"Error trying to add review -- bad formatting, try again: {str(e)}. Your provided Arxiv ID might not be valid. Make sure it references a real paper, which can be found using the SUMMARY command.", ""

    def format_review(self):
        """Format the literature review."""
        return "Provided here is a literature review on this topic:\n" + "\n".join(
            f"arXiv ID: {_l['arxiv_id']}, Summary: {_l['summary']}"
            for _l in self.lit_review)

class IdeaGeneratorAgent(BaseAgent):
    """Innovation-idea generation agent — supports multiple LLM backends and personality traits."""

    def __init__(self, name="idea_generator", model="deepseek/deepseek-v3.2", personality="balanced", 
                 temperature=0.7, max_tokens=4096, notes=None, max_steps=5):
        super().__init__(model, notes, max_steps)
        self.name = name
        self.personality = personality
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.phases = ["idea_generation"]

    def role_description(self):
        personality_roles = {
            "highly creative and novel approach focused": "an innovative AI researcher who excels at generating highly creative and unconventional solutions",
            "technically rigorous and implementation focused": "a pragmatic AI researcher who focuses on technically sound and implementable solutions",
            "balanced creativity and technical feasibility": "a well-rounded AI researcher who balances creativity with practical implementation",
            "deep learning architecture innovation specialist": "a deep learning architecture expert specializing in novel network designs and architectural innovations",
            "medical imaging and healthcare AI focused": "a medical AI specialist with deep expertise in healthcare applications and medical imaging"
        }
        return personality_roles.get(self.personality, "an AI researcher specializing in innovation generation")

    def phase_prompt(self, phase):
        if phase == "idea_generation":
            base_prompt = (
                "Your goal is to generate a single, highly innovative research idea based on the provided literature review and technical methods. "
                "Focus on creating a novel approach that builds upon the extracted technical methods from recent papers."
            )
            
            personality_prompts = {
                "highly creative and novel approach focused": (
                    "As a creative innovator, push the boundaries of conventional thinking. "
                    "Generate ideas that are bold, unconventional, and potentially paradigm-shifting. "
                    "Don't be afraid to propose radical departures from existing methods."
                ),
                "technically rigorous and implementation focused": (
                    "As a technical expert, ensure your ideas are grounded in solid engineering principles. "
                    "Focus on practical implementation details, computational efficiency, and real-world feasibility. "
                    "Provide specific technical specifications and consider implementation challenges."
                ),
                "balanced creativity and technical feasibility": (
                    "Strike an optimal balance between innovation and practicality. "
                    "Generate ideas that are both novel and technically sound, with clear implementation pathways. "
                    "Consider both creative potential and engineering constraints."
                ),
                "deep learning architecture innovation specialist": (
                    "Focus specifically on EMERGING, CUTTING-EDGE architectural innovations from recent top-tier papers. "
                    "Generate ideas involving DISTINCTIVE technical innovations that represent NEW research directions, not incremental improvements of mature methods. "
                    "Emphasize architectural novelty over training techniques or data processing."
                ),
                "medical imaging and healthcare AI focused": (
                    "Concentrate on innovations that address specific challenges in medical imaging and healthcare AI. "
                    "Consider clinical workflow integration, interpretability for medical professionals, and domain-specific constraints. "
                    "Ensure relevance to real medical imaging scenarios and patient care."
                )
            }
            
            return base_prompt + "\n\n" + personality_prompts.get(self.personality, "")
        
        return "Your goal is to generate innovative research ideas."

    def command_descriptions(self, phase):
        if phase == "idea_generation":
            return (
                "Generate a single innovative research idea in JSON format based on the provided literature review and extracted technical methods.\n\n"
                "**OUTPUT REQUIREMENTS:**\n"
                "You MUST output a valid JSON object with this exact structure:\n"
                "```json\n"
                "{\n"
                "  \"name\": \"descriptive_innovation_name\",\n"
                "  \"motivation\": \"Technical motivation and rationale for the innovation\",\n"
                "  \"description\": \"Detailed technical description including methodology, architecture, and key innovations\",\n"
                "  \"implementation\": \"Specific implementation details, pseudocode, or algorithmic steps\",\n"
                "  \"advantages\": \"Key advantages and expected benefits\",\n"
                "  \"challenges\": \"Potential implementation challenges and limitations\"\n"
                "}\n"
                "```\n\n"
                "**CRITICAL:** Output ONLY valid JSON. No additional text, explanations, or markdown formatting."
            )
        return ""

    def context(self, phase):
        return ""

    def generate_idea(self, research_topic: str, paper_info: Dict[str, str], paper_methods: str,
                     dataset_constraints: str, 
                     previous_innovations: List[Dict[str, str]] = None,
                     core_theme: str = None,
                     failed_innovation: Dict[str, str] = None,
                     patch_size: list = None) -> Dict[str, str]:
        """
        Generate a single innovation idea.

        Args:
            research_topic: research topic.
            paper_info: paper info.
            paper_methods: extracted technical methods.
            dataset_constraints: dataset constraints.
            previous_innovations: existing innovations list (used to maintain coherence).
            core_theme: core research architecture theme.
            failed_innovation: info about failed innovations (used to avoid known failure modes).

        Returns:
            Innovation-idea dict.
        """
        logger.debug(f"{self.name} ({self.personality[:20]}...) starting innovation idea generation")

        # Clear history
        self.history = []

        paper_id = paper_info.get('arxiv_id', 'unknown')
        paper_summary = paper_info.get('summary', 'No summary available')
        
        # Build context from existing innovations to keep coherence across ideas
        previous_context = ""
        if previous_innovations and len(previous_innovations) > 0:
            previous_context = "\n\n🔗 PREVIOUS INNOVATIONS (maintain thematic coherence):\n"
            for i, prev in enumerate(previous_innovations, 1):
                prev_name = prev.get('name', 'Unknown')
                prev_desc = prev.get('description', '') + "..." if len(prev.get('description', '')) > 200 else prev.get('description', '')
                previous_context += f"{i}. {prev_name}: {prev_desc}\n"
            previous_context += "\n⚠️ YOUR INNOVATION SHOULD:\n"
            previous_context += "- Build upon OR complement the previous innovations\n"
            previous_context += "- Form a coherent technical story with previous work\n"
            previous_context += "- Avoid redundancy - don't repeat what was already proposed\n"
            previous_context += "- Focus on a different aspect or extend the technical pipeline\n"
        
        # Core architecture constraints (emphasize technical relevance over naming rules)
        theme_context = ""
        if core_theme:
            theme_context = f"\n\n🎯 CORE RESEARCH ARCHITECTURE: {core_theme}\n\n"
            theme_context += f"ARCHITECTURAL COHERENCE:\n"
            theme_context += f"- The overall research is building a '{core_theme}'-based system\n"
            theme_context += f"- Your module should be a COMPONENT of this '{core_theme}' framework\n"
            theme_context += f"- Ensure technical COMPATIBILITY with '{core_theme}' principles in the implementation\n\n"
            theme_context += f"CRITICAL NAMING GUIDELINES:\n"
            theme_context += f"- Module name should reflect its PRIMARY FUNCTION, not the framework name\n"
            theme_context += f"- AVOID redundant prefixing (e.g., '{core_theme}-XXX Module')\n"
            theme_context += f"- Use descriptive, technical names (e.g., 'Dynamic Routing', 'Hierarchical Fusion', 'Adaptive Attention')\n"
            theme_context += f"- Connection to '{core_theme}' should be in DESCRIPTION/IMPLEMENTATION, not forced into NAME\n\n"
            theme_context += f"GOOD NAMING EXAMPLES:\n"
            theme_context += f"- ✅ 'Wavelet Synaptic Gating Module' (functional description, works with {core_theme})\n"
            theme_context += f"- ✅ 'Adaptive Token Aggregation' (describes what it does, compatible with {core_theme})\n"
            theme_context += f"- ✅ 'Hierarchical Expert Routing' (clear function, leverages {core_theme} principles)\n\n"
            theme_context += f"BAD NAMING EXAMPLES:\n"
            theme_context += f"- ❌ '{core_theme} Routing Module' (redundant prefix)\n"
            theme_context += f"- ❌ '{core_theme}-based Attention' (forced theme insertion)\n"
            theme_context += f"- ❌ 'Novel {core_theme} Component' (generic, lacks functional clarity)\n"

        # Failure-aware context (avoid known failure modes)
        failure_context = ""
        if failed_innovation:
            failed_name = failed_innovation.get('name', 'Unknown')
            failed_desc = failed_innovation.get('description', '')
            failure_context = f"\n\n⚠️ FAILURE AWARENESS - AVOID THESE APPROACHES:\n"
            failure_context += f"Previously Failed Innovation: {failed_name}\n"
            failure_context += f"Failed Approach Summary: {failed_desc[:300]}{'...' if len(failed_desc) > 300 else ''}\n\n"
            failure_context += f"CRITICAL FAILURE AVOIDANCE:\n"
            failure_context += f"- Do NOT repeat the same technical approach as the failed innovation\n"
            failure_context += f"- Use FUNDAMENTALLY DIFFERENT methods/architectures\n"
            failure_context += f"- Address potential weaknesses that may have caused the failure\n"
            failure_context += f"- Consider alternative paradigms (if failed used CNN, try attention; if failed used transformer, try CNN/RNN)\n"
            failure_context += f"- Focus on orthogonal technical dimensions\n"

        # Improved prompt focused on a single technical innovation rather than a complete method
        prompt = f"""Based on this specific paper and its extracted technical methods, generate ONE focused technical innovation point.

TARGET PAPER INFORMATION:
Paper ID: {paper_id}
Paper Summary: {paper_summary}

EXTRACTED TECHNICAL METHODS:
{paper_methods}

RESEARCH CONTEXT:
Topic: {research_topic}
Dataset Constraints: {dataset_constraints}{theme_context}{previous_context}{failure_context}

PERSONALITY FOCUS: {self.personality}

INNOVATION SCOPE:
- Generate a SINGLE, SPECIFIC technical innovation point that can serve as a paper subsection
- Focus on ONE novel component/module/mechanism that synthesizes insights from the paper
- Examples: "Dense Prompting Module", "Temporal Modeling Branch", "Region-Specific Consistency Regularization", "Gated Linear Matching"
- The innovation should be a cohesive network component that can be described in a dedicated subsection
- Can synthesize and combine multiple insights from the original paper into one unified component

REQUIREMENTS:
- Must be suitable for top-tier computer science conferences (CVPR, ICCV, NeurIPS, ICML, ICLR)
- Focus on a specific architectural component under supervised learning settings
- Can draw from multiple aspects of the original paper's methods to create one unified innovation
- Ensure compatibility with the given dataset constraints
- Provide specific technical details suitable for a paper subsection

🚨 CRITICAL: NETWORK INPUT SHAPE CONSTRAINT 🚨
The network processes 3D medical image patches after nnUNet preprocessing.
**Patch Size (Input Shape): {patch_size if patch_size else 'Not specified'}**

⚠️ YOUR INNOVATION MUST BE COMPATIBLE WITH THIS INPUT SHAPE:
- All proposed layers/modules must work with input tensors of shape (batch, channels, {f'{patch_size[0]}' if patch_size else 'D'}, {f'{patch_size[1]}' if patch_size else 'H'}, {f'{patch_size[2]}' if patch_size else 'W'})
- DO NOT propose fixed-size operations (e.g., fixed kernel sizes, fixed pooling) that assume specific dimensions
- Ensure any spatial operations (attention, pooling, upsampling) are compatible with these dimensions
- Avoid memory-intensive operations that may fail with this patch size
- If using positional encodings, they must be adaptable to this shape
- Be cautious with operations that require specific divisibility (e.g., patch embedding requiring divisibility by patch_size)

Generate ONE cohesive network component that can serve as a paper subsection, reflecting your specialized perspective as {self.personality}.

SYNTHESIS APPROACH:
- Analyze multiple innovations from the original paper
- Identify common themes, complementary mechanisms, or synergistic opportunities
- Combine insights into one unified, novel network component
- Ensure the component is substantial enough to warrant its own subsection in a paper

**CRITICAL: You must respond with ONLY a valid JSON object in the following format:**

```json
{{
    "name": "Descriptive name of the novel network component (e.g., 'Dense Prompting Module')",
    "motivation": "Why this component is needed and what specific problem it addresses",
    "description": "Detailed technical description suitable for a paper subsection, including mathematical formulations",
    "implementation": "Specific implementation details and architectural design of the component"
}}
```
Please handle all backslashes correctly by escaping them, ensuring the output is a valid JSON string.

Do not include any text before or after the JSON object."""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"{self.name} idea-generation attempt {attempt + 1}/{max_retries}")

                # Call the LLM with this agent's config
                response = query_model(
                    prompt=prompt,
                    system_prompt=f"You are {self.role_description()}\n{self.phase_prompt('idea_generation')}\n\nIMPORTANT: Respond ONLY with valid JSON format as specified in the prompt.",
                    model_str=self.model,
                    temp=self.temperature
                )

                # Check whether the response is empty
                if not response or not response.strip():
                    logger.warning(f"Attempt {attempt + 1}: LLM response is empty")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise RuntimeError("Every attempt returned an empty response")

                # Parse the response
                idea = self._parse_idea_response(response)
                if idea:
                    idea['generator_name'] = self.name
                    idea['generator_personality'] = self.personality
                    idea['source_paper_id'] = paper_id
                    logger.debug(f"{self.name} generated idea successfully: {idea.get('name', 'Unknown')}")
                    return idea
                else:
                    logger.warning(f"Attempt {attempt + 1}: parsing failed")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise RuntimeError(f"{self.name} failed to generate a valid innovation idea")

            except Exception as e:
                logger.error(f"{self.name} attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.debug(f"Attempt {attempt + 1} failed; retrying...")
                    continue
                else:
                    logger.error(f"{self.name} all attempts failed")
                    raise RuntimeError(f"{self.name} idea generation failed: {e}")

    def _parse_idea_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response into an innovation idea."""
        try:
            # Debug: print the raw response
            logger.info(f"Raw LLM response length: {len(response) if response else 0}")
            if response:
                logger.info(f"First 100 response chars: {response[:100]}")
            else:
                logger.error("LLM response is empty")
                raise RuntimeError("LLM response is empty")

            # Clean the response text
            cleaned_text = response.strip()

            if not cleaned_text:
                logger.error("Cleaned response text is empty")
                raise RuntimeError("Cleaned response text is empty")

            # Extract the JSON part
            if "```json" in cleaned_text:
                import re
                json_pattern = r'```json\s*(.*?)\s*```'
                json_match = re.search(json_pattern, cleaned_text, re.DOTALL)
                if json_match:
                    cleaned_text = json_match.group(1).strip()
                    logger.info(f"Extracted JSON length: {len(cleaned_text)}")
                else:
                    logger.warning("Found ```json marker but could not extract JSON content")

            # Debug: print the JSON text about to be parsed
            logger.info(f"JSON text to parse: {cleaned_text[:200]}...")

            # Parse the JSON
            idea_data = json_repair.loads(cleaned_text)

            # Validate required fields
            required_fields = ["name", "motivation", "description", "implementation"]
            for field in required_fields:
                if field not in idea_data:
                    raise ValueError(f"Missing required field: {field}")

            return idea_data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Text attempted to parse: {cleaned_text if 'cleaned_text' in locals() else 'N/A'}")
            raise RuntimeError(f"LLM response is not valid JSON: {e}")
        except Exception as e:
            logger.error(f"Failed to parse innovation idea: {e}")
            logger.error(f"Response content: {response[:500] if response else 'None'}...")
            raise RuntimeError(f"Failed to parse innovation idea: {e}")


class AssessmentAgent(BaseAgent):
    """Assessment agent — scores generated ideas across multiple dimensions."""

    def __init__(self, model="deepseek/deepseek-v3.2", evaluation_criteria=None, 
                 temperature=0.2, max_tokens=8192, notes=None, max_steps=5):
        super().__init__(model, notes, max_steps)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.evaluation_criteria = evaluation_criteria or self._default_criteria()
        self.phases = ["assessment"]

    def _default_criteria(self):
        """Default evaluation criteria"""
        return {
            "coherence": {"weight": 0.15, "description": "Logical consistency and internal coherence"},
            "credibility": {"weight": 0.15, "description": "Technical credibility and scientific reasonability"},
            "verifiability": {"weight": 0.15, "description": "Verifiability and experimental feasibility"},
            "novelty": {"weight": 0.25, "description": "Innovation and novelty level"},
            "alignment": {"weight": 0.15, "description": "Alignment with research objectives"},
            "modularity": {"weight": 0.15, "description": "Module design quality: naming clarity, functional independence, architectural integration"}
        }

    def role_description(self):
        return "an expert AI research evaluator specializing in rigorous multi-dimensional assessment of research innovations"

    def phase_prompt(self, phase):
        if phase == "assessment":
            criteria_desc = "\n".join([
                f"- {name.upper()}: {info['description']} (weight: {info['weight']})"
                for name, info in self.evaluation_criteria.items()
            ])
            
            return load_skill("agents/assessment_agent.md", criteria_desc=criteria_desc)
        return "Your goal is to evaluate research innovations."

    def command_descriptions(self, phase):
        if phase == "assessment":
            return (
                "Evaluate multiple research innovations using rigorous multi-dimensional criteria.\n\n"
                "**EVALUATION PROCESS:**\n"
                "1. Analyze each innovation against all evaluation criteria\n"
                "2. Provide detailed justification for each criterion score\n"
                "3. Calculate weighted final scores\n"
                "4. Rank innovations by final score\n\n"
                "**OUTPUT REQUIREMENTS:**\n"
                "You MUST output a valid JSON object with evaluation results.\n"
                "**CRITICAL:** Output ONLY valid JSON. No additional text or explanations."
            )
        return ""

    def evaluate_ideas(self, ideas: List[Dict[str, str]], research_topic: str, 
                      dataset_constraints: str, core_theme: str = None) -> Dict[str, Any]:
        """
        Evaluate multiple innovation ideas.

        Args:
            ideas: list of innovation ideas.
            research_topic: research topic.
            dataset_constraints: dataset constraints.
            core_theme: core research architecture theme (optional).

        Returns:
            Evaluation-result dict with ranked ideas and scores.
        """
        logger.debug(f"Assessment agent evaluating {len(ideas)} innovation ideas")

        # Clear history
        self.history = []

        # Build the evaluation prompt
        ideas_text = self._format_ideas_for_evaluation(ideas)
        criteria_text = self._format_criteria_for_prompt()

        # Build the core-theme description
        core_theme_context = ""
        if core_theme:
            core_theme_context = f"\nCore Architecture Theme: {core_theme}"
        
        prompt = f"""Evaluate the following research innovations using rigorous multi-dimensional criteria.

RESEARCH CONTEXT:
Topic: {research_topic}
Dataset Constraints: {dataset_constraints}{core_theme_context}

EVALUATION CRITERIA:
{criteria_text}

INNOVATIONS TO EVALUATE:
{ideas_text}

EVALUATION INSTRUCTIONS:
1. For each innovation, analyze it against ALL evaluation criteria
2. Assign a score from 0.0 to 10.0 for each criterion
3. Provide detailed justification for each score
4. Calculate the weighted final score using the given weights
5. Rank all innovations by their final scores

SPECIAL ATTENTION - MODULARITY CRITERION:
Evaluate the module design quality based on:
- **Naming Quality**: Is the module name descriptive of its FUNCTION? Does it avoid redundant prefixing?
  * High score: "Adaptive Token Aggregation", "Hierarchical Fusion Layer" (clear function)
  * Low score: "{core_theme or '[CoreTheme]'}-XXX Module" or "Novel Component" (redundant prefix with core theme or generic name)
  * Note: Module names should NOT simply prefix the core architecture theme{' "' + core_theme + '"' if core_theme else ''}
- **Functional Independence**: Does the module have a clear, distinct role?
- **Architecture Integration**: How well does it integrate into the overall framework?
  * The connection to core theme should be in implementation/description, not forced in name

OUTPUT REQUIREMENTS:
**CRITICAL**: The "idea_name" field in your output MUST be the EXACT name shown in the "Name:" field of each INNOVATION above. Copy it exactly as-is, character by character. Do NOT paraphrase, abbreviate, or modify it.

You MUST output a valid JSON object with this exact structure:
```json
{{
  "evaluations": [
    {{
      "idea_name": "EXACT name from the Name field above - copy it precisely",
      "generator_name": "generator_that_created_this_idea",
      "criterion_scores": {{
        "coherence": {{
          "score": 8.5,
          "justification": "Detailed explanation for this score"
        }},
        "credibility": {{
          "score": 7.0,
          "justification": "Detailed explanation for this score"
        }},
        "verifiability": {{
          "score": 9.0,
          "justification": "Detailed explanation for this score"
        }},
        "novelty": {{
          "score": 8.0,
          "justification": "Detailed explanation for this score"
        }},
        "alignment": {{
          "score": 7.5,
          "justification": "Detailed explanation for this score"
        }},
        "modularity": {{
          "score": 8.0,
          "justification": "Detailed explanation for naming quality, functional independence, and architecture integration"
        }}
      }},
      "weighted_final_score": 8.1,
      "overall_assessment": "Comprehensive overall evaluation summary"
    }}
  ],
  "ranking": [
    {{
      "rank": 1,
      "idea_name": "EXACT name from the Name field above - copy it precisely",
      "final_score": 8.1,
      "key_strengths": ["strength1", "strength2"],
      "key_weaknesses": ["weakness1", "weakness2"]
    }}
  ],
  "evaluation_summary": {{
    "total_ideas_evaluated": 5,
    "highest_score": 8.1,
    "lowest_score": 6.2,
    "average_score": 7.3,
    "recommended_idea": "EXACT name from the Name field above"
  }}
}}
```

**CRITICAL:** Output ONLY valid JSON. No additional text, explanations, or markdown formatting."""

        # Inject CamylaNet framework documentation for feasibility-aware evaluation
        try:
            _fw_doc_path = Path(__file__).parent.parent.parent / "skills" / "frameworks" / "camylanet" / "documentation.md"
            if _fw_doc_path.exists():
                _fw_doc = _fw_doc_path.read_text(encoding='utf-8')
                prompt += f"""

FRAMEWORK FEASIBILITY CHECK (affects VERIFIABILITY score):
Each proposal will be implemented in the CamylaNet framework (nnUNet v2 wrapper).
The ONLY customizable part is the network architecture (via build_network_architecture).

What IS feasible (do NOT penalize these):
- Any nn.Module architecture: attention, cross-attention, window attention, gating, multi-scale
- All standard PyTorch operations (Linear, Conv3d, Softmax, matmul, einsum, etc.)
- Complex module designs as long as they take a single input tensor and return a single output tensor

What is NOT feasible (hard constraints — penalize VERIFIABILITY if violated):
- Custom loss functions or loss modifications
- Training loop / optimizer / LR schedule changes
- Multi-input pipelines (model receives only ONE input tensor)
- Models returning tuples/dicts (must return a single tensor)
- torch.linalg.qr / torch.linalg.svd / torch.linalg.eigh (incompatible with float16)
- Self-supervised / contrastive / GAN training paradigms
- Data loading / preprocessing / augmentation changes

{_fw_doc}

When scoring VERIFIABILITY, check ONLY these hard constraints:
1. Does it require modifying loss, training loop, or data pipeline? (If yes → VERIFIABILITY <= 3.0)
2. Does the model need multiple input tensors? (If yes → VERIFIABILITY <= 3.0)
3. Does the model return tuples/dicts instead of a single tensor? (If yes → VERIFIABILITY <= 3.0)
4. Does it use torch.linalg.qr/svd/eigh? (If yes → VERIFIABILITY <= 3.0)
5. Does it require non-standard supervised training? (If yes → VERIFIABILITY <= 3.0)
6. Do the proposal's data assumptions match the actual dataset? (e.g., multi-modal when dataset is single-modality → VERIFIABILITY <= 3.0)

Do NOT penalize complex architectures (attention, multi-scale, gating, etc.) — these are fully supported.
"""
        except Exception as e:
            logger.warning(f"Could not inject framework documentation into assessment: {e}")

        try:
            # Call the LLM for evaluation
            response = query_model(
                prompt=prompt,
                system_prompt=f"You are {self.role_description()}\n{self.phase_prompt('assessment')}",
                model_str=self.model,
                temp=self.temperature
            )

            # Parse the evaluation result
            evaluation_result = self._parse_evaluation_response(response)
            if evaluation_result:
                logger.debug(f"Evaluation complete; recommended idea: {evaluation_result['evaluation_summary']['recommended_idea']}")
                return evaluation_result
            else:
                raise RuntimeError("Assessment agent failed to produce a valid evaluation result")

        except Exception as e:
            logger.error(f"Evaluation process failed: {e}")
            raise RuntimeError(f"Evaluation process failed: {e}")

    def _format_ideas_for_evaluation(self, ideas: List[Dict[str, str]]) -> str:
        """Format innovation ideas for evaluation.

        Supports the new proposal format (title, modules, generator, etc.)
        while staying backward-compatible with the old format (name, description, generator_name, etc.).
        """
        formatted_ideas = []
        for i, idea in enumerate(ideas, 1):
            # Handle both old/new field names; prefer new ones
            idea_name = idea.get('title', idea.get('name', 'Unknown'))
            generator_name = idea.get('generator', idea.get('generator_name', 'Unknown'))
            
            # motivation: dict in the new format, string in the old format
            motivation_raw = idea.get('motivation', 'Not provided')
            if isinstance(motivation_raw, dict):
                motivation_text = (
                    f"Background: {motivation_raw.get('background', 'N/A')}\n"
                    f"Limitations: {motivation_raw.get('limitations', 'N/A')}\n"
                    f"Insight: {motivation_raw.get('insight', 'N/A')}"
                )
            else:
                motivation_text = str(motivation_raw)
            
            # modules: core content in the new format; absent in the old format
            modules = idea.get('modules', [])
            if modules and isinstance(modules, list):
                modules_text = ""
                for j, mod in enumerate(modules, 1):
                    mod_name = mod.get('name', f'Module {j}')
                    mod_desc = mod.get('description', 'N/A')
                    mod_form = mod.get('formulation', '')
                    mod_role = mod.get('role', '')
                    modules_text += f"\n  Module {j}: {mod_name}"
                    modules_text += f"\n    Description: {mod_desc}"
                    if mod_form:
                        modules_text += f"\n    Formulation: {mod_form}"
                    if mod_role:
                        modules_text += f"\n    Role: {mod_role}"
            else:
                # Legacy-format fallback
                modules_text = idea.get('description', 'Not provided')
            
            # integration and contributions (new-format fields)
            integration = idea.get('integration', '')
            contributions = idea.get('contributions', [])
            
            idea_text = f"""
INNOVATION {i}:
Name: {idea_name}
Generator: {generator_name}
Core Theme: {idea.get('core_theme', 'N/A')}

Motivation:
{motivation_text}

Proposed Method:{modules_text}
"""
            if integration:
                idea_text += f"\nIntegration and Data Flow: {integration}\n"
            
            if contributions:
                contribs_str = "; ".join(contributions) if isinstance(contributions, list) else str(contributions)
                idea_text += f"\nExpected Contributions: {contribs_str}\n"
            
            formatted_ideas.append(idea_text)
        
        return "\n" + "="*80 + "\n".join(formatted_ideas) + "\n" + "="*80

    def _format_criteria_for_prompt(self) -> str:
        """Format evaluation criteria for the prompt."""
        criteria_lines = []
        for name, info in self.evaluation_criteria.items():
            criteria_lines.append(f"- {name.upper()} (weight {info['weight']}): {info['description']}")
        return "\n".join(criteria_lines)

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse the evaluation response."""
        try:
            # Clean the response text
            cleaned_text = response.strip()

            # Extract the JSON portion
            if "```json" in cleaned_text:
                import re
                json_pattern = r'```json\s*(.*?)\s*```'
                json_match = re.search(json_pattern, cleaned_text, re.DOTALL)
                if json_match:
                    cleaned_text = json_match.group(1).strip()

            # Parse JSON
            evaluation_data = json_repair.loads(cleaned_text)

            # Validate required fields
            required_fields = ["evaluations", "ranking", "evaluation_summary"]
            for field in required_fields:
                if field not in evaluation_data:
                    raise ValueError(f"Missing required field: {field}")

            return evaluation_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation result JSON: {e}")
            raise RuntimeError(f"Evaluation result is not valid JSON: {e}")
        except Exception as e:
            logger.error(f"Failed to parse evaluation result: {e}")
            raise RuntimeError(f"Failed to parse evaluation result: {e}")


class PostdocAgent(BaseAgent):
    """Postdoctoral agent."""

    def __init__(self, model="deepseek/deepseek-v3.2", notes=None, max_steps=100):
        super().__init__(model, notes, max_steps)
        self.phases = ["plan formulation", "innovation_brainstorming", "results interpretation"]

    def role_description(self):
        return load_skill("agents/postdoc/role_description.md")

    def phase_prompt(self, phase):
        if phase == "plan formulation":
            return load_skill("agents/postdoc/plan_formulation.md")
        elif phase == "results interpretation":
            return load_skill("agents/postdoc/results_interpretation.md")
        else:
            return "Your goal is to assist in the research process."

    def command_descriptions(self, phase):
        if phase == "plan formulation":
            return (
                "You can produce dialogue using: ```DIALOGUE\ndialogue here\n```\n"
                "When you have formulated THREE distinct research proposals, submit them using the new structured format:\n\n"
                "```json\n"
                "[\n"
                "  {\n"
                "    \"proposal_name\": \"Descriptive name of the proposed method\",\n"
                "    \"problem_statement_gap_analysis\": \"Detailed analysis of the problem and research gap this proposal addresses\",\n"
                "    \"proposed_architecture_novel_modules\": {\n"
                "      \"core_structure\": \"Description of the main architectural framework\",\n"
                "      \"novel_modules\": [\n"
                "        {\n"
                "          \"name\": \"Module name\",\n"
                "          \"design\": \"Detailed technical design and implementation\",\n"
                "          \"placement\": \"Where this module fits in the architecture\"\n"
                "        }\n"
                "      ],\n"
                "      \"interplay_of_modules\": \"How the modules work together\",\n"
                "      \"expected_benefits\": [\"Benefit 1\", \"Benefit 2\", \"Benefit 3\"],\n"
                "      \"alignment_with_constraints\": [\"Constraint 1\", \"Constraint 2\"]\n"
                "    }\n"
                "  },\n"
                "  {...},\n"
                "  {...}\n"
                "]\n"
                "```\n\n"
                "Ensure exactly 3 proposals with detailed technical specifications."
            )
        else:
            return ""

    def context(self, phase):
        if phase == "plan formulation":
            return f"Current Literature Review: {self.lit_review_sum}"
        else:
            return ""


class ChallengeDiscoveryAgent(BaseAgent):
    """
    Challenge-discovery agent — extracts network-architecture challenges from paper Introduction/Related Work sections.

    Design principles:
    - Extract real research challenges from papers rather than relying on LLM hallucinations.
    - Filtering rules mirror PhDStudentAgent (the same skill prompt is reused).
    - Extract at most 2 challenges, focused on supervised-learning network-architecture issues.
    """
    
    def __init__(self, model="deepseek/deepseek-v3.2", max_tokens=4096):
        super().__init__(model, notes=None, max_steps=5)
        self.phases = ["challenge_extraction"]
        self.max_tokens = max_tokens
    
    def role_description(self):
        return "a research challenge analyst who extracts network architecture challenges from academic papers"
    
    def phase_prompt(self, phase):
        return "Extract specific network architecture challenges from the paper's Introduction/Related Work section."
    
    def command_descriptions(self, phase):
        return ""
    
    def extract_challenges(self, paper_text: str, task_context: Dict) -> List[Dict]:
        """
        Extract up to 2 network-architecture challenges from a paper.

        Args:
            paper_text: full paper text (focus on Introduction/Related Work).
            task_context: task context with task_type, modality, etc.

        Returns:
            List[Dict]: challenge list; each item has name, description, source.
        """
        logger.debug(f"ChallengeDiscoveryAgent: starting challenge extraction")
        
        # Load the skill prompt, reusing the STRICTLY AVOID rules
        prompt = load_skill(
            "agents/challenge_discovery.md",
            task_type=task_context.get('task_type', 'segmentation'),
            modality=task_context.get('modality', ''),
            paper_content=paper_text[:50000]  # cap length
        )
        
        try:
            response = query_model(
                prompt=prompt,
                system_prompt="You are a research challenge analyst. Output valid JSON only.",
                model_str=self.model,
                temp=0.3
            )
            
            # Parse the JSON response
            response_text = self.clean_text(response)
            
            # Try to extract JSON
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                challenges = json_repair.loads(json_match.group())
                # Cap at 2 challenges
                challenges = challenges[:2] if len(challenges) > 2 else challenges
                logger.debug(f"Extracted {len(challenges)} challenges: {[c.get('name', 'Unknown') for c in challenges]}")
                return challenges
            else:
                logger.warning("Could not extract valid JSON from the response")
                return []
                
        except Exception as e:
            logger.error(f"Challenge extraction failed: {e}")
            return []


class InnovationGenerator:
    """Literature-search-driven innovation generator."""
    
    def __init__(self, model_backbone="deepseek/deepseek-v3.2", openai_api_key=None, verbose=True, config=None):
        self.model_backbone = model_backbone
        self.openai_api_key = openai_api_key
        self.verbose = verbose
        self.config = config

        # Resolve specialized models from llm_roles, fallback to model_backbone
        from camyla.model_config import get_role
        paper_extraction_model = get_role("paper_extraction").get("model") or model_backbone
        citation_analysis_model = get_role("citation_analysis").get("model") or model_backbone
        self.challenge_extraction_model = get_role("challenge_extraction").get("model") or model_backbone
        logger.info(f"Model config: backbone={model_backbone}, paper_extraction={paper_extraction_model}, "
                    f"citation_analysis={citation_analysis_model}, challenge_extraction={self.challenge_extraction_model}")

        # Initialize the agent
        self.phd = PhDStudentAgent(
            model=model_backbone,
            max_steps=20
        )
        self.postdoc = PostdocAgent(
            model=model_backbone,
            max_steps=20
        )

        # Initialize the paper-methods summarization agent
        self.paper_summary_agent = PaperSummaryAgent(
            model=paper_extraction_model,
            max_steps=5
        )

        # Initialize the citation-network agent
        self.citation_network_agent = CitationNetworkAgent(
            model=citation_analysis_model,
            max_steps=5
        )

        # Initialize the search engine (reads from idea_generation.literature_search in the new schema)
        lit_config = None
        if config and 'idea_generation' in config and 'literature_search' in config['idea_generation']:
            lit_config = config['idea_generation']['literature_search']

        if lit_config is not None:
            # New min_year schema: { phase1, phase2 }; phase2 = main-search year, phase1 = challenge-discovery year
            # Note: config may be an OmegaConf DictConfig instead of a native dict — cannot use isinstance(_, dict)
            min_year_cfg = lit_config.get('min_year', None)
            if min_year_cfg is not None and hasattr(min_year_cfg, 'get'):
                min_year = min_year_cfg.get('phase2', '2021-01-01')
                phase1_min_year = min_year_cfg.get('phase1', '2015-01-01')
            elif isinstance(min_year_cfg, str) and min_year_cfg:
                # Backward compat: flat string
                min_year = min_year_cfg
                phase1_min_year = '2015-01-01'
            else:
                min_year = '2021-01-01'
                phase1_min_year = '2015-01-01'

            enable_randomization = lit_config.get('enable_randomization', True)
            filter_open_access = lit_config.get('filter_open_access', True)

            # sources list: user explicitly specifies which sources to use
            sources = list(lit_config.get('sources', ['semantic_scholar']) or ['semantic_scholar'])
            use_arxiv = 'arxiv' in sources
            use_openalex = 'openalex' in sources
            use_pubmed = 'pubmed' in sources
            use_semantic_scholar = 'semantic_scholar' in sources
        else:
            # Default: use only semantic_scholar
            min_year = '2021-01-01'
            phase1_min_year = '2015-01-01'
            enable_randomization = True
            use_arxiv = False
            use_openalex = False
            use_pubmed = False
            use_semantic_scholar = True
            filter_open_access = True

        # Track searched papers to prevent duplicate searches
        self.searched_paper_ids = set()
        
        # Initialize multi-source literature search
        self.arxiv_search = MultiSourceLiteratureSearch(
            min_year=min_year,
            phase1_min_year=phase1_min_year,
            enable_randomization=enable_randomization,
            use_arxiv=use_arxiv,
            use_openalex=use_openalex,
            use_pubmed=use_pubmed,
            use_semantic_scholar=use_semantic_scholar,
            filter_open_access=filter_open_access
        )

        # Read paper counts and iteration parameters from idea_generation.literature_search (new schema)
        if lit_config is not None:
            self.max_papers_per_search = lit_config.get('max_papers_per_search', 20)
            # target_papers: { phase1, phase2 }
            tp = lit_config.get('target_papers', {}) or {}
            self.target_papers_in_review = tp.get('phase2', 6)
            self.target_phase1_papers = tp.get('phase1', 6)
            self.max_literature_iterations = lit_config.get('max_iterations', 20)
            # Phase 1 challenge-discovery iteration settings moved into challenge_discovery
            cd = lit_config.get('challenge_discovery', {}) or {}
            self.phase1_search_iterations = cd.get('iterations', 6)
            self.phase1_challenges_per_round = cd.get('challenges_per_round', 3)
            self.phase1_final_challenges = cd.get('final_challenges', 3)
        else:
            self.max_papers_per_search = 20
            self.target_papers_in_review = 6
            self.target_phase1_papers = 6
            self.max_literature_iterations = 20
            self.phase1_search_iterations = 6
            self.phase1_challenges_per_round = 3
            self.phase1_final_challenges = 3
        
        # Theme-tracking mechanism: progressive theme deepening
        self.core_theme = None  # core theme (generated independently, not extracted from papers)
        self.search_theme_history = []  # search-theme evolution history
        self.theme_established = False  # whether the core theme has been established
        self.keyword_pool = []  # keyword pool (5 keywords extracted from citation analysis)
        
        # ===== Independent per-theme literature search support =====
        self.theme_keyword_pools = {}    # {theme_idx: [keywords], ...} independent keyword pool per theme
        self.theme_lit_reviews = {}      # {theme_idx: [paper_entries], ...} papers collected per theme
        self.theme_paper_methods = {}    # {theme_idx: paper_methods_str, ...} technical-methods corpus per theme
        self.current_theme_idx = None    # index of the theme currently being processed
        

        # Challenge-discovery mechanism: model is resolved via llm_roles.challenge_extraction in the new schema
        # (model is no longer configured directly inside literature_search). The variable is retained for downstream use.
        challenge_discovery_model = self.challenge_extraction_model
        
        self.challenge_discovery_agent = ChallengeDiscoveryAgent(
            model=challenge_discovery_model
        )
        self.discovered_challenges = []  # stores the extracted challenges
        self.challenge_papers = []  # challenge-discovery papers kept separate from technical papers
        self.cached_idea_data = {}  # cached idea_data used for challenge extraction

        # Initialize the new multi-agent innovation generation system
        self.idea_generators = []
        self.assessment_agent = None
        self._initialize_idea_generation_agents(config)

    def _initialize_idea_generation_agents(self, config):
        """Initialize the innovation-generation and assessment agents (new schema: top-level idea_generation)."""
        try:
            idea_gen_config = None
            if config and 'idea_generation' in config:
                idea_gen_config = config['idea_generation']

            if idea_gen_config:
                # Initialize multiple Idea Generator agents
                for generator_config in idea_gen_config.get('idea_generators', []) or []:
                    generator = IdeaGeneratorAgent(
                        name=generator_config.get('name', 'idea_generator'),
                        model=generator_config.get('model', self.model_backbone),
                        personality=generator_config.get('personality', 'balanced'),
                        temperature=generator_config.get('temperature', 0.7),
                        max_tokens=generator_config.get('max_tokens', 4096)
                    )
                    self.idea_generators.append(generator)
                    logger.debug(f"Initialized {generator.name} ({generator.personality})")

                # Initialize the Assessment agent (new fields: criteria / scoring)
                assessment_config = idea_gen_config.get('assessment')
                if assessment_config:
                    evaluation_criteria = {}
                    for criterion, info in (assessment_config.get('criteria', {}) or {}).items():
                        evaluation_criteria[criterion] = {
                            'weight': info.get('weight', 0.2),
                            'description': info.get('description', f'{criterion} evaluation')
                        }
                    self.assessment_agent = AssessmentAgent(
                        model=assessment_config.get('model', self.model_backbone),
                        evaluation_criteria=evaluation_criteria,
                        temperature=assessment_config.get('temperature', 0.2),
                        max_tokens=assessment_config.get('max_tokens', 8192)
                    )
                    logger.debug("Assessment agent initialized")

            if not self.idea_generators:
                logger.debug("No idea generator configuration found; using defaults")
                self._initialize_default_agents()
                 
        except Exception as e:
            logger.error(f"Failed to initialize the innovation-generation agents: {e}")
            self._initialize_default_agents()

    def _initialize_default_agents(self):
        """Initialize default agent configuration."""
        default_generators = [
            {'name': 'creative_generator', 'personality': 'highly creative and novel approach focused', 'temperature': 0.8},
            {'name': 'technical_generator', 'personality': 'technically rigorous and implementation focused', 'temperature': 0.3},
            {'name': 'medical_generator', 'personality': 'medical imaging and healthcare AI focused', 'temperature': 0.6}
        ]
        
        for gen_config in default_generators:
            generator = IdeaGeneratorAgent(
                name=gen_config['name'],
                model=self.model_backbone,
                personality=gen_config['personality'],
                temperature=gen_config['temperature']
            )
            self.idea_generators.append(generator)
            logger.debug(f"Initialized default {generator.name} ({generator.personality})")
        
        self.assessment_agent = AssessmentAgent(model=self.model_backbone)
        logger.debug(f"Default assessment agent initialized")

    def _load_framework_documentation(self) -> str:
        """Load CamylaNet framework documentation for feasibility-aware proposal generation."""
        try:
            doc_path = Path(__file__).parent.parent.parent / "skills" / "frameworks" / "camylanet" / "documentation.md"
            if doc_path.exists():
                content = doc_path.read_text(encoding='utf-8')
                logger.info(f"Loaded CamylaNet framework documentation ({len(content)} chars)")
                return content
            else:
                logger.warning(f"Framework documentation not found at: {doc_path}")
                return ""
        except Exception as e:
            logger.warning(f"Failed to load framework documentation: {e}")
            return ""

    # ========================================================================
    # New feature: diverse Research Proposal generation
    # ========================================================================
    
    def generate_diverse_proposals(
        self, 
        idea_data: Dict[str, Any], 
        num_proposals: int = 5,
        output_dir: Path = None,
    ) -> List[Dict[str, str]]:
        """
        Generate Research Proposals using an iterative Best-of-N tournament.

        Flow:
        Repeat N times (N = num_proposals):
          1. Each generator produces one candidate proposal.
          2. AssessmentAgent evaluates all candidates.
          3. Pick the best one (Top 1).
          4. Add the Top 1 to the final list and use it as context for the next round (to avoid duplicates).
        
        Args:
            idea_data: idea.json data.
            num_proposals: desired proposal count.
            output_dir: directory where .md files are saved.
            
        Returns:
            List of proposal metadata.
        """
        candidates_per_round = len(self.idea_generators)
        logger.info(f"Generating {num_proposals} diverse Research Proposals (candidates per round: {candidates_per_round}, generators: {candidates_per_round})")
        
        # 1. Extract research context
        logger.info("Step 1/4: extract research context")
        self.cached_idea_data = idea_data
        research_context = self._extract_research_context(idea_data)
        logger.debug(f"Research topic: {research_context.get('research_topic', 'Unknown')[:50]}...")
        
        # 2. Run literature search and Phase 1 challenge discovery
        logger.info("Step 2/4: run literature search and challenge discovery")
        literature_review = self._perform_literature_search(research_context)
        
        # Ensure at least one challenge-theme pair exists
        if not hasattr(self, 'challenge_themes') or not self.challenge_themes:
            logger.warning("No challenge-theme pair; falling back to default theme generation")
            self.challenge_themes = [{
                'challenge': {'name': 'general improvement', 'description': 'general architectural improvements'},
                'core_theme': self.core_theme or 'architectural innovation'
            }]
        
        logger.debug(f"{len(self.challenge_themes)} challenge-theme pairs available")
        
        # 3. Iteratively generate Proposals
        logger.info("Step 3/4: iteratively generate Research Proposals")
        
        logger.debug(f"Configured generator count: {len(self.idea_generators)}")
        for i, gen in enumerate(self.idea_generators):
            logger.debug(f"  [{i}] {gen.name}: model={gen.model}, temp={gen.temperature}")
        
        final_proposals = []
        
        # Set output directory — absolute path to avoid CWD issues
        if output_dir is None:
            raise ValueError("output_dir must be explicitly provided to avoid path issues")
        output_dir = Path(output_dir).resolve()  # force to absolute path
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory: {output_dir}")
        
        # Replaced the global fetch with a per-theme fetch inside the loop
        # paper_methods_combined = self._get_combined_paper_methods()
        
        for round_idx in range(num_proposals):
            logger.debug(f"ROUND {round_idx + 1}/{num_proposals}")
            
            # Step 1: generate candidates
            candidates = []
            logger.debug(f"Generating {candidates_per_round} candidates...")
            
            # 🔄 Change: all candidates for one Proposal use the same theme; rotate themes between Proposals
            # (previous logic rotated themes per candidate; now we rotate at the Proposal level)
            theme_idx = round_idx % len(self.challenge_themes)
            current_theme_info = self.challenge_themes[theme_idx]
            
            logger.debug(f"Round theme: theme_{theme_idx} - {current_theme_info['core_theme']}, "
                        f"challenge: {current_theme_info['challenge'].get('name', 'Unknown')}")
            
            # ===== New: fetch paper_methods specific to this theme =====
            theme_paper_methods = self.theme_paper_methods.get(theme_idx, "")
            if not theme_paper_methods:
                # Fallback: use the globally merged methods
                theme_paper_methods = self._get_combined_paper_methods()
                logger.debug(f"Using global paper_methods (theme {theme_idx} has no dedicated data)")
            else:
                logger.debug(f"Using Theme {theme_idx} dedicated paper_methods ({len(theme_paper_methods)} characters)")
            
            for i in range(candidates_per_round):
                # All candidates share the same theme, but use different generators
                # Determine the generator
                generator_idx = i % len(self.idea_generators)
                current_generator = self.idea_generators[generator_idx]
                
                logger.debug(f"Candidate {i+1}: theme={current_theme_info['core_theme'][:30]}..., generator={current_generator.name}")
                
                try:
                    # 🔥 Critical fix: also pass in candidates generated this round to avoid intra-round duplicates
                    all_existing_proposals = final_proposals + candidates  # cross-round + already generated in this round
                    
                    proposal_data = self._generate_single_proposal(
                        research_context=research_context,
                        challenge_info=current_theme_info['challenge'],
                        core_theme=current_theme_info['core_theme'],
                        paper_methods=theme_paper_methods,  # use the theme-specific paper_methods

                        generator=current_generator,
                        proposal_id=f"candidate_r{round_idx}_{i}",
                        existing_proposals=all_existing_proposals  # includes candidates already generated this round
                    )
                    
                    if proposal_data:
                        # Simple duplication check (double check)
                        # if not self._check_proposal_duplicate(proposal_data, final_proposals):
                        candidates.append(proposal_data)
                        logger.debug(f"Candidate {i+1}: {proposal_data.get('title', 'Untitled')[:30]}...")
                        # else:
                        #     print(f"      ⚠️ Candidate {i+1} duplicate, skipped")
                except Exception as e:
                    logger.error(f"Candidate generation failed: {e}")
                    
            if not candidates:
                logger.warning(f"Round {round_idx+1} produced no valid candidates; skipping")
                continue
                
            # Step 2: evaluate
            logger.debug(f"Evaluating {len(candidates)} candidates...")
            try:
                if self.assessment_agent:
                    evaluation_result = self.assessment_agent.evaluate_ideas(
                        candidates,
                        research_topic=research_context.get('research_topic', ''),
                        dataset_constraints=research_context.get('dataset_constraints', '')
                    )
                    
                    # Step 3: pick the best
                    # Pick the first entry from the ranking list
                    ranking = evaluation_result.get('ranking', [])
                    if ranking:
                        best_idea_name = ranking[0].get('idea_name', '')
                        
                        # 🔥 Fix: use multi-strategy matching to support both old and new field names
                        # Strategy 1: exact-match title
                        best_proposal = next((p for p in candidates if p.get('title') == best_idea_name), None)
                        
                        # Strategy 2: exact-match legacy name field
                        if best_proposal is None:
                            best_proposal = next((p for p in candidates if p.get('name') == best_idea_name), None)
                        
                        # Strategy 3: case-insensitive match
                        if best_proposal is None:
                            best_idea_lower = best_idea_name.lower().strip()
                            best_proposal = next(
                                (p for p in candidates 
                                 if p.get('title', '').lower().strip() == best_idea_lower
                                 or p.get('name', '').lower().strip() == best_idea_lower),
                                None
                            )
                        
                        # Strategy 4: substring-containment match (the LLM may truncate or slightly tweak names)
                        if best_proposal is None and best_idea_name and best_idea_name != 'Unknown':
                            best_idea_lower = best_idea_name.lower().strip()
                            best_proposal = next(
                                (p for p in candidates
                                 if best_idea_lower in p.get('title', '').lower()
                                 or p.get('title', '').lower() in best_idea_lower),
                                None
                            )
                        
                        # Strategy 5: rank-order match based on weighted_final_score.
                        # When every name-based match fails, use the evaluation score order instead.
                        if best_proposal is None:
                            evaluations = evaluation_result.get('evaluations', [])
                            if evaluations and len(evaluations) == len(candidates):
                                # Evaluation order generally matches input order (LLM preserves it)
                                scored_candidates = []
                                for idx, ev in enumerate(evaluations):
                                    score = ev.get('weighted_final_score', 0)
                                    if idx < len(candidates):
                                        scored_candidates.append((score, idx))
                                if scored_candidates:
                                    scored_candidates.sort(key=lambda x: x[0], reverse=True)
                                    best_idx = scored_candidates[0][1]
                                    best_proposal = candidates[best_idx]
                                    logger.info(f"      📊 Name match failed; selecting by evaluation score rank: idx={best_idx}, score={scored_candidates[0][0]}")
                        
                        # Final fallback
                        if best_proposal is None:
                            best_proposal = candidates[0]
                            logger.warning(f"All match strategies failed (idea_name='{best_idea_name}'); falling back to candidates[0]")
                        
                        matched_generator = best_proposal.get('generator', best_proposal.get('generator_name', 'Unknown'))
                        logger.info(f"Round {round_idx+1} Winner: {best_proposal.get('title', best_proposal.get('name', 'Unknown'))} "
                                    f"(Score: {ranking[0].get('final_score')}, Generator: {matched_generator})")
                    else:
                        best_proposal = candidates[0]
                        logger.debug(f"No ranking returned, picking first candidate")
                else:
                    best_proposal = candidates[0]
                    logger.debug(f"No AssessmentAgent, picking first candidate")
            
            except Exception as e:
                logger.error(f"Assessment failed: {e}")
                best_proposal = candidates[0] # Fallback
            
            # Step 4: save and append to the final list
            md_file_path = self._save_proposal_as_markdown(
                best_proposal, 
                output_dir, 
                len(final_proposals) + 1
            )
            # Save the absolute path
            best_proposal['md_file_path'] = str(md_file_path.resolve())
            
            logger.debug(f"Winner saved: generator={best_proposal.get('generator')}, md_file={best_proposal['md_file_path']}")
            
            final_proposals.append(best_proposal)
            logger.info(f"Round {round_idx+1} Winner added: {best_proposal.get('title')}")
        
        # 4. Summarize results
        logger.info(f"Step 4/4: successfully generated {len(final_proposals)}/{num_proposals} Research Proposals")
        
        for i, p in enumerate(final_proposals, 1):
            logger.debug(f"  {i}. {p.get('title', 'Untitled')} -> {p.get('md_file_path', 'N/A')}")
        
        if not final_proposals:
            raise RuntimeError("Failed to generate any valid Research Proposal")
        
        return final_proposals
    
    def _generate_single_proposal(
        self,
        research_context: Dict[str, str],
        challenge_info: Dict[str, str],
        core_theme: str,
        paper_methods: str,
        generator: 'IdeaGeneratorAgent',
        proposal_id: str,
        existing_proposals: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete Research Proposal using the specified agent.
        
        Returns:
            Proposal data dict containing title, motivation, modules, integration, contributions.
        """
        # Build Negative Constraints (including candidates from the current round)
        negative_constraints = "AVOID outdated methods: Simple Multi-scale Fusion, Standard Self-Attention, DeepLabV3+, U-Net (unless significantly modified).\n"
        
        if existing_proposals:
            negative_constraints += "\n⛔ CRITICAL: YOU MUST AVOID REPEATING THE FOLLOWING PROPOSALS:\n"
            for i, p in enumerate(existing_proposals):
                negative_constraints += f"""   {i+1}. {p.get('title')}
      - Core Theme: {p.get('core_theme', 'N/A')}
      - Key Insight: {p.get('motivation', {}).get('insight', 'N/A')}
      - Main Modules: {', '.join([m.get('name', '') for m in p.get('modules', [])])}
      - Generator: {p.get('generator', 'N/A')}
"""
            negative_constraints += "\n⚠️ Your proposal MUST be functionally DIFFERENT from these. If you generate a similar idea, it will be rejected.\n"
            negative_constraints += "💡 DIFFERENTIATION STRATEGY: Focus on a different mathematical formulation, a different architectural paradigm (e.g. if previous used CNN, use State Space Model), or a different stage of the pipeline.\n"
            negative_constraints += f"💡 YOUR GENERATOR IDENTITY: {generator.name} ({generator.personality})\n"
            negative_constraints += "   - Use your unique perspective to propose something different from the above proposals.\n"
            negative_constraints += "   - Even if addressing the same core theme, approach it from YOUR distinctive angle."

        # Load the skill prompt
        prompt = load_skill(
            "agents/research_proposal_generation.md",
            task_type=research_context.get('task_type', 'segmentation'),
            modality=research_context.get('modality', ''),
            target_structure=research_context.get('target_structure', ''),
            core_theme=core_theme,
            challenge_name=challenge_info.get('name', 'Unknown'),
            challenge_description=challenge_info.get('description', ''),
            paper_methods=paper_methods[:50000],  # cap length
            dataset_constraints=research_context.get('dataset_constraints', ''),
            negative_constraints=negative_constraints,
            patch_size=research_context.get('patch_size', 'Not specified')
        )
        
        # Inject CamylaNet framework documentation for feasibility-aware generation
        framework_doc = self._load_framework_documentation()
        if framework_doc:
            prompt += f"""

## Implementation Framework (proposals violating hard constraints will be rejected)

Your proposal will be implemented using the **CamylaNet framework** (nnUNet v2 wrapper).
You can **ONLY customize the network architecture** (via a custom Trainer's `build_network_architecture` method).

What you CAN freely propose (all of these are supported and encouraged):
- Custom nn.Module network architectures of any complexity
- Attention mechanisms (self-attention, cross-attention, window attention, multi-head attention)
- Convolutions of any kind (standard, dilated, depthwise separable, grouped, deformable)
- Gating mechanisms, feature fusion, multi-scale architectures
- Any standard PyTorch operations (Linear, Softmax, matmul, einsum, etc.)
- Novel module designs that process a single input tensor and return a single output tensor
- Architecture hyperparameters (channel sizes, layer depths, kernel sizes, number of heads, etc.)

Hard constraints — you CANNOT propose these (they are fixed by the framework):
- Custom loss functions or loss modifications
- Changes to the training loop, optimizer type, or learning rate schedule
- Multi-input pipelines (model receives only ONE input tensor)
- Models returning tuples or dicts (must return a single tensor)
- torch.linalg.qr / torch.linalg.svd / torch.linalg.eigh (incompatible with float16)
- Self-supervised, contrastive, GAN-based, or non-standard supervised training
- Changes to data loading, preprocessing, or augmentation

{framework_doc}

Ensure ALL proposed modules are implementable as a single nn.Module within this framework.
"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Build a system prompt that emphasizes personality
                personality_guidance = {
                    "highly creative and novel approach focused": "Focus on BOLD, UNCONVENTIONAL ideas. Prioritize NOVELTY over conservatism. Think outside the box.",
                    "technically rigorous and implementation focused": "Focus on PRACTICAL, IMPLEMENTABLE solutions. Prioritize FEASIBILITY and technical soundness. Be pragmatic.",
                    "balanced creativity and technical feasibility": "Balance INNOVATION with PRACTICALITY. Find the sweet spot between novelty and implementability.",
                    "deep learning architecture innovation specialist": "Focus on ARCHITECTURAL INNOVATIONS. Think about novel network structures, connection patterns, and module designs.",
                    "medical imaging and healthcare AI focused": "Focus on MEDICAL DOMAIN needs. Consider clinical applicability, interpretability, and medical imaging characteristics."
                }.get(generator.personality, "Generate diverse and innovative proposals.")
                
                system_prompt = f"""You are {generator.personality} ({generator.name}).

{personality_guidance}

YOUR MISSION: Generate a UNIQUE research proposal that reflects YOUR perspective and expertise.
- If other proposals exist, ensure yours is DISTINCTLY DIFFERENT in approach, formulation, or architecture.
- Leverage YOUR unique strengths to create something others wouldn't think of.

Generate high-quality research proposals suitable for top-tier conferences.
YOU MUST STRICTLY FOLLOW NEGATIVE CONSTRAINTS.
Output ONLY valid JSON."""
                
                response = query_model(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model_str=generator.model,
                    temp=generator.temperature
                )
                
                if not response or not response.strip():
                    continue
                
                # Parse JSON
                response_text = BaseAgent.clean_text(response)
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    proposal = json_repair.loads(json_match.group())
                    
                    # Validate required fields
                    if 'title' in proposal and 'modules' in proposal:
                        proposal['source_challenge'] = challenge_info.get('name', 'Unknown')
                        proposal['core_theme'] = core_theme
                        proposal['generator'] = generator.name
                        proposal['proposal_id'] = proposal_id
                        return proposal
                
            except Exception as e:
                logger.error(f"Proposal generation attempt {attempt+1} failed: {e}")
                continue
        
        return None
    

    
    def _save_proposal_as_markdown(
        self,
        proposal: Dict[str, Any],
        output_dir: Path,
        proposal_number: int
    ) -> Path:
        """
        Save the proposal as a Markdown file.
        
        Returns:
            Saved file path.
        """
        from pathlib import Path
        
        title = proposal.get('title', 'Untitled Proposal')
        # Clean up the filename
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip().replace(' ', '_')
        filename = f"proposal_{proposal_number:02d}_{safe_title}.md"
        filepath = Path(output_dir) / filename
        
        # Build the Markdown content
        motivation = proposal.get('motivation', {})
        modules = proposal.get('modules', [])
        
        md_content = f"""# {title}

**Core Theme**: {proposal.get('core_theme', 'N/A')}  
**Source Challenge**: {proposal.get('source_challenge', 'N/A')}  
**Generated by**: {proposal.get('generator', 'N/A')}

---

## 1. Motivation

### 1.1 Background and Problem
{motivation.get('background', 'N/A')}

### 1.2 Existing Limitations
{motivation.get('limitations', 'N/A')}

### 1.3 Our Insight
{motivation.get('insight', 'N/A')}

---

## 2. Proposed Method

"""
        
        for i, module in enumerate(modules, 1):
            md_content += f"""### 2.{i} {module.get('name', 'Module')}

**Technical Description**:  
{module.get('description', 'N/A')}

**Mathematical Formulation**:  
{module.get('formulation', 'N/A')}

**Role in Architecture**:  
{module.get('role', 'N/A')}

"""
        
        md_content += f"""---

## 3. Integration and Data Flow

{proposal.get('integration', 'N/A')}

---

## 4. Expected Contributions

"""
        
        for i, contrib in enumerate(proposal.get('contributions', []), 1):
            md_content += f"{i}. {contrib}\n"
        
        # Save the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Proposal saved to: {filepath}")
        return filepath
    
    def _get_combined_paper_methods(self) -> str:
        """Return a technical-methods summary covering all collected papers."""
        if not hasattr(self, 'paper_summary_agent') or not self.paper_summary_agent.extracted_methods:
            return "No technical methods extracted."
        
        combined = "# Technical Methods from Literature\n\n"
        for i, method in enumerate(self.paper_summary_agent.extracted_methods, 1):  # cap quantity
            combined += f"## Paper {i}: {method.get('paper_title', 'Unknown')}\n"
            if method.get('json_data') and method['json_data'].get('innovations'):
                for j, innov in enumerate(method['json_data']['innovations'], 1):
                    combined += f"  - {innov.get('name', 'Unknown')}: {innov.get('description', '')}"
            combined += "\n"
        
        return combined

    def refine_proposal_from_md_content(
        self,
        md_content: str,
        proposal_title: str,
        failure_info: Dict[str, str]
    ) -> Optional[str]:
        """
        Refine a Proposal based on failure info (uses MD file content directly, without parsing).

        Args:
            md_content: full MD file content.
            proposal_title: proposal title (for logging).
            failure_info: failure info containing:
                - error_type: error type (e.g. "runtime_error", "performance_issue", "timeout").
                - error_message: error message.
                - execution_time: execution time.
                - performance_issues: description of performance issues.

        Returns:
            Refined Markdown-formatted proposal, or None on failure.
        """
        logger.info(f"🔧 Refining failed Proposal: {proposal_title}")
        
        # Build the prompt — use the MD content directly
        prompt = f"""Refine the following Research Proposal based on failure feedback.

## Original Proposal (Full Content)

{md_content}

---

## Failure Information
- **Error Type**: {failure_info.get('error_type', 'Unknown')}
- **Error Message**: {failure_info.get('error_message', 'No details')}
- **Execution Time**: {failure_info.get('execution_time', 'N/A')}
- **Performance Issues**: {failure_info.get('performance_issues', 'None specified')}

---

## Your Task

Based on the failure information above, REFINE this proposal by:
1. **REMOVING** modules that are clearly infeasible or causing errors
2. **MODIFYING** modules that have performance issues or are too complex
3. **DO NOT ADD** any new modules

### Rules
- The refined proposal should have **FEWER OR EQUAL** modules than the original
- Focus on making the remaining modules more robust and implementable
- Preserve the core insight and technical contribution
- Simplify complex mathematical formulations if they caused issues
- Remove interdependencies if they caused integration errors

---

## Output Format

Output ONLY a refined Markdown proposal with the same structure as the original, but:
- Remove or simplify problematic modules
- Add a new section "## Refinement Summary" at the end explaining what was changed and why
"""
        
        system_prompt = """You are a senior AI researcher who specializes in refining research proposals.
Your goal is to make proposals more feasible by SIMPLIFYING them:
- REMOVE modules that are causing errors or are too complex
- MODIFY modules to be simpler and more robust
- NEVER ADD new modules - this is critical
The refined proposal should be SIMPLER than the original.
Output ONLY the refined Markdown proposal, nothing else."""
        
        # Call the LLM
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = query_model(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model_str=self.model_backbone,
                    temp=0.3  # low temperature for consistency
                )
                
                if isinstance(response, tuple):
                    response = response[0]
                
                # Clean the response — strip possible code-block markers
                refined_content = response.strip()
                if refined_content.startswith("```"):
                    refined_content = re.sub(r'^```\w*\n?', '', refined_content)
                    refined_content = re.sub(r'\n?```$', '', refined_content)
                
                # Validate: must contain a title and some content
                if not refined_content or len(refined_content) < 200:
                    logger.warning(f"Refined result too short: {len(refined_content)} characters")
                    continue
                
                # Validate: must contain a Refinement Summary section
                if "Refinement Summary" not in refined_content and "refinement" not in refined_content.lower():
                    logger.warning("Refined result missing Refinement Summary section; adding a hint and retrying")
                    continue
                
                logger.info(f"✅ Proposal refined successfully: {proposal_title}")
                return refined_content
                
            except Exception as e:
                logger.warning(f"Proposal refinement failed (attempt {attempt+1}/{max_retries}): {e}")
                continue
        
        logger.error(f"❌ Unable to refine Proposal: {proposal_title}; returning None")
        return None

    # ========================================================================
    # Research-context extraction
    # ========================================================================

    def _extract_research_context(self, idea_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract research context from the idea data."""
        context = {}
        
        # Extract dataset info
        if "dataset" in idea_data:
            dataset_info = idea_data["dataset"]
            context["dataset_name"] = dataset_info.get("name", "Unknown Dataset")
            context["dataset_description"] = dataset_info.get("description", "")
            context["task_type"] = dataset_info.get("task_type", "")
            context["modality"] = dataset_info.get("modality", "")
            context["target_structure"] = dataset_info.get("target_structure", "")
            context["patch_size"] = dataset_info.get("patch_size", None)
        
        # Extract baseline info (auto-generate if missing)
        if "baseline" in idea_data and idea_data["baseline"]:
            baseline_info = idea_data["baseline"]
        else:
            # Auto-generate default baseline
            dataset_info = idea_data.get("dataset", {})
            baseline_info = {
                "name": "nnUNet Baseline",
                "description": f"Standard nnUNet implementation for {dataset_info.get('target_structure', 'target')} segmentation using camylanet framework",
                "architecture": "nnUNet"
            }
        context["baseline_description"] = baseline_info.get("description", "")
        context["baseline_architecture"] = baseline_info.get("architecture", "")
        
        # Build the research topic
        context["research_topic"] = self._construct_research_topic(context)
        
        # Build dataset constraints
        context["dataset_constraints"] = self._construct_dataset_constraints(context)
        
        return context
    
    def _construct_research_topic(self, context: Dict[str, str]) -> str:
        """Build the research topic (broad scope — not tied to a specific dataset or modality)."""
        task_type = context.get("task_type", "")

        # Broaden the search scope beyond a specific dataset, modality, or medical domain
        if "segmentation" in task_type.lower():
            # Covers all segmentation tasks, not just medical images
            research_topic = "Advanced deep learning architectures for medical image segmentation"
        elif "classification" in task_type.lower():
            research_topic = "Novel deep learning approaches for image classification"
        elif "detection" in task_type.lower():
            research_topic = "Innovative deep learning methods for object detection"
        elif "generation" in task_type.lower():
            research_topic = "Cutting-edge deep learning techniques for image generation"
        else:
            # General computer vision and deep learning research
            research_topic = "Advanced deep learning architectures and novel methodologies for computer vision"
        
        research_topic += "\n =========================================="
        research_topic += "\n IMPORTANT: The project's focus must be exclusively on innovating the network architecture. Do not introduce any pre-trained models, self-supervised or semi-supervised learning schemes, multi-task objectives, novel data augmentation strategies, custom loss functions, or advanced post-processing techniques. The training process will be standard supervised learning using the provided images and labels."
        research_topic += "\n =========================================="
        
        return research_topic
    
    def _construct_dataset_constraints(self, context: Dict[str, str]) -> str:
        """Build the dataset-constraint description."""
        constraints = []
        
        dataset_name = context.get("dataset_name", "")
        if dataset_name and dataset_name != "Unknown Dataset":
            constraints.append(f"Dataset: {dataset_name}")
        
        dataset_desc = context.get("dataset_description", "")
        if dataset_desc:
            constraints.append(f"Description: {dataset_desc}")
        
        task_type = context.get("task_type", "")
        if task_type:
            constraints.append(f"Task: {task_type}")
        
        modality = context.get("modality", "")
        if modality:
            constraints.append(f"Modality: {modality}")
        
        target_structure = context.get("target_structure", "")
        if target_structure:
            constraints.append(f"Target: {target_structure}")
        
        patch_size = context.get("patch_size", None)
        if patch_size:
            constraints.append(f"Network Input Patch Size: {patch_size}")
            constraints.append("⚠️ CRITICAL SHAPE CONSTRAINT - All architectural innovations MUST be compatible with this input shape")
            constraints.append(f"🚨 Input tensor shape: (batch, channels, {patch_size})")
            constraints.append("❌ AVOID: Fixed kernel sizes that assume specific dimensions")
            constraints.append("❌ AVOID: Operations requiring specific divisibility (e.g., patch size must be divisible by 16)")
            constraints.append("❌ AVOID: Memory-intensive operations (e.g., full self-attention on high-resolution 3D volumes)")
            constraints.append("⚠️ DOWNSAMPLING LIMIT: (image) feature maps must NOT be smaller than 4 in any spatial dimension during downsampling")
            constraints.append("✅ USE: Adaptive pooling, relative position encoding, local attention windows")
        
        baseline_arch = context.get("baseline_architecture", "")
        if baseline_arch:
            constraints.append(f"Baseline Architecture: {baseline_arch}")
        
        return "; ".join(constraints) if constraints else "General medical image analysis task"
    
    def _phase1_challenge_discovery(self, research_context: Dict[str, str]) -> str:
        """
        Phase 1: challenge-discovery stage (memory-driven iterative version).

        New flow:
        1. Run a fixed number of search iterations.
        2. Extract challenges from 20 abstracts each round and append them to memory.
        3. PhD agent produces a new query from memory.
        4. Aggregate all challenges, distilling N distinct ones.
        5. Pick one at random via random.choice.
        6. Generate core_theme.

        Args:
            research_context: research context with task_type, modality, etc.

        Returns:
            core_theme: the generated core research theme.
        """
        logger.info(f"Phase 1: challenge-discovery stage (iterations: {self.phase1_search_iterations}, "
                    f"challenges per round: {self.phase1_challenges_per_round}, final challenges: {self.phase1_final_challenges})")
        
        challenge_memory = []  # stores all extracted challenges
        query_history = []     # tracks executed queries
        
        try:
            # ========== Step 1: build the initial search query ==========
            task_type = research_context.get('task_type', 'segmentation')
            modality = research_context.get('modality', '')
            target_structure = research_context.get('target_structure', '')
            
            # Concise query: target_structure + modality + task_type
            query_parts = []
            if target_structure:
                query_parts.append(target_structure)
            if modality:
                query_parts.append(modality)
            query_parts.append(task_type)
            current_query = " ".join(query_parts)
            
            logger.debug(f"Phase 1 initial search query: {current_query}")
            
            # ========== Step 2: fixed-count iterative search ==========
            for iteration in range(self.phase1_search_iterations):
                logger.debug(f"Phase 1 iteration {iteration + 1}/{self.phase1_search_iterations}, query: {current_query}")
                query_history.append(current_query)
                
                # Step 2a: run the search to fetch 20 paper abstracts
                papers = self._execute_phase1_search(current_query)
                
                if not papers:
                    logger.debug(f"Search returned no papers")
                    # Simplify the query and retry (except on the final round)
                    if iteration < self.phase1_search_iterations - 1:
                        current_query = self._simplify_query(
                            current_query, research_context, query_history
                        )
                        logger.debug(f"Query simplified: {current_query}")
                    continue
                
                logger.debug(f"Retrieved {len(papers)} paper abstracts")
                
                # Step 2b: extract challenges directly from abstracts
                round_challenges = self._extract_challenges_from_abstracts_batch(
                    papers, research_context, max_challenges=self.phase1_challenges_per_round
                )
                
                if round_challenges:
                    challenge_memory.extend(round_challenges)
                    logger.debug(f"Extracted {len(round_challenges)} challenges this round: {[c.get('name', 'Unknown') for c in round_challenges]}")
                else:
                    logger.debug(f"No challenges extracted this round")
                
                logger.debug(f"{len(challenge_memory)} challenges in memory")
                
                # Step 2c: if this is not the final round, generate a new query
                if iteration < self.phase1_search_iterations - 1:
                    current_query = self._generate_next_query_from_memory(
                        challenge_memory, research_context, query_history
                    )
                    logger.debug(f"Next-round query: {current_query}")
            
            # ========== Step 3: aggregate all challenges, distill N distinct ones ==========
            if challenge_memory:
                logger.debug(f"Aggregating {len(challenge_memory)} challenges...")
                final_challenges = self._consolidate_challenges(
                    challenge_memory, research_context, target_count=self.phase1_final_challenges
                )
                
                if final_challenges:
                    logger.debug(f"Aggregated {len(final_challenges)} distinct challenges: {[c.get('name', 'Unknown') for c in final_challenges]}")
                    
                    logger.debug(f"Retaining all {len(final_challenges)} challenges for diverse Proposal generation")
                    
                    # Store all challenges in discovered_challenges
                    self.discovered_challenges = final_challenges
                    
                    # Generate a core_theme for each challenge
                    self.challenge_themes = []
                    generated_themes = []  # tracks generated themes for deduplication
                    
                    for idx, challenge in enumerate(final_challenges):
                        logger.debug(f"Generating core_theme for challenge {idx+1}...")
                        theme = self._generate_core_theme_independently(
                            challenges=[challenge],
                            task_context=research_context,
                            previous_themes=generated_themes  # pass in already-generated themes
                        )
                        generated_themes.append(theme)  # record the new theme
                        self.challenge_themes.append({
                            'challenge': challenge,
                            'core_theme': theme
                        })
                        logger.debug(f"Challenge {idx+1}: {challenge.get('name', 'Unknown')} -> {theme}")
                    
                    # Return the first core_theme as the default (for backward compatibility)
                    core_theme = self.challenge_themes[0]['core_theme'] if self.challenge_themes else "architectural innovation"
                    
                    logger.info(f"Phase 1 complete: primary core_theme={core_theme}, {len(self.challenge_themes)} challenge-theme pairs total")
                    return core_theme
            
            logger.warning("Could not extract any valid challenges; using the default core_theme")
            return "architectural innovation"
            
        except Exception as e:
            logger.error(f"Phase 1 execution failed: {e}")
            return "architectural innovation"
    

    def _execute_phase1_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Run the Phase 1 search and return the paper list.

        Args:
            query: search query.

        Returns:
            List[Dict]: paper list; each entry includes id, title, abstract, etc.
        """
        try:
            papers = self.arxiv_search.find_papers_by_str(
                query=query,
                N=self.max_papers_per_search,
                disable_venue_filter=True  # disable top-venue filtering in Phase 1
            )
            return papers if papers else []
        except Exception as e:
            logger.error(f"Phase 1 search failed: {e}")
            return []
    
    def _extract_challenges_from_abstracts_batch(self, papers: List[Dict[str, Any]], 
                                                   research_context: Dict[str, str],
                                                   max_challenges: int = 3) -> List[Dict]:
        """
        Extract challenges from a batch of paper abstracts (processes 20 at a time).

        Args:
            papers: paper list; each entry has title and abstract.
            research_context: research context.
            max_challenges: maximum number of challenges to extract.

        Returns:
            List[Dict]: challenge list.
        """
        if not papers:
            return []
        
        # Build the combined-abstracts text
        combined_abstracts = ""
        for i, paper in enumerate(papers[:20], 1):  # process at most 20
            title = paper.get('title', 'Unknown')
            abstract = paper.get('abstract', '')
            if abstract and len(abstract) >= 50:
                combined_abstracts += f"\n--- Paper {i}: {title} ---\n{abstract}\n"
        
        if not combined_abstracts:
            return []
        
        logger.info(f"Batch-extracting challenges from {len(papers)} abstracts...")
        
        # Extract challenges via the skill prompt
        prompt = load_skill(
            "agents/challenge_discovery.md",
            task_type=research_context.get('task_type', 'segmentation'),
            modality=research_context.get('modality', ''),
            paper_content=combined_abstracts[:50000],  # cap length
            max_challenges=max_challenges
        )
        
        try:
            response = query_model(
                prompt=prompt,
                system_prompt="You are a research challenge analyst. Output valid JSON only.",
                model_str=self.challenge_extraction_model,
                temp=0.3
            )
            
            # Parse the JSON response
            response_text = BaseAgent.clean_text(response)
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                challenges = json_repair.loads(json_match.group())
                return challenges[:max_challenges] if len(challenges) > max_challenges else challenges
            return []
            
        except Exception as e:
            logger.error(f"Challenge extraction failed: {e}")
            return []
    
    def _generate_next_query_from_memory(self, challenge_memory: List[Dict], 
                                          research_context: Dict[str, str],
                                          query_history: List[str]) -> str:
        """
        Generate the next-round search query based on memory of discovered challenges.

        Args:
            challenge_memory: list of previously extracted challenges.
            research_context: research context.
            query_history: list of past queries.

        Returns:
            str: the new search query.
        """
        # Format the challenge memory
        challenges_text = ""
        for i, c in enumerate(challenge_memory, 1):
            challenges_text += f"{i}. {c.get('name', 'Unknown')}: {c.get('description', '')}\n"
        
        previous_queries = ", ".join(query_history) if query_history else "None"
        
        prompt = load_skill(
            "agents/phase1_query_generation.md",
            task_type=research_context.get('task_type', 'segmentation'),
            modality=research_context.get('modality', ''),
            target_structure=research_context.get('target_structure', ''),
            challenges_memory=challenges_text,
            previous_queries=previous_queries
        )
        
        try:
            response = query_model(
                prompt=prompt,
                system_prompt="You are a literature search query generator. Output only the search query, nothing else.",
                model_str=self.model_backbone,
                temp=0.5
            )
            
            # Clean the response, keeping only the query
            new_query = response.strip().strip('"').strip("'").strip()
            
            # Truncate if the response is too long
            if len(new_query) > 100:
                # Cap search-query length (search engines have length limits)
                new_query = new_query[:100]
            
            return new_query if new_query else f"{research_context.get('modality', '')} segmentation innovation"
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            # Backup query
            return f"{research_context.get('modality', '')} {research_context.get('task_type', 'segmentation')} architecture"
    
    def _simplify_query(self, failed_query: str, research_context: Dict[str, str], 
                        query_history: List[str]) -> str:
        """
        Simplify a failed search query to a more general query.
        Called when a search returns 0 results to generalize overly specific terms.
        
        Args:
            failed_query: The query that returned no results
            research_context: Research context with task_type, modality, etc.
            query_history: List of previous queries to avoid repetition
            
        Returns:
            str: A simplified, more general search query
        """
        logger.debug(f"Simplifying failed query: {failed_query}")
        
        previous_queries = ", ".join(query_history) if query_history else "None"
        task_type = research_context.get('task_type', 'segmentation')
        modality = research_context.get('modality', '')
        target_structure = research_context.get('target_structure', '')
        
        prompt = f"""You are a search query optimizer. A literature search query returned ZERO results because it was too specific.

Failed Query: "{failed_query}"
Task Type: {task_type}
Modality: {modality}
Target Structure: {target_structure}
Previous Queries: {previous_queries}

Your task: Generate a SIMPLER, MORE GENERAL query that is likely to find relevant papers.

Guidelines:
1. REMOVE overly specific anatomical terms (e.g., "Genioglossus, Geniohyoid, Mylohyoid" → "tongue muscle" or "oral muscle")
2. REMOVE rare medical terminology that may not appear in paper titles/abstracts
3. KEEP the core task type (e.g., "segmentation", "detection")
4. KEEP the imaging modality if relevant (e.g., "MRI", "CT")
5. ADD general deep learning terms if helpful (e.g., "deep learning", "neural network", "3D")
6. The query should be 3-8 words, suitable for academic paper search
7. DO NOT repeat any previous queries

Examples of simplification:
- "Genioglossus, Geniohyoid, Mylohyoid, Anterior Digastric muscles MRI segmentation" → "tongue muscle MRI segmentation deep learning"
- "Hypoxic ischemic encephalopathy lesion neonatal brain diffusion MRI" → "brain lesion MRI segmentation neonatal"
- "Hepatocellular carcinoma portal venous phase CT" → "liver tumor CT segmentation"

Output ONLY the simplified query, nothing else."""
        
        try:
            response = query_model(
                prompt=prompt,
                system_prompt="You are a search query optimizer. Output only the simplified query, nothing else.",
                model_str=self.model_backbone,
                temp=1.0
            )
            
            simplified_query = response.strip().strip('"').strip("'").strip()
            
            # Limit query length
            if len(simplified_query) > 80:
                simplified_query = simplified_query[:80]
            
            # Fallback if empty
            if not simplified_query:
                simplified_query = f"{modality} {task_type} deep learning" if modality else f"medical image {task_type}"
            
            logger.info(f"Simplified query: {failed_query} → {simplified_query}")
            return simplified_query
            
        except Exception as e:
            logger.error(f"Query simplification failed: {e}")
            # Fallback: use modality + task_type
            fallback = f"{modality} {task_type} deep learning" if modality else f"medical image {task_type}"
            logger.debug(f"Simplification failed, using fallback: {fallback}")
            return fallback
    
    def _consolidate_challenges(self, challenge_memory: List[Dict], 
                                 research_context: Dict[str, str],
                                 target_count: int = 3) -> List[Dict]:
        """
        Aggregate and deduplicate all challenges, distilling N distinct ones.

        Args:
            challenge_memory: all extracted challenges.
            research_context: research context.
            target_count: target number of challenges.

        Returns:
            List[Dict]: deduplicated challenge list.
        """
        if not challenge_memory:
            return []
        
        if len(challenge_memory) <= target_count:
            return challenge_memory

        # Format all challenges, tolerating varying data structures
        all_challenges_text = ""
        for i, c in enumerate(challenge_memory, 1):
            name = "Unknown"
            description = ""
            if isinstance(c, dict):
                name = str(c.get("name", "Unknown"))
                description = str(c.get("description", ""))
            elif isinstance(c, (list, tuple)):
                if len(c) >= 1:
                    name = str(c[0]) or "Unknown"
                if len(c) >= 2:
                    description = str(c[1])
            else:
                description = str(c)
                name = f"Challenge {i}"
            all_challenges_text += f"{i}. {name}: {description}\n"

        # Build modality description with dataset detail for single/multi-modal awareness
        modality_desc = research_context.get("modality", "")
        dataset_desc = research_context.get("dataset_constraints", "")
        if dataset_desc:
            modality_desc += f" — Dataset: {dataset_desc}"

        base_prompt = load_skill(
            "agents/challenge_consolidation.md",
            task_type=research_context.get("task_type", "segmentation"),
            modality=modality_desc,
            all_challenges=all_challenges_text,
            target_count=target_count,
        )

        def _call_and_parse(prompt: str, system_prompt: str, temp: float) -> Optional[List[Dict]]:
            response = query_model(
                prompt=prompt,
                system_prompt=system_prompt,
                model_str=self.model_backbone,
                temp=temp,
            )

            response_text = BaseAgent.clean_text(response)
            json_match = re.search(r"\[.*?\]", response_text, re.DOTALL)
            if not json_match:
                return None

            parsed = json_repair.loads(json_match.group())
            if not isinstance(parsed, list):
                return None

            valid_items: List[Dict] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                description = item.get("description")
                if not isinstance(name, str) or not isinstance(description, str):
                    continue
                valid_items.append({"name": name, "description": description})

            if not valid_items:
                return None

            return valid_items[:target_count] if len(valid_items) > target_count else valid_items

        # First attempt: standard constraints
        try:
            primary_system_prompt = (
                "You are consolidating research challenges. "
                "Output ONLY a valid JSON array of objects with keys 'name' and 'description'. "
                "Do not include any explanations, comments, or markdown."
            )
            primary_result = _call_and_parse(
                prompt=base_prompt,
                system_prompt=primary_system_prompt,
                temp=0.3,
            )
            if primary_result is not None:
                return primary_result

            logger.warning(
                "Challenge consolidation first attempt failed to produce valid JSON, "
                "retrying with stricter instructions."
            )

            # Second attempt: stronger formatting emphasis, lower temperature
            strict_suffix = (
                "\n\nIMPORTANT FORMAT REQUIREMENTS:\n"
                "- Return ONLY a JSON array `[...]`.\n"
                "- Each element MUST be a JSON object with exactly two string fields: \"name\" and \"description\".\n"
                "- Do NOT include any markdown code fences, natural language explanations, or additional text."
            )
            strict_prompt = f"{base_prompt}{strict_suffix}"
            strict_system_prompt = (
                "You are a strict JSON generator. "
                "Output ONLY a JSON array of objects with keys 'name' and 'description'. "
                "Do not output any explanation, comments, or markdown."
            )
            strict_result = _call_and_parse(
                prompt=strict_prompt,
                system_prompt=strict_system_prompt,
                temp=0.0,
            )
            if strict_result is not None:
                return strict_result

            logger.error("Both challenge-aggregation attempts failed to produce valid JSON; falling back to the original challenge list.")
        except Exception as e:
            logger.error(f"Challenge aggregation failed: {e}")

        # On failure, return the first target_count original challenges
        return challenge_memory[:target_count]

    
    def _perform_literature_search(self, research_context: Dict[str, str]) -> str:
        """Run literature search."""
        logger.info("Starting literature search")

        research_topic = research_context["research_topic"]
        dataset_constraints = research_context["dataset_constraints"]

        logger.debug(f"Literature-search topic: {research_topic}")
        logger.debug(f"Dataset constraints: {dataset_constraints}")
        
        # ========== Phase 1: challenge discovery ==========
        # Run challenge discovery before the technical search to determine core_theme
        if not self.theme_established:
            logger.info("Running Phase 1: challenge-discovery stage")
            self.core_theme = self._phase1_challenge_discovery(research_context)
            self.theme_established = True
            logger.info(f"Phase 1 complete, core_theme: {self.core_theme}. Moving to Phase 2: technical paper search")
        
        # ========== Phase 2: independent literature search (plan A) ==========
        # Run an independent literature-search loop for each theme
        logger.info(f"Phase 2: running independent literature search for {len(self.challenge_themes)} core themes")
        
        for theme_idx, theme_info in enumerate(self.challenge_themes):
            self._perform_independent_literature_search_for_theme(
                theme_idx=theme_idx,
                theme_info=theme_info,
                research_context=research_context
            )
        
        # Print Phase 2 summary
        logger.info("Phase 2 completion summary:")
        for idx, theme_info in enumerate(self.challenge_themes):
            papers_count = len(self.theme_lit_reviews.get(idx, []))
            keywords = self.theme_keyword_pools.get(idx, [])
            logger.debug(f"  Theme {idx}: {theme_info['core_theme'][:50]}... - papers: {papers_count}, keywords: {keywords[:3]}")
        
        # Print the overall literature-search summary
        self._display_literature_search_summary()

        # Generate the literature-review summary
        literature_summary = self._generate_literature_summary()
        
        # Print theme evolution history
        if self.search_theme_history:
            logger.debug(f"Theme evolution: core theme={self.core_theme}, trace={' -> '.join(self.search_theme_history)}")


        return literature_summary

    def _perform_independent_literature_search_for_theme(
        self, 
        theme_idx: int,
        theme_info: Dict[str, Any],
        research_context: Dict[str, str]
    ) -> str:
        """
        Run an independent literature-search loop for a single theme.

        Args:
            theme_idx: theme index.
            theme_info: dict containing 'challenge' and 'core_theme'.
            research_context: research context.

        Returns:
            Technical-methods corpus string for this theme.
        """
        core_theme = theme_info['core_theme']
        challenge = theme_info['challenge']
        
        logger.info(f"Theme {theme_idx}: {core_theme} (challenge: {challenge.get('name', 'Unknown')})")
        
        # Mark the current theme being processed
        self.current_theme_idx = theme_idx
        
        # Initialize per-theme storage
        self.theme_keyword_pools[theme_idx] = []
        self.theme_lit_reviews[theme_idx] = []
        
        # ===== Save original state (for later restoration) =====
        original_keyword_pool = self.keyword_pool
        original_lit_review = self.phd.lit_review.copy()
        original_lit_review_sum = getattr(self.phd, 'lit_review_sum', '')
        original_search_history = self.search_theme_history.copy()
        original_history = self.phd.history.copy()  # conversation history
        original_paper_cache = self.phd.paper_cache.copy() if hasattr(self.phd, 'paper_cache') else {}
        
        # ===== Reset to a clean state (for independent search) =====
        self.keyword_pool = []
        self.phd.lit_review = []
        self.phd.lit_review_sum = ''
        self.phd.history = []  # reset conversation history to avoid cross-theme contamination
        if hasattr(self.phd, 'paper_cache'):
            self.phd.paper_cache = {}
        self.search_theme_history = []
        # Note: searched_paper_ids is kept to prevent selecting the same paper across themes
        
        research_topic = research_context["research_topic"]
        dataset_constraints = research_context["dataset_constraints"]
        
        # Set agent notes (including current-theme info)
        notes = [
            {
                "phases": ["literature review"],
                "note": f"Current Core Theme: {core_theme}"
            },
            {
                "phases": ["literature review"],
                "note": f"Challenge Focus: {challenge.get('name', 'Unknown')} - {challenge.get('description', '')}"
            },
            {
                "phases": ["literature review"],
                "note": f"Dataset and Task Constraints: {dataset_constraints}"
            },
            {
                "phases": ["literature review"],
                "note": "STRICT FOCUS: Only search for NETWORK ARCHITECTURES and MODULES related to the core theme."
            }
        ]
        self.phd.notes = notes
        
        # Run the search loop (fixed number of iterations per theme)
        iterations_per_theme = max(5, self.max_literature_iterations // len(self.challenge_themes))
        target_papers_per_theme = max(2, self.target_papers_in_review // len(self.challenge_themes))
        
        logger.debug(f"Target paper count: {target_papers_per_theme}, max iterations: {iterations_per_theme}")
        
        for iteration in range(iterations_per_theme):
            logger.debug(f"Theme {theme_idx} iteration {iteration + 1}/{iterations_per_theme}")
            
            # Check completion condition
            if len(self.phd.lit_review) >= target_papers_per_theme:
                logger.debug(f"Theme {theme_idx} reached target paper count ({len(self.phd.lit_review)}/{target_papers_per_theme})")
                break
            
            # Decide the search query
            if self.keyword_pool and len(self.keyword_pool) > 0:
                selected_query = random.choice(self.keyword_pool)
                logger.debug(f"Selected from keyword pool: {selected_query} (pool: {self.keyword_pool})")
            else:
                # First search: build the initial query from core_theme
                selected_query = f"{core_theme}"
                logger.debug(f"Initial query (based on core_theme): {selected_query}")
            
            # Run the search
            search_data = {
                'function_name': 'search_papers',
                'query': selected_query
            }
            feedback = self._handle_search_papers(search_data)
            self.search_theme_history.append(selected_query)
            
            # Let the PhD agent pick papers
            selection_prompt = f"{feedback}\n\n⚠️ IMPORTANT: Choose papers most relevant to the core theme: {core_theme}\nUse get_full_text ONLY on papers you want to ADD to your review!"
            resp = self.phd.inference(research_topic, "literature review", step=iteration, feedback=selection_prompt, temp=0.4)
            
            if isinstance(resp, dict):
                self._handle_function_call_response(resp, iteration)
            else:
                self._handle_text_response(resp)
        
        # Persist this theme's search results
        self.theme_keyword_pools[theme_idx] = self.keyword_pool.copy()
        self.theme_lit_reviews[theme_idx] = self.phd.lit_review.copy()
        self.theme_paper_methods[theme_idx] = self._get_combined_paper_methods()
        
        logger.debug(f"Theme {theme_idx} complete: papers={len(self.theme_lit_reviews[theme_idx])}, "
                     f"keywords={self.theme_keyword_pools[theme_idx]}, "
                     f"methods corpus={len(self.theme_paper_methods[theme_idx])} characters")
        
        # ===== Restore original state (accumulate all papers) =====
        # Paper list: accumulate papers from every theme
        self.phd.lit_review = original_lit_review + self.phd.lit_review
        # Update the literature-review summary
        self.phd.lit_review_sum = "Provided here is a literature review on this topic:\n" + "\n".join(
            f"arXiv ID: {entry['arxiv_id']}, Summary: {entry.get('summary', '')}"
            for entry in self.phd.lit_review
        )
        # Keyword pool: restore the original (per-theme keywords already live in theme_keyword_pools)
        self.keyword_pool = original_keyword_pool
        # Conversation history: restore the original (don't accumulate to avoid huge histories)
        self.phd.history = original_history
        # Paper cache: accumulate
        if hasattr(self.phd, 'paper_cache'):
            current_cache = self.phd.paper_cache.copy()
            self.phd.paper_cache = original_paper_cache
            self.phd.paper_cache.update(current_cache)  # newer entries overwrite older ones
        # Search history: accumulate
        self.search_theme_history = original_search_history + self.search_theme_history
        
        self.current_theme_idx = None
        
        logger.info(f"Theme {theme_idx} literature search complete: {len(self.theme_lit_reviews[theme_idx])} papers")
        
        return self.theme_paper_methods[theme_idx]

    def _display_literature_search_summary(self):

        """Print the literature-search completion summary, including the list of post-2023 papers."""
        current_papers = len(self.phd.lit_review)

        logger.info(f"Literature search complete: collected {current_papers} papers after 2023 (target: {self.target_papers_in_review})")

        if current_papers > 0:
            for i, paper in enumerate(self.phd.lit_review, 1):
                arxiv_id = paper.get('arxiv_id', 'Unknown ID')
                summary = paper.get('summary', 'No summary available')
                pub_year = self._extract_publication_year(paper)
                year_str = f" ({pub_year})" if pub_year else ""
                logger.debug(f"  {i:2d}. {arxiv_id}{year_str}: {summary[:150]}{'...' if len(summary) > 150 else ''}")
        else:
            logger.warning(f"No papers were collected")

    def _extract_core_theme_from_method_info(self, method_info: Dict, paper_title: str = "") -> str:
        """Extract the core theme from technical details."""
        logger.debug(f"Extracting core theme from technical details...")
        
        # Fetch the extracted innovation points
        innovations = method_info.get('json_data', {}).get('innovations', [])
        if not innovations:
            logger.debug(f"No technical innovations found; using the default theme")
            return "architectural innovation"
        
        # Build the innovation summary
        innovations_summary = []
        for i, innovation in enumerate(innovations, 1):  # use all innovations
            name = innovation.get('name', '')
            desc = innovation.get('description', '')  # use the full description
            if name:
                innovations_summary.append(f"{i}. {name}: {desc}")
        
        innovations_text = "\n".join(innovations_summary)
        
        # Use the LLM to extract the core theme
        prompt = f"""Based on the technical innovations extracted from a paper titled "{paper_title}", identify ONE core technical theme that best represents the paper's main research direction.

Technical Innovations:
{innovations_text}

Requirements:
- Extract a RESEARCH-LEVEL theme that could serve as a paper title (2-6 words)
- Focus on the RESEARCH DIRECTION rather than specific implementation details
- The theme should represent a conceptual contribution suitable as a paper title
- Should be abstract enough to encompass multiple technical approaches
- Prefer emerging research directions over specific technical methods

Examples of GOOD themes (paper-title level) DO NOT USE THESE EXAMPLES:
- "Dynamic Aggregation", "Token-driven Architecture", "Structure-Aware Fusion"
- "Diversity-enhanced Collaborative Mamba", "Mask-Attribute Alignment"
- "Structure-Aware State Fusion", "Uncertainty-aware Reward Modeling"
- "Prototype-Guided Graph Reasoning Network", "Neighborhood Correlation Mining"

Examples of TOO SPECIFIC (avoid these):
- "Batch Normalization Improvement", "3x3 Convolution Optimization"
- "Learning Rate Scheduling", "Dropout Rate Tuning"

Examples of TOO GENERIC (avoid these):
- "Novel Architecture", "Efficient Network", "Improved Model"

Examples of TOO OUTDATED (avoid these established technologies):
- "Attention Mechanism", "Residual Connection", "Skip Connection"

The theme should be a research direction that can support multiple innovations, not a single technical implementation detail or well-established baseline technology.

Output ONLY the core theme, nothing else."""

        try:
            response = query_model(
                prompt=prompt,
                system_prompt="You are an AI research expert who identifies core technical themes from architectural innovations.",
                model_str=self.model_backbone,
                temp=0.3
            )
            
            core_theme = response.strip().strip('"').strip("'")
            
            logger.debug(f"Core theme established: {core_theme} (paper: {paper_title})")
            return core_theme
            
        except Exception as e:
            logger.error(f"Failed to extract core theme from technical details: {e}")
            # Fallback: use the first innovation's name
            if innovations:
                fallback_theme = innovations[0].get('name', 'architectural innovation')
                logger.debug(f"Extraction failed; using the first innovation name: {fallback_theme}")
                return fallback_theme
            return "architectural innovation"
    
    def _generate_core_theme_independently(self, challenges: List[Dict], 
                                           task_context: Dict,
                                           previous_themes: List[str] = None) -> str:
        """
        Independently generate core_theme (do not extract from papers; avoids incremental variants).

        Differences vs. the old _extract_core_theme_from_method_info:
        - Input: task challenges + task context (instead of paper technical details).
        - Generation: LLM forms the theme independently without copying any existing method name.
        - Goal: an original research direction suitable as a paper-title theme.

        Args:
            challenges: list of challenges extracted from papers.
            task_context: task-context info.
            previous_themes: list of already-generated themes (to avoid duplicates).

        Returns:
            Core research theme (2-6 words).
        """
        logger.debug("Independently generating the core research theme...")
        
        if not challenges:
            logger.debug(f"No challenges provided; using the default theme")
            return "architectural innovation"
        
        # Build previous_themes constraint
        previous_themes_constraint = ""
        if previous_themes:
            previous_themes_constraint = "\n⛔ ALREADY GENERATED THEMES (DO NOT REPEAT):\n"
            for i, theme in enumerate(previous_themes, 1):
                previous_themes_constraint += f"   {i}. {theme}\n"
            previous_themes_constraint += "\n⚠️ Your theme MUST be SIGNIFICANTLY DIFFERENT from these.\n"
            previous_themes_constraint += "💡 DIFFERENTIATION STRATEGY: Use a different core architecture paradigm or focus on a different aspect of the network."
        
        # Use the skill prompt
        prompt = load_skill(
            "agents/core_theme_generation.md",
            task_type=task_context.get('task_type', 'segmentation'),
            modality=task_context.get('modality', ''),
            target_structure=task_context.get('target_structure', ''),
            challenge_name=challenges[0].get('name', 'Unknown challenge'),
            challenge_description=challenges[0].get('description', ''),
            previous_themes_constraint=previous_themes_constraint
        )
        
        # Get the dedicated core_theme model from configuration
        try:
            from camyla.model_config import get_model_name
            core_theme_model = get_model_name('core_theme')
            core_theme_temp = 0.5  # temperature from config
            logger.debug(f"Using dedicated model: {core_theme_model}")
        except Exception as e:
            logger.warning(f"Could not load core_theme model config; using defaults: {e}")
            core_theme_model = self.model_backbone
            core_theme_temp = 0.5
        
        try:
            response = query_model(
                prompt=prompt,
                system_prompt="You are an expert AI researcher proposing novel research directions for top-tier conferences. Focus on practical, implementable architectures using standard deep learning terminology. Output the theme only.",
                model_str=core_theme_model,
                temp=core_theme_temp
            )
            
            core_theme = response.strip().strip('"').strip("'")
            
            logger.debug(f"Core theme independently generated: {core_theme} (based on challenge: {challenges[0].get('name', 'Unknown')}"
                        f"{f', avoiding {len(previous_themes)} existing themes' if previous_themes else ''})")
            return core_theme
            
        except Exception as e:
            logger.error(f"Failed to independently generate core_theme: {e}")
            return "architectural innovation"
    
    def _handle_function_call_response(self, resp_data, iteration):
        """Handle the structured response from modern function calling."""
        if not isinstance(resp_data, dict):
            return "Error: Expected structured function call response"

        function_name = resp_data.get('function_name', '')

        # Handle the new composite-function format
        if function_name == 'literature_review_actions':
            action = resp_data.get('action', '')
            if action == 'search_papers':
                # Convert to search_papers format
                search_data = {
                    'function_name': 'search_papers',
                    'query': resp_data.get('query', '')
                }
                feedback = self._handle_search_papers(search_data)
                
                # Record the search theme
                query = resp_data.get('query', '')
                if query:
                    self.search_theme_history.append(query)
                
                return feedback
            elif action == 'get_full_text':
                # Convert to get_full_text format
                fulltext_data = {
                    'function_name': 'get_full_text',
                    'paper_id': resp_data.get('paper_id', '')
                }
                return self._handle_get_full_text(fulltext_data)
            else:
                return f"Error: Unknown action '{action}' for literature_review_actions"

        else:
            return f"Error: Unknown function '{function_name}'. Please use literature_review_actions."

    def _handle_text_response(self, resp):
        """Handle a text response."""
        current_papers = len(self.phd.lit_review)
        if current_papers >= self.target_papers_in_review:
            return "Literature review completed."
        elif current_papers > 0:
            return f"You have collected {current_papers} papers so far. Continue searching until you have at least {self.target_papers_in_review} papers in your review."
        else:
            return "Please start by searching for relevant papers."

    def _handle_search_papers(self, resp_data, disable_venue_filter: bool = False):
        """Handle the search_papers function call.
        
        Args:
            resp_data: response data containing query.
            disable_venue_filter: whether to disable top-venue filtering (used by Phase 1).
        """
        query = resp_data.get('query', '')
        if not query:
            return "Error: Missing query parameter for search_papers"

        logger.debug(f"Running search query: {query}")

        papers_list = self.arxiv_search.find_papers_by_str(
            query, 
            N=self.max_papers_per_search,
            disable_venue_filter=disable_venue_filter
        )
        if papers_list is None:
            logger.error(f"ArXiv search failed")
            return "Error: ArXiv search failed"

        # ✨ Paper deduplication: drop papers already selected via get_full_text.
        # Note: only filter here — do not add to searched_paper_ids.
        # Additions happen in _handle_get_full_text.
        original_count = len(papers_list) if papers_list else 0
        filtered_papers = []
        
        for paper in papers_list:
            paper_id = paper.get('id', '')
            if paper_id and paper_id not in self.searched_paper_ids:
                filtered_papers.append(paper)
                # Do not add here — only add on get_full_text
            elif paper_id:
                logger.debug(f"Skipping already-selected paper: {paper_id}")

        
        papers_list = filtered_papers
        paper_count = len(papers_list)
        
        logger.debug(f"Found {original_count} papers; {paper_count} remain after deduplication (post-2023)")
        for i, paper in enumerate(papers_list, 1):
            logger.debug(f"  [{i}] {paper.get('title', 'Unknown Title')}")

        # Cache structured paper data in the PhD student's search cache
        if not hasattr(self.phd, 'paper_cache'):
            self.phd.paper_cache = {}

        for paper in papers_list:
            self.phd.paper_cache[paper['id']] = paper

        # Format the paper list as readable text
        formatted_papers = self._format_papers_for_display(papers_list, query)

        feedback = f"Search completed successfully. Found {paper_count} papers from 2023 to 2025:\n{formatted_papers}\n\n"
        feedback += f"NEXT STEP: Choose one most promising paper that meet our criteria and are published in top-tier CV conferences or journals, and get their full text using their paper IDs."
        return feedback

    def _format_papers_for_display(self, papers_list: List[Dict], query: str) -> str:
        """Format a structured paper list as readable text."""
        if not papers_list:
            return "No papers found."

        formatted_papers = []
        for i, paper in enumerate(papers_list, 1):
            paper_text = f"Paper {i}:\n"
            paper_text += f"ID: {paper.get('id', 'Unknown')}\n"
            paper_text += f"Title: {paper.get('title', 'Unknown Title')}\n"
            paper_text += f"Abstract: {paper.get('abstract', 'No abstract available')}\n"
            paper_text += f"Publication Date: {paper.get('publication_date', 'Unknown')}\n"
            if paper.get('authors'):
                authors_str = ', '.join(paper['authors'][:3])  # show only the first 3 authors
                if len(paper['authors']) > 3:
                    authors_str += f" et al. ({len(paper['authors'])} authors total)"
                paper_text += f"Authors: {authors_str}\n"
            paper_text += f"URL: {paper.get('url', 'No URL')}\n"
            formatted_papers.append(paper_text)

        return "\n" + "="*80 + "\n".join(formatted_papers) + "\n" + "="*80

    def _handle_get_full_text(self, resp_data):
        """Handle the get_full_text function call — uses PaperSummaryAgent to extract method content."""
        paper_id = resp_data.get('paper_id', '')
        if not paper_id:
            return "Error: Missing paper_id parameter for get_full_text"

        logger.debug(f"Fetching paper full text: {paper_id}")

        # Fetch paper full text
        # Fetch paper full text (try reading source info from cache)
        source = None
        if hasattr(self.phd, 'paper_cache') and paper_id in self.phd.paper_cache:
            source = self.phd.paper_cache[paper_id].get('source')
        full_text = self.arxiv_search.retrieve_full_paper_text(paper_id, source=source)
        text_length = len(full_text) if full_text else 0
        logger.debug(f"Full-text length: {text_length} characters")

        if not full_text or len(full_text.strip()) < 100:
            return f"Full text retrieval failed for {paper_id}"

        try:
            # Try to obtain the paper title from search results
            paper_title = self._extract_paper_title_from_history(paper_id)

            logger.debug(f"Using PaperSummaryAgent to extract method content...")
            method_info = self.paper_summary_agent.extract_methods_from_full_text(
                paper_id=paper_id,
                full_text=full_text,
                paper_title=paper_title
            )

            # Note: core_theme is now determined in Phase 1 (_phase1_challenge_discovery) at the start of literature search.
            # Challenge-driven core_theme generation is no longer needed here.

            logger.debug(f"Using CitationNetworkAgent to analyze the citation network...")
            citation_info = self.citation_network_agent.analyze_citations_from_full_text(
                paper_id=paper_id,
                full_text=full_text,
                paper_title=paper_title,
                core_theme=self.core_theme if self.theme_established else ""
            )

            # Store keywords in the keyword pool
            keywords = citation_info.get('search_keywords', [])
            if keywords:
                # Update the keyword pool (keep the latest 5 keywords)
                self.keyword_pool = keywords[-5:] if len(keywords) >= 5 else keywords
                logger.debug(f"Keyword pool updated: {self.keyword_pool}")
                
                # ===== New: synchronously update the current theme's keyword pool =====
                if self.current_theme_idx is not None:
                    self.theme_keyword_pools[self.current_theme_idx] = self.keyword_pool.copy()
                    logger.debug(f"Theme {self.current_theme_idx} keyword pool synchronized")


            # Format the method summary for return
            method_summary = self._format_method_summary_for_feedback(method_info)

            # Format the citation-analysis result
            citation_summary = self._format_citation_analysis_for_feedback(citation_info)

            logger.debug(f"Method extraction complete: {paper_id}, summary length: {len(method_summary)} characters")
            logger.debug(f"Citation analysis complete: {paper_id}, keyword pool: {self.keyword_pool}")

            # 🆕 Automatically add the paper to review (get_full_text implies an add)
            paper_info = self.phd.paper_cache.get(paper_id, {}) if hasattr(self.phd, 'paper_cache') else {}
            review_entry = {
                "arxiv_id": paper_id,
                "summary": method_summary,  # use the extracted method summary
                "title": paper_info.get('title', paper_title if paper_title else ''),
                "abstract": paper_info.get('abstract', ''),
                "publication_date": paper_info.get('publication_date', ''),
                "authors": paper_info.get('authors', []),
                "url": paper_info.get('url', '')
            }
            self.phd.lit_review.append(review_entry)
            
            # ✨ Record paper IDs in the searched set for downstream deduplication
            self.searched_paper_ids.add(paper_id)
            
            # Update lit_review_sum
            self.phd.lit_review_sum = "Provided here is a literature review on this topic:\n" + "\n".join(
                f"arXiv ID: {entry['arxiv_id']}, Title: {entry.get('title', 'Unknown')}, Summary: {entry['summary']}"
                for entry in self.phd.lit_review
            )
            
            logger.debug(f"Paper automatically added to review: {paper_id}, total: {len(self.phd.lit_review)}")

            # Return a simplified success message
            feedback = f"✅ Successfully retrieved, analyzed, and ADDED paper {paper_id} to your review.\n\n"
            feedback += f"📊 Current progress: {len(self.phd.lit_review)} papers in review.\n\n"
            feedback += f"🔑 Keywords updated: {self.keyword_pool}\n\n"
            feedback += "Continue searching for more papers using the updated keywords."

            return feedback

        except Exception as e:
            logger.error(f"Paper method extraction failed: {paper_id}, error: {e}")

            # Even if method extraction fails, still add the paper with basic info
            paper_info = self.phd.paper_cache.get(paper_id, {}) if hasattr(self.phd, 'paper_cache') else {}
            
            # Use the paper's abstract or preview as the summary
            if len(full_text) > 5000:
                summary = full_text[:5000] + "...[Full text retrieved but detailed method extraction failed]"
            else:
                summary = paper_info.get('abstract', full_text[:5000] if full_text else 'No summary available')
            
            review_entry = {
                "arxiv_id": paper_id,
                "summary": summary,
                "title": paper_info.get('title', ''),
                "abstract": paper_info.get('abstract', ''),
                "publication_date": paper_info.get('publication_date', ''),
                "authors": paper_info.get('authors', []),
                "url": paper_info.get('url', '')
            }
            self.phd.lit_review.append(review_entry)
            
            # Update lit_review_sum
            self.phd.lit_review_sum = "Provided here is a literature review on this topic:\n" + "\n".join(
                f"arXiv ID: {entry['arxiv_id']}, Title: {entry.get('title', 'Unknown')}, Summary: {entry['summary']}"
                for entry in self.phd.lit_review
            )
            
            logger.warning(f"Paper added but method extraction failed: {paper_id}, total: {len(self.phd.lit_review)}")
            
            return f"⚠️ Retrieved and ADDED paper {paper_id} (method extraction failed, using basic info). Current progress: {len(self.phd.lit_review)} papers."

    def _extract_paper_title_from_history(self, paper_id: str) -> str:
        """Extract the paper title from the cache or history."""
        try:
            # First try the structured cache
            if hasattr(self.phd, 'paper_cache') and paper_id in self.phd.paper_cache:
                return self.phd.paper_cache[paper_id].get('title', '')

            # If missing from cache, fall back to history (backward-compatible with older format)
            for record in self.phd.history:
                if paper_id in record and "Title: " in record:
                    lines = record.split('\n')
                    for line in lines:
                        if line.strip().startswith('Title: ') and paper_id in record:
                            # Located the search-result block containing this paper_id
                            title_line = line.strip()
                            return title_line[7:]  # strip the "Title: " prefix
            return ""
        except Exception as e:
            logger.warning(f"Failed to extract paper title: {e}")
            return ""

    def _format_method_summary_for_feedback(self, method_info: Dict[str, str]) -> str:
        """Format a method summary for feedback."""
        if not method_info or not method_info.get('summary'):
            return "Method extraction failed - no technical content available."

        # Build a structured method summary
        formatted_summary = []

        # Add basic paper info
        if method_info.get('paper_title'):
            formatted_summary.append(f"Paper: {method_info['paper_title']}")

        # Prefer the detailed JSON info when available
        if method_info.get('json_data') and method_info['json_data'].get('innovations'):
            formatted_summary.append("\n🔧 EXTRACTED INNOVATIONS:")

            for i, innovation in enumerate(method_info['json_data']['innovations'], 1):
                formatted_summary.append(f"\n  {i}. {innovation.get('name', 'Unknown Innovation')}")
                formatted_summary.append(f"     Description: {innovation.get('description', 'No description')}")

                if innovation.get('implementation'):
                    formatted_summary.append(f"     Implementation: {innovation['implementation']}")
        else:
            # Fall back to the raw summary
            formatted_summary.append(f"\nExtracted Content:\n{method_info.get('summary', 'No method content extracted.')}")

        return '\n'.join(formatted_summary)

    def _format_citation_analysis_for_feedback(self, citation_info: Dict[str, Any]) -> str:
        """Format the citation-analysis result for feedback."""
        if not citation_info or not citation_info.get('json_data'):
            return "Citation analysis failed - no citation data available."

        # Build a structured citation-analysis summary
        formatted_summary = []

        # Add basic paper info
        if citation_info.get('paper_title'):
            formatted_summary.append(f"Paper: {citation_info['paper_title']}")

        json_data = citation_info.get('json_data', {})

        # Add citation statistics
        all_citations = json_data.get('all_citations', [])
        filtered_citations = json_data.get('filtered_citations', [])
        search_keywords = json_data.get('search_keywords', [])

        formatted_summary.append(f"\n📊 CITATION STATISTICS:")
        formatted_summary.append(f"   Total citations found: {len(all_citations)}")
        formatted_summary.append(f"   High-quality recent citations (2023+): {len(filtered_citations)}")
        formatted_summary.append(f"   Generated search keywords: {len(search_keywords)}")

        # Show the top filtered citations
        if filtered_citations:
            formatted_summary.append(f"\n🔗 HIGH-QUALITY RECENT CITATIONS:")
            for i, citation in enumerate(filtered_citations[:5], 1):  # only show the top 5
                title = citation.get('title', 'Unknown Title')
                venue = citation.get('venue', 'Unknown Venue')
                year = citation.get('year', 'Unknown Year')
                reason = citation.get('relevance_reason', 'No reason provided')
                formatted_summary.append(f"   {i}. {title}")
                formatted_summary.append(f"      Venue: {venue} ({year})")
                formatted_summary.append(f"      Relevance: {reason}")

        # Show the generated search keywords
        if search_keywords:
            formatted_summary.append(f"\n🎯 GENERATED SEARCH KEYWORDS:")
            for i, keyword in enumerate(search_keywords, 1):
                formatted_summary.append(f"   {i}. {keyword}")
            formatted_summary.append(f"\n💡 These keywords will be added to the keyword pool for future searches.")

        return '\n'.join(formatted_summary)

    def _extract_publication_year(self, paper: Dict) -> str:
        """Extract publication year from paper info."""
        try:
            # Try to extract the year from arxiv_id
            arxiv_id = paper.get('arxiv_id', '')
            if arxiv_id:
                # arXiv ID format is usually YYMM.NNNNN or YYYY.NNNNN
                if '.' in arxiv_id:
                    year_part = arxiv_id.split('.')[0]
                    if len(year_part) == 4 and year_part.isdigit():
                        # Four-digit year like 2103.14030 -> 2023
                        year_str = year_part[:2]  # take the first two digits
                        if year_str.isdigit():
                            year_num = int(year_str)
                            # 21xx -> 2023, 22xx -> 2022, etc.
                            if year_num >= 91:  # 91xx -> 1991
                                return f"19{year_str}"
                            else:  # 00xx-90xx -> 2000-2090
                                return f"20{year_str}"
                    elif len(year_part) == 2 and year_part.isdigit():
                        # Two-digit year like 21.03545 -> 2023
                        year_num = int(year_part)
                        if year_num >= 91:
                            return f"19{year_part}"
                        else:
                            return f"20{year_part}"

            # If the year cannot be extracted from arxiv_id, return an empty string
            return ""

        except Exception as e:
            logger.warning(f"Error extracting publication year: {e}")
            return ""

    def _generate_literature_summary(self) -> str:
        """Generate the literature-review summary (incorporating extracted method content)."""
        if not self.phd.lit_review:
            return "No literature review available."

        # Collect abstracts for every paper
        papers_summary = []
        for paper in self.phd.lit_review:
            paper_info = f"Paper: {paper['arxiv_id']}\nSummary: {paper['summary']}\n"
            papers_summary.append(paper_info)

        combined_summary = "\n".join(papers_summary)

        # Append the extracted technical methods
        extracted_methods = self.paper_summary_agent.format_methods_for_prompt()
        if extracted_methods and "No technical methods extracted yet." not in extracted_methods:
            logger.debug(f"Integrated extracted technical methods; length: {len(extracted_methods)} characters")

            # Append the method content to the literature review
            enhanced_summary = f"{combined_summary}\n\n{'='*80}\nEXTRACTED TECHNICAL METHODS FROM PAPERS:\n{extracted_methods}"
        else:
            enhanced_summary = combined_summary
            logger.debug("No extracted technical methods found; using the traditional abstract instead")

        # Set the lit-review summary
        self.phd.lit_review_sum = enhanced_summary
        self.postdoc.lit_review_sum = enhanced_summary

        return enhanced_summary

    def _reset_literature_search_state(self):
        """Reset literature-search state to avoid re-searching the same papers."""
        # Clear the PhDStudentAgent's literature review
        self.phd.lit_review.clear()
        self.phd.lit_review_sum = ""
        
        # Clear the paper cache
        if hasattr(self.phd, 'paper_cache'):
            self.phd.paper_cache.clear()
        
        # Reset the citation-network agent
        if hasattr(self.citation_network_agent, 'extracted_keywords'):
            self.citation_network_agent.extracted_keywords = []
        if hasattr(self.citation_network_agent, 'citation_analysis_results'):
            self.citation_network_agent.citation_analysis_results = []
            
        logger.debug("Literature-search state has been reset")
    
