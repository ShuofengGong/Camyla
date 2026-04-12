#!/usr/bin/env python3
"""
OpenHands log analyzer.

Automatically analyzes OpenHands code-generation logs and produces summary reports.
Includes:
1. A detailed summary of each individual interaction.
2. A full report for the entire stage (journal + openhands).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

logger = logging.getLogger("camyla")


class OpenHandsLogAnalyzer:
    """OpenHands log analyzer - generates summary reports."""

    def __init__(self, cfg):
        """Initialize the analyzer.

        Args:
            cfg: Configuration object.
        """
        self.cfg = cfg

        # LLM configuration
        self.llm_client = None
        self.llm_model = None

        logger.info("📊 OpenHandsLogAnalyzer initialized")

    def _init_llm(self):
        """Initialize the LLM client (lazy initialization)."""
        if self.llm_client is None:
            from camyla.llm import create_client
            from camyla.model_config import get_model_name

            model_name = get_model_name('log_summary')
            self.llm_client, self.llm_model = create_client(model_name)
            logger.info(f"🤖 LLM initialized: {model_name}")

    def _find_log_pairs(self, logs_dir: Path) -> List[Tuple[Path, Path, str]]:
        """Find and pair log files.

        Args:
            logs_dir: Path to the openhands_logs directory.

        Returns:
            List[(jsonl_path, md_path, timestamp)] sorted by timestamp.
        """
        if not logs_dir.exists():
            logger.warning(f"Logs directory not found: {logs_dir}")
            return []

        # Scan all .jsonl files
        jsonl_files = list(logs_dir.glob("openhands_events_*.jsonl"))

        log_pairs = []

        for jsonl_path in jsonl_files:
            # Extract the timestamp from the filename.
            # Format: openhands_events_20250101_120000.jsonl
            match = re.search(r'openhands_events_(\d{8}_\d{6})\.jsonl', jsonl_path.name)
            if not match:
                logger.warning(f"Could not extract timestamp from: {jsonl_path.name}")
                continue

            timestamp = match.group(1)

            # Find the corresponding .md file
            md_path = logs_dir / f"openhands_summary_{timestamp}.md"

            if md_path.exists():
                log_pairs.append((jsonl_path, md_path, timestamp))
            else:
                logger.warning(f"Matching .md file not found for: {jsonl_path.name}")

        # Sort by timestamp
        log_pairs.sort(key=lambda x: x[2])

        logger.info(f"Found {len(log_pairs)} log pairs")
        return log_pairs

    def _merge_log_files(self, jsonl_path: Path, md_path: Path) -> str:
        """Directly merge the contents of the two log files.

        Args:
            jsonl_path: JSONL event log path.
            md_path: Markdown summary path.

        Returns:
            Merged text content.
        """
        merged_content = []

        # 1. Read JSONL content
        merged_content.append("## JSONL Event Log\n")
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                jsonl_content = f.read()
                # Cap the length to avoid exceeding LLM limits
                max_jsonl_length = 50000  # roughly 12.5k tokens
                if len(jsonl_content) > max_jsonl_length:
                    jsonl_content = jsonl_content[:max_jsonl_length] + "\n...(truncated)"
                merged_content.append(jsonl_content)
        except Exception as e:
            logger.error(f"Error reading JSONL file: {e}")
            merged_content.append(f"Error reading file: {e}\n")

        merged_content.append("\n\n---\n\n")

        # 2. Read Markdown content
        merged_content.append("## Markdown Interaction Summary\n")
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
                # Cap the length
                max_md_length = 50000
                if len(md_content) > max_md_length:
                    md_content = md_content[:max_md_length] + "\n...(truncated)"
                merged_content.append(md_content)
        except Exception as e:
            logger.error(f"Error reading MD file: {e}")
            merged_content.append(f"Error reading file: {e}\n")

        return '\n'.join(merged_content)

    def _generate_interaction_summary_prompt(self, merged_log: str, timestamp: str) -> str:
        """Build the LLM analysis prompt.

        Loads a prompt template from the skills directory and substitutes variables.

        Args:
            merged_log: Merged log content.
            timestamp: Interaction timestamp.

        Returns:
            The full prompt.
        """
        from skills.prompt_loader import load_skill

        # Load the prompt template from the skills directory
        prompt = load_skill(
            "agents/openhands_interaction_summary.md",
            timestamp=timestamp,
            merged_log=merged_log
        )

        return prompt

    def _call_llm_for_summary(self, prompt: str) -> str:
        """Call the LLM to generate a summary.

        Args:
            prompt: Analysis prompt.

        Returns:
            The LLM-generated summary.
        """
        self._init_llm()

        # Type check: ensure the LLM has been initialized
        if self.llm_client is None or self.llm_model is None:
            logger.error("LLM client initialization failed")
            return "## Summary generation failed\n\nError: LLM client not initialized"

        from camyla.llm import get_response_from_llm

        system_msg = "You are a professional AI systems analyst skilled at analyzing and summarizing AI agent behavior logs."

        try:
            response = get_response_from_llm(
                prompt,
                self.llm_client,
                self.llm_model,
                system_msg
            )
            return response[0]
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"## Summary generation failed\n\nError: {str(e)}"

    def analyze_openhands_interactions(self, openhands_logs_dir: Path) -> List[Path]:
        """Analyze OpenHands interaction logs and produce per-interaction summaries.

        Args:
            openhands_logs_dir: Path to the openhands_logs directory.

        Returns:
            List of generated summary file paths.
        """
        logger.info(f"📊 Analyzing OpenHands interactions in: {openhands_logs_dir}")

        # 1. Find all log pairs
        log_pairs = self._find_log_pairs(openhands_logs_dir)

        if not log_pairs:
            logger.warning("No log pairs found")
            return []

        # 2. Create the output directory
        output_dir = openhands_logs_dir / "openhands_logs_summary"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 3. Generate a summary for each interaction
        generated_files = []

        for i, (jsonl_path, md_path, timestamp) in enumerate(log_pairs, 1):
            logger.info(f"Processing interaction {i}/{len(log_pairs)}: {timestamp}")

            # Merge the log files
            merged_log = self._merge_log_files(jsonl_path, md_path)

            # Build the prompt
            prompt = self._generate_interaction_summary_prompt(merged_log, timestamp)

            # Call the LLM
            summary = self._call_llm_for_summary(prompt)

            # Save the summary
            output_file = output_dir / f"interaction_{i}_{timestamp}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# OpenHands Interaction {i} - {timestamp}\n\n")
                f.write(summary)

            generated_files.append(output_file)
            logger.info(f"✅ Saved summary to: {output_file}")

        logger.info(f"📊 Generated {len(generated_files)} interaction summaries")
        return generated_files

    def _correlate_journal_with_openhands(
        self,
        journal_path: Path,
        openhands_summaries: List[Dict]
    ) -> List[Dict]:
        """Correlate journal nodes with OpenHands logs by node order.

        Because journal nodes do not have explicit timestamps, we rely on node order:
        - Assume nodes are created in chronological order.
        - OpenHands interactions are also in chronological order.
        - Simple strategy: distribute interactions evenly across nodes.

        Args:
            journal_path: Path to journal.json.
            openhands_summaries: List of OpenHands interaction summaries.

        Returns:
            Correlated data structure.
        """
        # 1. Read journal.json
        with open(journal_path, 'r', encoding='utf-8') as f:
            journal_data = json.load(f)

        nodes = journal_data.get('nodes', [])

        if not nodes:
            logger.warning("No nodes found in journal")
            return []

        # 2. Simple strategy: assign interactions to nodes in order.
        # If there are N nodes and M interactions, each node gets roughly M/N interactions.
        total_interactions = len(openhands_summaries)
        total_nodes = len(nodes)

        correlated_data = []

        if total_interactions == 0:
            # No interactions: every node has no associated interactions
            for node in nodes:
                correlated_data.append({
                    'node': node,
                    'openhands_interactions': []
                })
        else:
            # Compute the number of interactions allotted per node
            interactions_per_node = total_interactions / total_nodes

            for i, node in enumerate(nodes):
                # Compute the interaction range for this node
                start_idx = int(i * interactions_per_node)
                end_idx = int((i + 1) * interactions_per_node)

                # The last node absorbs all remaining interactions
                if i == total_nodes - 1:
                    end_idx = total_interactions

                # Get the interactions for this node
                node_interactions = openhands_summaries[start_idx:end_idx]

                correlated_data.append({
                    'node': node,
                    'openhands_interactions': node_interactions
                })

        logger.info(f"Correlated {total_nodes} nodes with {total_interactions} interactions")
        return correlated_data

    def _format_metric(self, metric: Dict) -> str:
        """Format metric info into a human-readable string.

        Args:
            metric: Metric dict.

        Returns:
            Formatted string.
        """
        if isinstance(metric, dict):
            if 'value' in metric:
                metric_value = metric['value']
                if isinstance(metric_value, dict) and 'metric_names' in metric_value:
                    # Handle the metric_names format
                    metric_names = metric_value['metric_names']
                    parts = []
                    for m in metric_names:
                        metric_name = m.get('metric_name', 'Unknown')
                        description = m.get('description', 'N/A')
                        parts.append(f"{metric_name}: {description}")
                    return ', '.join(parts)
            return str(metric)
        return "N/A"

    def generate_stage_summary_report(
        self,
        stage_logs_dir: Path,
        stage_name: str
    ) -> Optional[Path]:
        """Generate a complete stage summary report.

        Args:
            stage_logs_dir: Stage log directory (containing journal.json and openhands_workspace).
            stage_name: Stage name.

        Returns:
            Path to the generated report, or None on failure.
        """
        logger.info(f"📊 Generating complete stage summary for: {stage_name}")

        # 1. Get the journal.json path
        journal_path = stage_logs_dir / "journal.json"

        if not journal_path.exists():
            logger.warning(f"journal.json not found (likely precomputed baseline stage): {journal_path}")
            return None

        # 2. Fetch OpenHands summaries (openhands_logs sits alongside openhands_workspace)
        openhands_summaries_dir = stage_logs_dir / "openhands_logs" / "openhands_logs_summary"

        # Read all OpenHands interaction summaries
        openhands_summaries = []
        if openhands_summaries_dir.exists():
            for summary_file in sorted(openhands_summaries_dir.glob("interaction_*.md")):
                # Extract the timestamp from the filename
                match = re.search(r'interaction_(\d+)_(\d{8}_\d{6})\.md', summary_file.name)
                if match:
                    interaction_num = int(match.group(1))
                    timestamp = match.group(2)
                else:
                    logger.warning(f"Could not parse filename: {summary_file.name}")
                    continue

                with open(summary_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                openhands_summaries.append({
                    'num': interaction_num,
                    'timestamp': timestamp,
                    'content': content,
                    'file': summary_file.name
                })

        # 3. Correlate journal with openhands
        correlated_data = self._correlate_journal_with_openhands(
            journal_path,
            openhands_summaries
        )

        # 4. Build the report
        report_lines = []

        # Report header
        report_lines.append(f"# Stage Summary: {stage_name}\n\n")
        report_lines.append(f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"**Total nodes**: {len(correlated_data)}\n")
        report_lines.append(f"**OpenHands interactions**: {len(openhands_summaries)}\n")
        report_lines.append("\n---\n\n")

        # Emit content in node order
        for i, item in enumerate(correlated_data, 1):
            node = item['node']
            interactions = item['openhands_interactions']

            # Node heading
            plan = node.get('plan', 'Unknown plan')
            report_lines.append(f"## Step {i}: {plan}\n\n")

            # Basic node info
            node_id = node.get('id', 'unknown')
            report_lines.append(f"**Node ID**: `{node_id}`\n")

            # Metric info
            metric = node.get('metric', {})
            if metric:
                formatted_metric = self._format_metric(metric)
                report_lines.append(f"**Metric**: {formatted_metric}\n")

            # Success/failure status
            is_buggy = node.get('is_buggy', False)
            status_icon = "❌ Failed" if is_buggy else "✅ Succeeded"
            report_lines.append(f"**Status**: {status_icon}\n")

            # Analysis (if any)
            analysis = node.get('analysis', '')
            if analysis:
                report_lines.append(f"\n**Analysis**: {analysis}\n")

            report_lines.append("\n")

            # Detailed OpenHands operation log
            if interactions:
                report_lines.append(f"### Detailed operation log for Step {i}\n\n")

                for j, interaction in enumerate(interactions, 1):
                    timestamp = interaction.get('timestamp', 'unknown')
                    report_lines.append(f"#### Interaction {i}.{j} ({timestamp})\n\n")
                    report_lines.append(interaction['content'])
                    report_lines.append("\n\n")
            else:
                report_lines.append(f"### 📝 No detailed operation log\n\n")
                report_lines.append("This step has no associated OpenHands interaction log.\n\n")

            report_lines.append("---\n\n")

        # Overall stage analysis
        report_lines.append("## Overall Stage Analysis\n\n")

        total_nodes = len(correlated_data)
        successful_nodes = sum(1 for item in correlated_data if not item['node'].get('is_buggy', False))
        failed_nodes = total_nodes - successful_nodes

        report_lines.append("### Statistics Overview\n\n")
        report_lines.append(f"- **Total experiments**: {total_nodes}\n")
        report_lines.append(f"- **Successes**: {successful_nodes}\n")
        report_lines.append(f"- **Failures**: {failed_nodes}\n")

        if total_nodes > 0:
            success_rate = successful_nodes / total_nodes * 100
            report_lines.append(f"- **Success rate**: {success_rate:.1f}%\n")

        # 5. Save the report
        output_path = stage_logs_dir / "stage_openhands_summary.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(report_lines))

        logger.info(f"📊 Complete stage report generated: {output_path}")
        return output_path
