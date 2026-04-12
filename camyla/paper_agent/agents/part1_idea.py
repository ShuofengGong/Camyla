import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader

from camyla.paper_agent.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class PaperSummaryAgent(BaseAgent):
    """
    Agent responsible for reading PDFs and extracting innovations.
    """
    def run(self, pdf_paths: List[str], topic: str, dataset_info: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        summaries = []
        for idx, pdf_path in enumerate(pdf_paths, 1):
            text = self._read_pdf(pdf_path)
            if not text:
                logger.warning(f"Could not read text from {pdf_path}")
                continue
            
            logger.info(f"Summarizing {pdf_path}...")
            
            # Generate dataset context (handles single/multiple datasets)
            dataset_context = ""
            if dataset_info and len(dataset_info) > 0:
                if len(dataset_info) == 1:
                    # Single dataset
                    ds = dataset_info[0]
                    task_type = ds.get('task', 'segmentation')
                    name = ds.get('name', 'Unknown')
                    modalities = ', '.join(ds.get('modalities', []))
                    classes_list = ds.get('classes', [])
                    classes_brief = ', '.join(classes_list[:3])
                    if len(classes_list) > 3:
                        classes_brief += ', etc.'
                    
                    context_content = f"""You are analyzing papers with potential applicability to {task_type} tasks.

Example target scenario:
- Dataset: {name}
- Modalities: {modalities}
- Typical targets: {classes_brief}"""
                    
                    extraction_focus = """✓ Innovations applicable to SIMILAR tasks (not just this dataset)
✓ General principles with TRANSFER POTENTIAL
✓ Methodological contributions beyond dataset-specific optimizations"""
                    
                else:
                    # Multiple datasets
                    dataset_names = [ds.get('name', 'Unknown') for ds in dataset_info]
                    
                    context_content = f"""You are analyzing papers with potential applicability to multi-domain medical image segmentation.

Target scenarios ({len(dataset_info)} datasets):
{chr(10).join([f"  - {name}" for name in dataset_names])}"""
                    
                    extraction_focus = """✓ Innovations with CROSS-DOMAIN applicability
✓ General principles that transfer across different imaging modalities
✓ Methodological contributions beyond dataset-specific optimizations"""
                
                # Use skill file to format the context
                dataset_context = self.load_skill(
                    "common/dataset_context.md",
                    context_content=context_content,
                    extraction_focus=extraction_focus
                )
            
            prompt = self.load_skill(
                "medical_segmentation/summary_agent.md",
                paper_text=text[:100000],  # Truncate to avoid context limit if necessary
                dataset_context=dataset_context
            )
            
            response = self.chat(messages=[{"role": "user", "content": prompt}])
            summaries.append({
                "source_id": f"Source {idx}",
                "summary": response
            })
        return summaries

    def _read_pdf(self, path: str) -> str:
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {path}: {e}")
            return ""

class IdeaRefineAgent(BaseAgent):
    """
    Agent responsible for refining and generalizing innovations.
    """
    def run(self, summaries: List[Dict[str, Any]]) -> str:
        # Extract innovations from summaries (assuming summaries are text blobs containing "Innovations:" sections)
        # For simplicity, we pass the full summaries to the LLM to extract/refine.
        
        combined_summaries = "\n\n".join([
            f"=== {s['source_id']} ===\n{s['summary']}" 
            for s in summaries
        ])
        
        logger.info("Refining innovations...")
        prompt = self.load_skill(
            "medical_segmentation/idea_refine.md",
            innovations=combined_summaries
        )
        
        response = self.chat(messages=[{"role": "user", "content": prompt}])
        return response

class IdeaGenerationAgent(BaseAgent):
    """
    Agent responsible for generating a new idea from refined innovations.
    """
    def run(
        self, 
        refined_innovations: str, 
        dataset_info: Optional[List[Dict]] = None,
        dataset_challenges: Optional[str] = None,
        image_dimension: str = "Not specified",
        custom_instructions: Optional[str] = None,
        task_mode: str = "fully_supervised"
    ) -> str:
        logger.info("Generating new idea...")
        
        # Process custom instructions
        instruction_context = ""
        if custom_instructions:
            logger.info("Using custom instructions for idea generation")
            instruction_context = f"""
## Custom Research Guidance
================================================================
{custom_instructions}
================================================================
Please **prioritize** the above instructions when generating ideas.
"""
        
        # Generate dataset challenge/context description
        dataset_context = ""
        
        if dataset_challenges:
            # Use extracted challenges (already considers all datasets)
            logger.info("Using extracted dataset challenges from PDF analysis")
            dataset_context = f"""Target Dataset Information and Challenges:
================================================================
{dataset_challenges}

**Your Analysis Task**:
1. Map innovations from the pool to the above challenges
2. Propose a unified core method that addresses these challenges
3. Select 3 techniques that synergistically implement this method
================================================================
"""
        elif dataset_info and len(dataset_info) > 0:
            # Generate context based on dataset count
            if len(dataset_info) == 1:
                # Single dataset - detailed description
                ds = dataset_info[0]
                name = ds.get('name', 'Unknown')
                full_name = ds.get('full_name', 'N/A')
                task = ds.get('task', 'segmentation')
                modalities = ', '.join(ds.get('modalities', []))
                classes = ', '.join(ds.get('classes', []))
                metrics = ', '.join(ds.get('metrics', []))
                train_samples = ds.get('train_samples', 'N/A')
                val_samples = ds.get('val_samples', 'N/A')
                background = ds.get('background', '')
                
                dataset_context = f"""Target Dataset Information:
================================================================
**Dataset**: {name}
**Full Name**: {full_name}
**Task Type**: {task}

**Key Characteristics**:
-  **Imaging Modalities**: {modalities}
- **Target Structures**: {classes}
- **Evaluation Metrics**: {metrics}
- **Dataset Scale**: {train_samples} training / {val_samples} validation samples

**Background Context**: 
{background}

**Your Analysis Task**:
1. Based on the above characteristics, identify 2-3 SPECIFIC technical challenges this dataset/task presents
2. Map innovations from the pool to these challenges
3. Propose a unified core method that addresses these challenges through synergistic combination
================================================================
"""
            else:
                # Multiple datasets - summarized description
                ds_summaries = []
                for i, ds in enumerate(dataset_info, 1):
                    ds_summaries.append(f"""Dataset {i}: {ds.get('name', 'Unknown')}
- Task: {ds.get('task', 'N/A')}
- Modalities: {', '.join(ds.get('modalities', []))}
- Key targets: {', '.join(ds.get('classes', [])[:3])}""")
                
                # Differentiate based on task_mode
                if task_mode in ["domain_adaptation", "domain_generalization"]:
                    # True domain adaptation/generalization task
                    dataset_context = f"""Multi-Dataset Domain Adaptation/Generalization ({len(dataset_info)} datasets):
================================================================
{chr(10).join(ds_summaries)}

**TASK MODE**: {task_mode}
**CRITICAL REQUIREMENT**: Your method MUST address domain shift:
✓ Design for CROSS-DOMAIN generalization
✓ Handle distribution differences between source and target domains
✓ Propose domain-invariant feature learning or adaptation strategies
✓ Consider using "Robust", "Generalizable", or "Domain-Adaptive" in title

**Your Analysis Task**:
1. Identify domain shift challenges between these datasets
2. Propose domain adaptation/generalization techniques
3. Design methods that transfer across different imaging domains
================================================================
"""
                else:
                    # Fully supervised or other modes - multi-dataset evaluation (NOT domain generalization)
                    dataset_context = f"""Multi-Dataset Evaluation ({len(dataset_info)} datasets):
================================================================
{chr(10).join(ds_summaries)}

**TASK MODE**: {task_mode} (Multi-dataset evaluation)
**IMPORTANT CLARIFICATION**:
⚠️ This is MULTI-DATASET EVALUATION, NOT domain adaptation/generalization
✓ Each dataset is trained and tested INDEPENDENTLY in standard fully-supervised manner
✓ Multiple datasets demonstrate method's BROAD APPLICABILITY
✓ Do NOT add "Robust", "Generalizable", or "Domain-Adaptive" modifiers to title
✓ Do NOT focus on domain shift or cross-domain transfer
✓ Focus on the specific domain (e.g., "Thyroid Nodule Segmentation")

**Your Analysis Task**:
1. Identify 2-3 COMMON technical challenges across these datasets
2. Map innovations to address these shared challenges
3. Propose a unified method applicable to this specific domain
✓ Main experiments will evaluate on ALL {len(dataset_info)} datasets
✓ Ablation studies will focus on the primary (first) dataset
================================================================
"""
        
        prompt = self.load_skill(
            "medical_segmentation/idea_gen.md",
            refined_innovations=refined_innovations,
            dataset_challenges=dataset_context,
            image_dimension=image_dimension,
            custom_instructions=instruction_context
        )
        
        response = self.chat(messages=[{"role": "user", "content": prompt}])
        return response

class IdeaVerificationAgent(BaseAgent):
    """
    Agent responsible for verifying and detailing the proposed idea.
    """
    def run(
        self, 
        proposed_idea: str, 
        dataset_info: Optional[List[Dict]] = None, 
        image_dimension: str = "Not specified",
        custom_instructions: Optional[str] = None,
        task_mode: str = "fully_supervised"
    ) -> str:
        logger.info("Verifying idea (Stage 1/2: Critical Review)...")
        
        # Process custom instructions
        instruction_context = ""
        if custom_instructions:
            instruction_context = f"""
## Custom Research Guidance
When verifying the idea, ensure the research proposal aligns with the following user-specified directions:
{custom_instructions}
"""
        
        # Generate dataset information for prompt
        dataset_prompt = ""
        if dataset_info and len(dataset_info) > 0:
            if len(dataset_info) == 1:
                # Single dataset - detailed context
                ds = dataset_info[0]
                name = ds.get('name', 'Unknown')
                full_name = ds.get('full_name', 'N/A')
                task_type = ds.get('task', 'N/A')
                modalities = ', '.join(ds.get('modalities', []))
                classes = ', '.join(ds.get('classes', []))
                metrics = ', '.join(ds.get('metrics', []))
                train = ds.get('train_samples', 'N/A')
                val = ds.get('val_samples', 'N/A')
                background = ds.get('background', '')
                
                dataset_prompt = f"""Application Context (for validation purposes):

We will validate the proposed method on the following benchmark:

**Benchmark Dataset**: {name}
- **Full Name**: {full_name}
- **Task**: {task_type} (representative of Medical Image Segmentation challenges)
- **Key Characteristics**:
  - Imaging modalities: {modalities}
  - Target structures: {classes}
  - Standard evaluation: {metrics}
- **Dataset Scale**:
  - Training: {train} cases
  - Validation: {val} cases

**Context**: {background}

**How to use this information**:
================================================================
✓ Treat as a VALIDATION SCENARIO to demonstrate method effectiveness
✓ Methods should be GENERAL SOLUTIONS applicable to similar segmentation tasks
✓ Dataset-specific details belong in 'Experimental Validation' section
✓ Frame your approach as solving BROAD CHALLENGES this dataset exemplifies
✓ Avoid over-constraining the method design to this specific benchmark
================================================================
"""
            else:
                # Multiple datasets - summarized context
                ds_list = []
                for i, ds in enumerate(dataset_info, 1):
                    ds_list.append(f"{i}. {ds.get('name', 'Unknown')} ({ds.get('task', 'N/A')})")
                
                primary_name = dataset_info[0].get('name', 'Unknown')
                
                # Differentiate based on task_mode
                if task_mode in ["domain_adaptation", "domain_generalization"]:
                    dataset_prompt = f"""Multi-Dataset Domain Adaptation/Generalization Strategy:

We will validate the proposed method on {len(dataset_info)} datasets for domain transfer:
{chr(10).join(ds_list)}

**Task Mode**: {task_mode}
**Validation Approach**:
- **Main Experiments**: Evaluate cross-domain generalization across ALL {len(dataset_info)} datasets
- **Ablation Studies**: Conduct detailed component analysis on {primary_name} (primary dataset)

**How to use this information**:
================================================================
✓ Focus on DOMAIN SHIFT challenges between datasets
✓ Method should handle distribution differences
✓ Consider domain-invariant features and adaptation strategies
✓ Use "Robust", "Generalizable", or "Domain-Adaptive" modifiers in title
✓ Dataset-specific details belong in 'Experiments' section
================================================================
"""
                else:
                    dataset_prompt = f"""Multi-Dataset Evaluation Strategy:

We will evaluate the proposed method on {len(dataset_info)} benchmark datasets:
{chr(10).join(ds_list)}

**Task Mode**: {task_mode} (Standard multi-dataset evaluation)
**Validation Approach**:
- **Main Experiments**: Evaluate independently on ALL {len(dataset_info)} datasets
- **Ablation Studies**: Conduct detailed component analysis on {primary_name} (primary dataset)

**IMPORTANT - This is NOT domain adaptation**:
================================================================
⚠️ Each dataset is trained and tested INDEPENDENTLY (fully supervised)
⚠️ Do NOT use "Robust", "Generalizable", or "Domain-Adaptive" modifiers
⚠️ Do NOT claim "cross-domain generalization" - this is multi-dataset evaluation
✓ Method demonstrates broad applicability within this specific domain
✓ Frame as solving COMMON CHALLENGES in this medical imaging field
✓ Use specific task name (e.g., "Thyroid Nodule Segmentation")
================================================================
"""
        
        # === STAGE 1: Critical Review ===
        review_prompt = self.load_skill(
            "medical_segmentation/idea_critical_review.md",
            proposed_idea=proposed_idea,
            dataset_info=dataset_prompt,
            image_dimension=image_dimension,
            custom_instructions=instruction_context
        )
        
        # First LLM call: Get critical review
        review_response = self.chat(messages=[{"role": "user", "content": review_prompt}])
        logger.info("  ✓ Critical review completed")
        
        # === STAGE 2: Generate Polished Proposal ===
        logger.info("Verifying idea (Stage 2/2: Generate Proposal)...")
        
        # Build second prompt with FULL CONTEXT
        # Includes: original idea + dataset info + review feedback
        proposal_prompt = self.load_skill(
            "medical_segmentation/verification.md",
            proposed_idea=proposed_idea,
            dataset_info=dataset_prompt,
            image_dimension=image_dimension,
            custom_instructions=instruction_context,
            review_feedback=review_response  # NEW: Include review from stage 1
        )
        
        # Second LLM call: Get polished proposal
        final_proposal = self.chat(messages=[{"role": "user", "content": proposal_prompt}])
        logger.info("  ✓ Research proposal generated")
        
        return final_proposal

