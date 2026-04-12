import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LaTeX post-processing: automatic deduplication
# ---------------------------------------------------------------------------

def deduplicate_latex(content: str) -> str:
    """Remove duplicate figures, sections, and structural anomalies from
    LLM-generated LaTeX.  Designed to be called as a batch-safe post-processor
    before compilation.

    Handles:
      1. Duplicate figure environments sharing the same \\label
      2. Consecutive identical \\section / \\subsection / \\subsubsection headings
      3. \\subsection{Conclusion} inside Discussion when a standalone
         \\section{Conclusion} also exists
    """
    content = _dedup_figure_environments(content)
    content = _dedup_consecutive_headings(content)
    content = _dedup_conclusion(content)
    return content


def deduplicate_bibtex_entries(content: str) -> str:
    """Drop duplicate BibTeX entries by citation key while preserving order.

    The first occurrence of a key is kept. Non-entry text outside BibTeX
    records is preserved as-is.
    """
    if not content or not content.strip():
        return content

    text = content.replace("\r\n", "\n")
    length = len(text)
    i = 0
    chunks: list[str] = []
    seen_keys: set[str] = set()
    removed = 0

    while i < length:
        at_pos = text.find("@", i)
        if at_pos == -1:
            chunks.append(text[i:])
            break

        chunks.append(text[i:at_pos])

        header_match = re.match(r"@(\w+)\s*([({])\s*", text[at_pos:])
        if not header_match:
            chunks.append(text[at_pos])
            i = at_pos + 1
            continue

        open_char = header_match.group(2)
        close_char = "}" if open_char == "{" else ")"
        entry_start = at_pos
        body_start = at_pos + header_match.end()

        comma_pos = text.find(",", body_start)
        if comma_pos == -1:
            chunks.append(text[entry_start:])
            break

        key = text[body_start:comma_pos].strip()
        if not key:
            chunks.append(text[entry_start:comma_pos + 1])
            i = comma_pos + 1
            continue

        depth = 1
        j = body_start
        while j < length and depth > 0:
            char = text[j]
            if char == "\\":
                j += 2
                continue
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
            j += 1

        entry_text = text[entry_start:j]
        if key in seen_keys:
            removed += 1
        else:
            seen_keys.add(key)
            chunks.append(entry_text)

        i = j

    result = "".join(chunks)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    if result:
        result += "\n"

    if removed:
        logger.info(
            "deduplicate_bibtex_entries: removed %s duplicate BibTeX entr%s",
            removed,
            "y" if removed == 1 else "ies",
        )
    return result


_FIGURE_BLOCK_RE = re.compile(
    r"(\\begin\{figure\}.*?\\end\{figure\})",
    re.DOTALL,
)
_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")


def _dedup_figure_environments(content: str) -> str:
    """Keep only the first \\begin{figure}...\\end{figure} for each \\label."""
    seen_labels: set[str] = set()
    removals = 0

    def _replacer(m: re.Match) -> str:
        nonlocal removals
        block = m.group(0)
        label_m = _LABEL_RE.search(block)
        if label_m is None:
            return block
        label = label_m.group(1)
        if label in seen_labels:
            removals += 1
            return ""
        seen_labels.add(label)
        return block

    result = _FIGURE_BLOCK_RE.sub(_replacer, content)
    if removals:
        logger.info(f"deduplicate_latex: removed {removals} duplicate figure environment(s)")
        result = re.sub(r"\n{3,}", "\n\n", result)
    return result


_HEADING_RE = re.compile(
    r"(\\(?:section|subsection|subsubsection)\{[^}]*\})"
)


def _dedup_consecutive_headings(content: str) -> str:
    """Remove a heading that is immediately repeated (with only whitespace between)."""
    removals = 0

    def _replacer(m: re.Match) -> str:
        nonlocal removals
        removals += 1
        return m.group(1)

    pattern = re.compile(
        r"(\\(?:section|subsection|subsubsection)\{([^}]*)\})"
        r"\s*"
        r"\\(?:section|subsection|subsubsection)\{\2\}"
    )
    result = pattern.sub(_replacer, content)
    if removals:
        logger.info(f"deduplicate_latex: removed {removals} consecutive duplicate heading(s)")
    return result


def _dedup_conclusion(content: str) -> str:
    r"""If both \subsection{Conclusion} and \section{Conclusion} exist,
    fold the subsection body into the section and remove the subsection."""
    subsec_pat = re.compile(
        r"\\subsection\{Conclusion\}\s*\n(.*?)(?=\\(?:sub)?section\{|\\end\{document\})",
        re.DOTALL,
    )
    has_section = bool(re.search(r"\\section\{Conclusion\}", content))
    subsec_match = subsec_pat.search(content)

    if has_section and subsec_match:
        subsec_body = subsec_match.group(1).strip()
        content = content[:subsec_match.start()] + content[subsec_match.end():]
        content = re.sub(r"\n{3,}", "\n\n", content)

        sec_pat = re.compile(r"(\\section\{Conclusion\}\s*\n)")
        sec_m = sec_pat.search(content)
        if sec_m and subsec_body:
            insert_pos = sec_m.end()
            content = content[:insert_pos] + subsec_body + "\n\n" + content[insert_pos:]

        logger.info("deduplicate_latex: merged \\subsection{Conclusion} into \\section{Conclusion}")

    return content


@dataclass
class CompilationResult:
    """LaTeX compilation result."""

    success: bool
    output_path: Optional[Path]
    log_path: Optional[Path]
    stdout: str
    stderr: str


@dataclass
class TemplateInfo:
    """Template configuration."""

    name: str
    path: Path
    main_file: str = "main.tex"
    description: str = ""


DEFAULT_TEMPLATE_ROOT = Path(__file__).with_name("latex_templates")


class LatexCompiler:
    """General LaTeX compiler with template-extension support."""

    def __init__(self, timeout: int = 60, template_root: Optional[Path] = None):
        self.timeout = timeout
        self.template_root = template_root or DEFAULT_TEMPLATE_ROOT
        self.template_root.mkdir(parents=True, exist_ok=True)

    def compile_content(
        self,
        latex_content: str,
        output_pdf: Path,
        template_name: Optional[str] = None,
        template_dir: Optional[Path] = None,
        assets: Optional[list] = None,
        log_path: Optional[Path] = None,
        figure_dir: Optional[Path] = None,
    ) -> CompilationResult:
        """
        Compile a LaTeX string to PDF.

        Args:
            latex_content: Full LaTeX text.
            output_pdf: Target PDF path.
            template_name: Template name (looked up under template_root).
            template_dir: Optional template directory; all files are copied.
            assets: Additional resources (images, etc.) to copy into the temporary directory.
            log_path: Optional log file to which pdflatex/bibtex stdout/stderr is written.
            figure_dir: Figure directory, copied into the temp compile dir for \\includegraphics resolution.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            template_info = None
            selected_template_dir = self._resolve_template_dir(template_name, template_dir)

            if selected_template_dir:
                template_info = self._load_template_info(selected_template_dir, template_name or selected_template_dir.name)
                self._copy_template_dir(selected_template_dir, tmp_dir)

            # Write main.tex
            main_file = template_info.main_file if template_info else "main.tex"
            main_tex = tmp_dir / main_file
            main_tex.parent.mkdir(parents=True, exist_ok=True)
            main_tex.write_text(latex_content, encoding="utf-8")

            # Copy assets
            if assets:
                assets_dir = tmp_dir / "assets"
                assets_dir.mkdir(exist_ok=True)
                for asset in assets:
                    asset_path = Path(asset)
                    if asset_path.exists() and asset_path.is_file():
                        shutil.copy2(asset_path, assets_dir / asset_path.name)
            
            # Copy references.bib to compilation directory
            bib_source = output_pdf.parent / "references.bib"
            if bib_source.exists():
                bib_dest = tmp_dir / "references.bib"
                bib_text = bib_source.read_text(encoding="utf-8", errors="replace")
                bib_dest.write_text(
                    deduplicate_bibtex_entries(bib_text),
                    encoding="utf-8",
                )
                logger.info(f"Copied references.bib to compilation directory")
            else:
                logger.warning(f"No references.bib found at {bib_source}")

            # Copy figure directory so \includegraphics{figure/...} resolves
            if figure_dir and figure_dir.exists() and figure_dir.is_dir():
                dest_figure_dir = tmp_dir / "figure"
                shutil.copytree(figure_dir, dest_figure_dir, dirs_exist_ok=True)
                fig_count = sum(1 for _ in dest_figure_dir.rglob("*") if _.is_file())
                logger.info(f"Copied {fig_count} figure files to compilation directory")
            elif figure_dir:
                logger.warning(f"Figure directory not found: {figure_dir}")

            entry_file = main_file
            result = self._run_pdflatex_pipeline(tmp_dir, entry_file, log_path)

            pdf_path = tmp_dir / f"{Path(entry_file).stem}.pdf"
            if pdf_path.exists():
                output_pdf.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pdf_path, output_pdf)
                return CompilationResult(
                    success=True,
                    output_path=output_pdf,
                    log_path=log_path,
                    stdout=result["stdout"],
                    stderr=result["stderr"],
                )

            return CompilationResult(
                success=False,
                output_path=None,
                log_path=log_path,
                stdout=result["stdout"],
                stderr=result["stderr"],
            )

    def compile_directory(
        self,
        working_dir: Path,
        main_file: str,
        output_pdf: Path,
        log_path: Optional[Path] = None,
    ) -> CompilationResult:
        """
        Compile from an existing LaTeX directory.

        Args:
            working_dir: Directory containing the LaTeX sources.
            main_file: Entry tex filename.
            output_pdf: PDF output path after compilation.
            log_path: Optional log file.
        """
        result = self._run_pdflatex_pipeline(working_dir, main_file, log_path)

        pdf_src = working_dir / f"{Path(main_file).stem}.pdf"
        if pdf_src.exists():
            output_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf_src, output_pdf)
            return CompilationResult(
                success=True,
                output_path=output_pdf,
                log_path=log_path,
                stdout=result["stdout"],
                stderr=result["stderr"],
            )

        return CompilationResult(
            success=False,
            output_path=None,
            log_path=log_path,
            stdout=result["stdout"],
            stderr=result["stderr"],
        )

    def _run_pdflatex_pipeline(
        self,
        working_dir: Path,
        main_file: str,
        log_path: Optional[Path],
    ) -> dict:
        """Run the pdflatex → bibtex → pdflatex ×2 pipeline."""
        commands = [
            ["pdflatex", "-interaction=nonstopmode", main_file],
            ["bibtex", Path(main_file).stem],
            ["pdflatex", "-interaction=nonstopmode", main_file],
            ["pdflatex", "-interaction=nonstopmode", main_file],
        ]

        stdout_all = []
        stderr_all = []

        for cmd in commands:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=working_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',  # Fix Windows GBK decode error
                    errors='replace',   # Replace invalid UTF-8 chars instead of crashing
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout_all.append(exc.stdout or "")
                stderr_all.append(
                    f"[TIMEOUT] Command {' '.join(cmd)} after {self.timeout}s"
                )
                break

            stdout_all.append(proc.stdout)
            stderr_all.append(proc.stderr)

            if log_path:
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"\n--- Running: {' '.join(cmd)} ---\n")
                    log_file.write(proc.stdout)
                    log_file.write(proc.stderr)

        return {"stdout": "\n".join(stdout_all), "stderr": "\n".join(stderr_all)}

    def _resolve_template_dir(
        self,
        template_name: Optional[str],
        template_dir: Optional[Path],
    ) -> Optional[Path]:
        if template_dir:
            return template_dir
        if template_name:
            candidate = self.template_root / template_name
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Template '{template_name}' does not exist. Please create the corresponding directory under {self.template_root}."
                )
            return candidate
        return None

    def _load_template_info(self, template_dir: Path, template_name: str) -> TemplateInfo:
        config_path = template_dir / "template.json"
        main_file = "main.tex"
        description = ""

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                main_file = data.get("main_file", main_file)
                description = data.get("description", "")

        return TemplateInfo(name=template_name, path=template_dir, main_file=main_file, description=description)

    _TEMPLATE_SKIP_FILES = {"template.json", "references.bib"}

    def _copy_template_dir(self, template_dir: Path, destination: Path) -> None:
        for item in template_dir.iterdir():
            if item.name in self._TEMPLATE_SKIP_FILES:
                continue
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    @classmethod
    def list_available_templates(cls, template_root: Optional[Path] = None) -> dict:
        """List available template name -> path mappings."""
        root = template_root or DEFAULT_TEMPLATE_ROOT
        if not root.exists():
            return {}
        templates = {}
        for path in root.iterdir():
            if path.is_dir() and (path / "template.json").exists():
                templates[path.name] = path
        return templates


def package_latex_project(
    latex_content: str,
    bibtex_content: str,
    template_name: str,
    output_dir: Path,
    figure_dir: Optional[Path] = None,
    template_root: Optional[Path] = None
) -> Optional[Path]:
    """
    Package a complete LaTeX project into a single folder ready to upload to Overleaf.

    Includes:
    - main.tex (full LaTeX content)
    - references.bib (bibliography database)
    - All template files (.cls, .sty, .bst, etc.)
    - Figure folder (if present)

    Args:
        latex_content: Full LaTeX document content.
        bibtex_content: BibTeX references content.
        template_name: Template name to use (e.g., "elsevier").
        output_dir: Output directory path.
        figure_dir: Figure folder path (optional).
        template_root: Template root directory (optional; defaults to DEFAULT_TEMPLATE_ROOT).

    Returns:
        The packaged directory path, or None on failure.
    """
    try:
        # Create the output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating LaTeX package in: {output_dir}")

        # 1. Write the main LaTeX file
        main_tex = output_dir / "main.tex"
        main_tex.write_text(latex_content, encoding="utf-8")
        logger.info(f"  ✓ Created main.tex")
        
        # 2. Write the BibTeX file
        if bibtex_content and bibtex_content.strip():
            references_bib = output_dir / "references.bib"
            references_bib.write_text(
                deduplicate_bibtex_entries(bibtex_content),
                encoding="utf-8",
            )
            logger.info(f"  ✓ Created references.bib")
        else:
            logger.warning("  ⚠ No BibTeX content provided, skipping references.bib")
        
        # 3. Copy template files
        root = template_root or DEFAULT_TEMPLATE_ROOT
        template_path = root / template_name
        
        if template_path.exists():
            # Copy all template files (.cls, .sty, .bst, etc.)
            template_files = [
                f for f in template_path.iterdir() 
                if f.is_file() and f.suffix in ['.cls', '.sty', '.bst', '.clo']
            ]
            
            for template_file in template_files:
                dest = output_dir / template_file.name
                shutil.copy2(template_file, dest)
                logger.info(f"  ✓ Copied {template_file.name}")
            
            # Copy the thumbnails folder (if it exists)
            thumbnails_src = template_path / "thumbnails"
            if thumbnails_src.exists() and thumbnails_src.is_dir():
                thumbnails_dest = output_dir / "thumbnails"
                shutil.copytree(thumbnails_src, thumbnails_dest, dirs_exist_ok=True)
                logger.info(f"  ✓ Copied thumbnails folder")
        else:
            logger.warning(f"  ⚠ Template path not found: {template_path}")
        
        # 4. Copy the figure folder
        if figure_dir and figure_dir.exists():
            figures_dest = output_dir / "figures"
            
            # Copy all figure files
            figure_files = list(figure_dir.rglob("*.png")) + \
                          list(figure_dir.rglob("*.jpg")) + \
                          list(figure_dir.rglob("*.jpeg")) + \
                          list(figure_dir.rglob("*.pdf")) + \
                          list(figure_dir.rglob("*.eps"))
            
            if figure_files:
                figures_dest.mkdir(exist_ok=True)
                for fig_file in figure_files:
                    # Preserve relative path structure
                    rel_path = fig_file.relative_to(figure_dir)
                    dest_file = figures_dest / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fig_file, dest_file)
                
                logger.info(f"  ✓ Copied {len(figure_files)} figure files")
        
        # 5. Create the README file
        readme_content = f"""# LaTeX Project Package

This folder contains a complete LaTeX project ready for compilation.

## Files:
- main.tex: Main LaTeX document
- references.bib: Bibliography database
- *.cls, *.sty, *.bst: Template files (do not modify)
- figures/: Image files (if any)

## How to compile:

### Option 1: Upload to Overleaf
1. Compress this entire folder into a .zip file
2. Go to Overleaf (https://www.overleaf.com)
3. Click "New Project" → "Upload Project"
4. Upload the .zip file
5. Set the compiler to "pdfLaTeX"
6. Click "Recompile"

### Option 2: Local compilation
Run the following commands:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Template: {template_name}
Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        readme_file = output_dir / "README.txt"
        readme_file.write_text(readme_content, encoding="utf-8")
        logger.info(f"  ✓ Created README.txt")
        
        logger.info(f"✓ LaTeX project packaged successfully")
        return output_dir
        
    except Exception as e:
        logger.error(f"Failed to package LaTeX project: {e}")
        return None
