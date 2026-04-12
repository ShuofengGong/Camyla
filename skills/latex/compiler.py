import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


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


DEFAULT_TEMPLATE_ROOT = Path(__file__).parent / "templates"


class LatexCompiler:
    """General-purpose LaTeX compiler with template support."""

    def __init__(self, timeout: int = 60, template_root: Optional[Path] = None):
        self.timeout = timeout
        self.template_root = template_root or DEFAULT_TEMPLATE_ROOT
        self.template_root.mkdir(parents=True, exist_ok=True)

    def compile_content(
        self,
        latex_content: str,
        output_pdf: Path,
        template_dir: Optional[Path] = None,
        template_name: Optional[str] = None,
        assets: Optional[Iterable[Path]] = None,
        log_path: Optional[Path] = None,
    ) -> CompilationResult:
        """
        Compile a LaTeX string into a PDF.

        Args:
            latex_content: full LaTeX text
            output_pdf: target PDF path
            template_dir: optional template directory; all files are copied into the build dir
            template_name: template name loaded from template_root
            assets: additional resources (images, etc.) to copy into the temp directory
            log_path: optional log file receiving pdflatex/bibtex stdout/stderr
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
        Compile an existing LaTeX directory.

        Args:
            working_dir: directory containing the LaTeX sources
            main_file: name of the entry .tex file
            output_pdf: output PDF path after compilation
            log_path: optional log file
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
        """Run the pdflatex -> bibtex -> pdflatex x2 pipeline."""
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
                    f"Template '{template_name}' not found. Create the corresponding directory under {self.template_root}."
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

    def _copy_template_dir(self, template_dir: Path, destination: Path) -> None:
        for item in template_dir.iterdir():
            if item.name == "template.json":
                continue
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    @classmethod
    def list_available_templates(cls, template_root: Optional[Path] = None) -> dict:
        """List available templates as a {name: path} mapping."""
        root = template_root or DEFAULT_TEMPLATE_ROOT
        if not root.exists():
            return {}
        templates = {}
        for path in root.iterdir():
            if path.is_dir() and (path / "template.json").exists():
                templates[path.name] = path
        return templates

