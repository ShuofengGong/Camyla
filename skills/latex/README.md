# LatexCompiler Guide

## Overview

`LatexCompiler` is a general-purpose LaTeX compilation utility that supports:
- Compiling scientific papers with predefined templates (Elsevier, ICML, etc.)
- Compiling raw LaTeX strings
- Compiling existing LaTeX project directories
- Automatically running the `pdflatex -> bibtex -> pdflatex x2` pipeline

## Requirements

### System dependencies

Make sure the following LaTeX tools are installed:
- `pdflatex` - PDF compilation engine
- `bibtex` - bibliography processor

On Ubuntu/Debian:
```bash
sudo apt-get install texlive-latex-base texlive-bibtex-extra
```

On macOS:
```bash
brew install --cask mactex
```

### Python dependencies

```bash
pip install pathlib
```

(`pathlib` is built in since Python 3.4, so it is normally not needed.)

## Quick Start

### 1. Basic usage: compile a simple document

```python
from pathlib import Path
from func.latex_utils import LatexCompiler

# Create a compiler instance
compiler = LatexCompiler(timeout=60)

# Prepare LaTeX content
latex_content = r"""
\documentclass{article}
\begin{document}
\title{Hello World}
\author{Your Name}
\maketitle
\section{Introduction}
This is a test document.
\end{document}
"""

# Compile to PDF
output_pdf = Path("output.pdf")
result = compiler.compile_content(
    latex_content=latex_content,
    output_pdf=output_pdf
)

if result.success:
    print(f"✓ PDF generated: {result.output_path}")
else:
    print(f"✗ Compilation failed: {result.stderr}")
```

### 2. Compile with a template

#### 2.1 Using a predefined template (recommended)

```python
from pathlib import Path
from func.latex_utils import LatexCompiler

compiler = LatexCompiler()

# Inspect available templates
templates = LatexCompiler.list_available_templates()
print("Available templates:", list(templates.keys()))

# Use the Elsevier template
latex_content = r"""
\documentclass[a4paper,fleqn]{cas-sc}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage[authoryear,longnamesfirst]{natbib}

\begin{document}
\title [mode = title]{My Research Paper}
\author{Author Name}
\affiliation{organization={University Name}}

\begin{abstract}
This is the abstract of my paper.
\end{abstract}

\begin{keywords}
keyword1 \sep keyword2 \sep keyword3
\end{keywords}

\maketitle

\section{Introduction}
Your content here.

\bibliographystyle{cas-model2-names}
\bibliography{references}
\end{document}
"""

result = compiler.compile_content(
    latex_content=latex_content,
    output_pdf=Path("paper.pdf"),
    template_name="elsevier"  # Use the elsevier template
)
```

#### 2.2 Using a custom template directory

```python
from pathlib import Path
from func.latex_utils import LatexCompiler

compiler = LatexCompiler()

# Point at a custom template directory
custom_template = Path("/path/to/my/template")

result = compiler.compile_content(
    latex_content=latex_content,
    output_pdf=Path("output.pdf"),
    template_dir=custom_template  # Directly specify the template directory
)
```

### 3. Compile an existing directory

If you already have a complete LaTeX project directory:

```python
from pathlib import Path
from func.latex_utils import LatexCompiler

compiler = LatexCompiler()

# Compile the project directory
result = compiler.compile_directory(
    working_dir=Path("/path/to/latex/project"),
    main_file="main.tex",  # entry filename
    output_pdf=Path("output.pdf"),
    log_path=Path("compile.log")  # optional: save the compilation log
)

if result.success:
    print(f"✓ Compilation succeeded: {result.output_path}")
    print(f"  Log file: {result.log_path}")
```

### 4. Add figures and assets

```python
from pathlib import Path
from func.latex_utils import LatexCompiler

compiler = LatexCompiler()

latex_content = r"""
\documentclass{article}
\usepackage{graphicx}
\begin{document}
\title{Document with Figures}
\maketitle

\section{Figures}
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{assets/figure1.png}
    \caption{My Figure}
\end{figure}
\end{document}
"""

# Specify the resource files to copy
assets = [
    Path("figures/figure1.png"),
    Path("data/table.csv")
]

result = compiler.compile_content(
    latex_content=latex_content,
    output_pdf=Path("output.pdf"),
    assets=assets  # Assets are copied to the assets/ subdirectory of the temp dir
)
```

## Template System

### Template directory layout

Each template should follow this structure:

```
latex_templates/
├── template_name/
│   ├── template.json      # Template configuration (required)
│   ├── main.tex           # Main file (or whatever is listed in template.json)
│   ├── references.bib     # Bibliography (optional)
│   ├── *.sty              # Style files (optional)
│   ├── *.cls              # Document class files (optional)
│   └── ...                # Additional resources
```

### `template.json` format

```json
{
    "main_file": "template.tex",
    "description": "Elsevier CAS single-column template"
}
```

- `main_file`: main LaTeX filename (relative to the template directory)
- `description`: template description (optional)

### Adding a new template

1. Create a new folder under `latex_templates/`, e.g. `icml/`.
2. Copy the template files into that folder.
3. Create a `template.json` configuration file.
4. Call `LatexCompiler.list_available_templates()` to verify the template is recognized.

Example:

```bash
mkdir -p func/latex_templates/icml
cp -r /path/to/icml/template/* func/latex_templates/icml/
# Create template.json
echo '{"main_file": "main.tex", "description": "ICML template"}' > func/latex_templates/icml/template.json
```

## API Reference

### `LatexCompiler`

#### Constructor

```python
LatexCompiler(
    timeout: int = 60,
    template_root: Optional[Path] = None
)
```

- `timeout`: compilation timeout in seconds (default 60)
- `template_root`: template root directory (defaults to `func/latex_templates/`)

#### Methods

##### `compile_content()`

Compile a LaTeX string.

```python
def compile_content(
    self,
    latex_content: str,
    output_pdf: Path,
    template_dir: Optional[Path] = None,
    template_name: Optional[str] = None,
    assets: Optional[Iterable[Path]] = None,
    log_path: Optional[Path] = None,
) -> CompilationResult
```

**Parameters:**
- `latex_content`: LaTeX source code string
- `output_pdf`: output PDF path
- `template_dir`: custom template directory (mutually exclusive with `template_name`)
- `template_name`: predefined template name (e.g. "elsevier")
- `assets`: list of resource files to copy
- `log_path`: optional compilation log path

**Returns:** `CompilationResult` object

##### `compile_directory()`

Compile an existing LaTeX project directory.

```python
def compile_directory(
    self,
    working_dir: Path,
    main_file: str,
    output_pdf: Path,
    log_path: Optional[Path] = None,
) -> CompilationResult
```

**Parameters:**
- `working_dir`: LaTeX project directory
- `main_file`: main filename (e.g. "main.tex")
- `output_pdf`: output PDF path
- `log_path`: optional compilation log path

**Returns:** `CompilationResult` object

##### `list_available_templates()` (class method)

List all available templates.

```python
@classmethod
def list_available_templates(
    cls,
    template_root: Optional[Path] = None
) -> dict
```

**Returns:** dictionary mapping template names to template directories.

### `CompilationResult`

Compilation result data class.

```python
@dataclass
class CompilationResult:
    success: bool                # Whether compilation succeeded
    output_path: Optional[Path]  # PDF output path (when successful)
    log_path: Optional[Path]     # Log file path
    stdout: str                  # Standard output
    stderr: str                  # Standard error
```

## Error Handling

### Common errors and solutions

#### 1. Template not found

**Error:** `Template not found: xxx`

**Solution:**
- Check the template name
- Call `LatexCompiler.list_available_templates()` to see available templates
- Verify that `template.json` exists in the template directory

#### 2. Compilation timeout

**Error:** `Compilation timed out`

**Solution:**
- Increase the `timeout` argument
- Check the LaTeX code for infinite loops or expensive computations

#### 3. Missing LaTeX package

**Error:** `LaTeX Error: File 'xxx.sty' not found`

**Solution:**
- Install the missing LaTeX package
- On Ubuntu/Debian: `sudo apt-get install texlive-latex-extra`
- Check whether the template contains the required `.sty` files

#### 4. Bibliography errors

**Error:** BibTeX-related errors

**Solution:**
- Ensure `references.bib` exists and is well-formed
- Check that the `.bst` style file is present
- Make sure the filename in `\bibliography{references}` matches

## Testing

Run the test suite to verify functionality:

```bash
conda activate py310
python func/test_latex_utils.py
```

The tests cover:
1. Listing available templates
2. Compiling with the Elsevier template
3. Compiling a raw string
4. Compiling an existing directory
5. Error-handling validation

## Example Projects

See the test cases in `func/test_latex_utils.py` for full examples.

## Notes

1. **Temp directory**: `compile_content()` compiles inside a temporary directory which is cleaned up afterwards.
2. **File paths**: when referencing resources in LaTeX, pay attention to paths relative to the main file.
3. **Encoding**: make sure LaTeX files use UTF-8 encoding.
4. **Timeouts**: complex documents may need more time — tune `timeout` accordingly.
5. **Log file**: we recommend saving `log_path` to help debug compilation issues.

## License

This tool is extracted from the camyla project and inherits its license.

