# LaTeX Templates

This directory stores reusable LaTeX templates. Expected layout:

```
latex_templates/
├── <template_name>/
│   ├── template.json   # Configures entry file, description, etc.
│   ├── main.tex        # Template entry point (filename can be customized)
│   └── ...             # Additional resources such as .cls, .sty, bib
```

Fields supported by `template.json`:

- `main_file`: name of the entry `.tex` file (default `main.tex`)
- `description`: short template description (optional)

To add a new template:

1. Create a new folder under this directory (e.g. `acl2025`).
2. Copy the official template files into that folder.
3. Create `template.json` and specify the entry filename.
4. Call it via `LatexCompiler(template_name="acl2025", ...)`.

