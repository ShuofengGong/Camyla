# LaTeX Template Usage

This directory stores reusable LaTeX templates. Conventional structure:

```
latex_templates/
├── <template_name>/
│   ├── template.json   # config: entry file, description, etc.
│   ├── main.tex        # template entry file (filename is customizable)
│   └── ...             # other resources, e.g. .cls, .sty, bib
```

`template.json` supports the following fields:

- `main_file`: entry tex filename; defaults to `main.tex`.
- `description`: template description (optional).

To add a new template:

1. Create a new folder in this directory (e.g. `acl2025`).
2. Copy the official template files into that folder.
3. Create a `template.json` and specify the entry filename.
4. Invoke it via `LatexCompiler(template_name="acl2025", ...)`.

