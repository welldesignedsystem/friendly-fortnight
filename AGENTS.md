# friendly-fortnight

Anomaly detection learning repo — Jupyter notebooks demonstrating statistical, distance-based, ensemble, and time-series outlier detection methods.

## Quick start

```bash
uv sync          # install deps (Python >=3.12, uv required)
```

## Repo layout

- `blog/*.ipynb` — Jupyter notebooks, one per anomaly detection topic
- `blog/custom_markdown.tpl` — nbconvert template wrapping outputs in ` ```text ` blocks
- `blog/blog.md` — compiled blog post (hand-edited content)
- `img/` — generated images (output of nbconvert)
- `pyproject.toml` — single package, no monorepo structure
- `welldesignedsystem.github.io/` — empty; the actual site is a sibling at `../welldesignedsystem.github.io/`

## Key workflow

Convert a notebook to markdown for the blog:

```bash
cd blog
jupyter nbconvert histogram.ipynb --to markdown \
  --template=custom_markdown.tpl \
  --NbConvertApp.output_files_dir='../img'
```

- Always run from `blog/` so the `img/` relative path resolves correctly.
- The custom template strips ANSI from execution outputs and renders them as ` ```text ` blocks (not the default ` ``` ` with syntax highlighting).

## Caveats

- No tests, no linter, no type checker, no CI.
- All deps managed by `uv` — do not use pip directly.
- Use `.python-version` (3.12) via `uv python pin` if needed.
