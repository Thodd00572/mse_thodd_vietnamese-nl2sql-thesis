# Thesis Document

This directory contains the main thesis document and related files.

## Structure
- `main.tex` - Main LaTeX document
- `chapters/` - Individual chapter files
- `bibliography.bib` - References and citations
- `style/` - Custom style files and templates

## LaTeX Setup
Recommended LaTeX distribution: TeX Live or MiKTeX
Required packages will be listed in the main document.

## Compilation
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk for automated compilation:
```bash
latexmk -pdf main.tex
```
