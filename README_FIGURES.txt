LATEX FIGURES FOR PAPER - INSTRUCTIONS
========================================

Files Created:
--------------
1. preamble_additions.tex              - Add these packages to your document
2. figure_backend_comparison.tex       - Figure 1: Backend comparison bar plot
3. figure_encoding_heatmap.tex         - Figure 2: Encoding performance heatmap
4. figure_task_comparison.tex          - Figure 3: Classification vs Regression
5. section_results_replacement.tex     - Replacement text for Results sections
6. README.txt                          - This file


How to Use:
-----------

STEP 1: Update Document Preamble
---------------------------------
Open your main .tex file and add the content from 'preamble_additions.tex'
after your \documentclass line and other \usepackage commands.

It should look like:
  \documentclass{article}
  \usepackage[utf8]{inputenc}
  ...your existing packages...
  
  % ADD THESE:
  \usepackage{pgfplots}
  \pgfplotsset{compat=1.17}
  \usepgfplotslibrary{colormaps}
  \usepackage{subcaption}


STEP 2: Replace Results Section Text
-------------------------------------
In your main .tex file, find the section:
  \section{Experiments and Results}
  
Look for these subsections:
  \subsection{Backend Comparison...}
  \subsection{Encoding Comparison...}

REPLACE those two subsections with the content from:
  'section_results_replacement.tex'


STEP 3: Include the Figures
----------------------------
The section_results_replacement.tex file contains \input commands like:
  \input{figure_backend_comparison.tex}
  
Make sure ALL the figure_*.tex files are in the SAME DIRECTORY
as your main .tex file.

OR you can copy-paste the figure content directly instead of using \input.


What Tables to Remove/Keep:
----------------------------
REMOVE from main text:
  - Table with backend comparison (replaced by Figure 1)
  - Large encoding performance table (replaced by Figures 2 & 3)

KEEP in main text:
  - Table 1: Overall Performance Results (top-3 per model)
  - Dataset Statistics table
  - Encoding Properties table


Expected Final Structure:
-------------------------
Your Results section will have:
  4.1 Overall Performance [keep existing table]
  4.2 Backend Comparison [Figure 1 - bar plot]
  4.3 Encoding Comparison [Figures 2 & 3 - heatmap + side-by-side]
  4.4 Statistical Significance [keep existing text]


Compilation Notes:
------------------
- First compilation with pgfplots may be SLOW (30-60 seconds)
- Subsequent compilations will be faster
- If you get errors, make sure all packages are installed
- You may need to compile TWICE for figure references to work


Troubleshooting:
----------------
Error: "pgfplots.sty not found"
  Solution: Install pgfplots package for your LaTeX distribution
  
Error: "Undefined control sequence \begin{subfigure}"
  Solution: Make sure \usepackage{subcaption} is in preamble
  
Figures don't appear:
  Solution: Check that figure_*.tex files are in same directory as main .tex
  
Compilation very slow:
  Solution: Normal for pgfplots. Consider using externalization:
    \usepgfplotslibrary{external}
    \tikzexternalize


Alternative: Copy-Paste Method
-------------------------------
If \input doesn't work for you:

1. Open figure_backend_comparison.tex
2. Copy ALL the content (including \begin{figure*}...\end{figure*})
3. Paste it directly into your main .tex file where you want the figure
4. Repeat for other figures

This avoids any file path issues.


Questions?
----------
All figures use data from your statistical_summary CSV files.
Values are hardcoded in the .tex files.
If you update results, you'll need to update the coordinate values in the figures.
