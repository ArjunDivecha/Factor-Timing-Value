---
trigger: always_on
description: 
globs: 
---

# Your rule content
# Jupyter Notebook Editing Instructions

When editing Jupyter notebooks:

1. NEVER try to edit notebooks directly with edit_file - it will likely fail due to complex JSON structure

2. Instead, follow this process:
   a. Install jupytext if needed: pip install jupytext
   b. Convert notebook to Python: jupytext --to py "Notebook.ipynb" --output "Notebook.py"
   c. Edit the Python file with edit_file
   d. Convert back to notebook: jupytext --to notebook "Notebook.py"
   
3. Alternatively use nbconvert:
   a. Convert notebook: jupyter nbconvert --to python "Notebook.ipynb" --output "temp.py"
   b. Edit the Python file
   c. Convert back: jupyter nbconvert --to notebook "temp.py" --output "Notebook.ipynb"
   
4. After conversion, check that the notebook structure is preserved correctly

5. Note: Converting will remove cell outputs - the notebook will need to be re-run to generate outputs

Always prefer jupytext when available as it preserves more notebook metadata.
- You can @ files here
- You can use markdown but dont have to
