# Python virtual environments

A Python virtual environment isolates a project's installed packages from other Python projects. Create one with `python -m venv .venv`. On Windows PowerShell activate it with `.venv\Scripts\Activate.ps1`; on macOS or Linux use `source .venv/bin/activate`.

Install dependencies using `python -m pip install -r requirements.txt`. If a module cannot be imported, verify that the active interpreter is the same interpreter where the package was installed. Use `python -m pip show PACKAGE` to inspect an installation and `python -c "import sys; print(sys.executable)"` to inspect the interpreter path.

Pin or constrain dependencies for reproducibility, and avoid committing virtual environment folders or secret files to version control.
