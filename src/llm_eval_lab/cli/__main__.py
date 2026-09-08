"""Entry point for ``python -m llm_eval_lab.cli``.

The installed console script points at ``llm_eval_lab.cli.main:app``. This
module exists so the same CLI can be invoked without the script on PATH, which
is what the end-to-end tests do when they need a real, signal-receiving process.
"""

from llm_eval_lab.cli.main import app

if __name__ == "__main__":
    app()
