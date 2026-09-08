"""Path resolution that refuses to escape a configured root.

A suite name arriving from a CLI argument or an HTTP request is untrusted
input. Resolving it under a root without this check is how ``../../etc/passwd``
becomes a readable file.

:class:`PathEscapeError` descends from :class:`ValueError` rather than from the
project's own ``LLMEvalError``: ``utils`` sits below ``models`` in the layered
dependency contract, so it cannot import the exception hierarchy. Callers in
``datasets`` and ``services`` translate it into the domain error their own
surface promises.
"""

from pathlib import Path


class PathEscapeError(ValueError):
    """A path resolved outside the root it was required to stay under."""


def resolve_under_root(root: Path | None, candidate: Path | str) -> Path:
    """Resolve `candidate`, requiring the result to stay under `root`.

    With ``root=None`` the candidate is resolved and returned unchecked: no
    root was configured, so there is no boundary to enforce and pretending
    otherwise would just move the decision somewhere less visible.

    Raises:
        PathEscapeError: when the resolved path is not inside `root`.
    """
    resolved = Path(candidate).expanduser().resolve()
    if root is None:
        return resolved
    resolved_root = Path(root).expanduser().resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        msg = f"{resolved} is outside the configured root {resolved_root}"
        raise PathEscapeError(msg)
    return resolved
