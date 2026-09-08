"""HTTP routers: one module per area of the surface.

A handler here validates its inputs, calls exactly one service, and maps the
result onto a DTO. It owns no business logic and touches no repository, no
registry and no loader; the layered import contract in `pyproject.toml` makes
that mechanical rather than aspirational.
"""
