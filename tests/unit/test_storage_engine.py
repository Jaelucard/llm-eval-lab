"""`create_engine` must never leak a raw database URL or its credentials.

A malformed URL that carries a password is the worst case for
``StorageError``: the string that triggered the failure IS the secret. See
`llm_eval_lab.redaction` for the shared scrubbing helpers this exercises, and
`tests/integration/test_secret_leak.py` for the same canary discipline applied
at the CLI and logging boundaries.
"""

import traceback

import pytest

from llm_eval_lab.models import StorageError
from llm_eval_lab.storage.engine import create_engine

CANARY_USER = "canary-user-2b9d"
CANARY_PASSWORD = "canary-pw-7f3a9c1e"  # noqa: S105 - a planted canary, not a credential
MALFORMED_URL = f"postgresql+asyncpg://{CANARY_USER}:{CANARY_PASSWORD}@dbhost:notaport/evals"
"""A malformed port fails URL parsing before the `asyncpg` driver is ever
imported, so this raises the same way whether or not the optional Postgres
extra is installed - unlike an unreachable-but-well-formed URL, which would
need a real network attempt."""


def test_create_engine_raises_before_touching_the_url() -> None:
    """Sanity check the fixture: this URL really does fail to open."""
    with pytest.raises((StorageError,)):
        create_engine(MALFORMED_URL)


def test_a_malformed_url_never_reaches_the_storage_error_text() -> None:
    with pytest.raises(StorageError) as excinfo:
        create_engine(MALFORMED_URL)
    exc = excinfo.value
    text = str(exc)

    assert CANARY_PASSWORD not in text
    assert CANARY_USER not in text
    assert MALFORMED_URL not in text

    # The message stays useful: which host, and what kind of failure.
    assert "dbhost" in text
    assert "ValueError" in text


def test_the_original_exception_is_not_chained_as_a_cause() -> None:
    """`exc.args` on the underlying ValueError does not carry the raw URL here,
    but the chain is still severed on principle: the underlying exception is
    the ONE place a future SQLAlchemy version could start putting it, and nothing
    downstream should depend on `__cause__` being scrubbed rather than absent.
    """
    with pytest.raises(StorageError) as excinfo:
        create_engine(MALFORMED_URL)
    exc = excinfo.value

    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True


def test_a_formatted_traceback_never_carries_the_canary() -> None:
    try:
        create_engine(MALFORMED_URL)
    except StorageError as exc:
        rendered = "".join(traceback.format_exception(exc))
    else:
        pytest.fail("create_engine did not raise for a malformed URL")

    assert CANARY_PASSWORD not in rendered
    assert CANARY_USER not in rendered
    assert MALFORMED_URL not in rendered


def test_a_password_holding_a_raw_slash_never_reaches_the_storage_error_text() -> None:
    """A raw `/` in the password splits the URL in the wrong place for the
    standard-library parser, so a userinfo-only redaction would print the tail
    of the password. The redactor fails closed instead, and this guards that
    at the engine boundary rather than only at the redactor's own tests."""
    url = f"postgresql+asyncpg://{CANARY_USER}:p/w{CANARY_PASSWORD}@dbhost:notaport/evals"
    with pytest.raises(StorageError) as excinfo:
        create_engine(url)
    exc = excinfo.value
    text = str(exc) + "".join(traceback.format_exception(exc)) + repr(exc.args)

    assert CANARY_PASSWORD not in text
    assert CANARY_USER not in text
    assert "ValueError" in str(exc)


def test_a_bare_token_holding_a_raw_slash_never_reaches_the_storage_error_text() -> None:
    """A token used as the username, with no password and a raw `/` inside it,
    leaves no colon in the truncated authority; the redactor must fail closed
    on the authority's mere existence, not on its shape."""
    url = f"nosuch+x://tok/en{CANARY_PASSWORD}@dbhost/evals"
    with pytest.raises(StorageError) as excinfo:
        create_engine(url)
    exc = excinfo.value
    text = str(exc) + "".join(traceback.format_exception(exc)) + repr(exc.args)

    assert CANARY_PASSWORD not in text
    assert CANARY_USER not in text
    assert "NoSuchModuleError" in str(exc)
