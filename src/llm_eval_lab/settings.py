"""Process configuration, read from the environment exactly once, here.

This module is the **only** place in the package that touches ``os.environ``
(decision D8). Every other layer receives the values it needs as arguments, so
a test configures behaviour by constructing objects rather than by mutating
global state, and an evaluator or provider can never quietly depend on an
environment variable nobody declared.

Credentials are deliberately absent from this model. ``Settings`` carries only
non-secret operational configuration; a vendor credential is fetched on demand,
by name, through :func:`read_credential` - the one function in the package that
reads a credential out of the environment, and the implementation site a
``CredentialResolver`` is built on. A credential is never a field on anything
that gets persisted, logged or returned over the API. The single exception is
:attr:`Settings.api_token`, which is a :class:`~pydantic.SecretStr` guarding
the API's own bind, not a vendor credential.

A `.env` file is honoured. :func:`get_settings` loads it with ``override=False``
so a real environment variable always wins, and ``SettingsConfigDict`` reads the
same file for the ``LLM_EVAL_*`` fields. Both halves are needed: pydantic-settings
populates ``Settings`` fields only, so a vendor key in `.env` would otherwise never
reach :func:`read_credential`, and the repository ships a template for exactly that
workflow.

Every field is read from ``LLM_EVAL_<FIELD_NAME>``, upper-cased. Values that
are not simple scalars (``cors_origins``, ``plugins_allowed``) are parsed as
JSON, so ``LLM_EVAL_PLUGINS_ALLOWED='["acme_evals"]'``.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_eval_lab.models import EvaluatorSettings

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})

_DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "llm-eval-lab"
_DEFAULT_DEV_ORIGIN = "http://localhost:5173"


@lru_cache(maxsize=1)
def _dotenv_path() -> str:
    """Resolve the one `.env` path both halves of the dotenv story use, at first use.

    Resolved lazily and cached, rather than as a module-level constant computed
    at import time: importing this module should not itself walk the
    filesystem, and a constant resolved at import time can never be corrected
    afterward - there was no way to clear it, even in a test that legitimately
    needed a different working directory.

    Resolved with ``usecwd=True`` so the search starts at the working
    directory, which is where a user's `.env` sits. The default,
    ``usecwd=False``, walks up from this module's own file instead: identical
    in a source checkout, and wrong from an installed wheel, where it would
    search site-packages while pydantic-settings resolved a relative
    ``env_file`` against the working directory. Both :func:`_load_dotenv_once`
    and :func:`get_settings` call this SAME function, which is what stops the
    two disagreeing.

    Cached for the life of the process: this module does not support an
    in-process ``chdir()`` after the first resolution. A process is expected
    to establish its working directory once, before touching settings, and
    not change it afterward. Tests that need a different resolution call
    ``_dotenv_path.cache_clear()``.
    """
    return find_dotenv(usecwd=True) or ".env"


class Settings(BaseSettings):
    """Operational configuration for one process.

    Frozen, so a component that received a ``Settings`` cannot reconfigure the
    process behind its caller's back.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_EVAL_",
        extra="ignore",
        frozen=True,
        # A plain relative default, resolved by pydantic-settings itself
        # against the working directory at construction time - never the
        # eagerly-resolved, import-time absolute path `_dotenv_path()`
        # produces. `get_settings()` passes that resolved path explicitly, as
        # `_env_file`, which is what lets the two stay in step without this
        # class body computing anything at import time on its own.
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- storage ---------------------------------------------------------
    data_dir: Path = _DEFAULT_DATA_DIR
    """Root for generated state. Only the database lives here in v1."""

    database_url: str | None = None
    """SQLAlchemy async URL. ``None`` resolves to a SQLite file under `data_dir`."""

    # --- benchmark loading -----------------------------------------------
    suites_root: Path | None = None
    """Directory a suite name is resolved under. Paths escaping it are refused."""

    max_suite_bytes: int = Field(default=8_388_608, ge=1024)
    """Hard cap on a suite file, applied before parsing. 8 MiB."""

    # --- run defaults -----------------------------------------------------
    concurrency: int = Field(default=8, ge=1, le=256)
    """Default in-flight cases per run when the caller does not say."""

    judge_concurrency: int = Field(default=4, ge=1, le=64)
    """Default in-flight judge calls, held below `concurrency` on purpose."""

    case_timeout_s: float = Field(default=120.0, gt=0.0)
    """Default wall-clock budget for one case, retries included."""

    max_raw_bytes: int = Field(default=65536, ge=1024)
    """Cap on a stored vendor payload. Oversized payloads are replaced by a marker."""

    max_regex_match_s: float = Field(default=2.0, gt=0.0)
    """Bound on one regex evaluation, so a pathological pattern cannot hang a run."""

    max_output_chars: int = Field(default=8000, ge=1)
    """Candidate output is truncated to this before being shown to a judge."""

    # --- observability ----------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    """Threshold for the structured logger."""

    log_format: Literal["console", "json"] = "console"
    """Console renders for humans; json renders one object per line for machines."""

    log_prompts: bool = False
    """Log prompt and output text at DEBUG. Off by default: prompts are user data."""

    # --- api --------------------------------------------------------------
    api_host: str = "127.0.0.1"
    """Bind address. A non-loopback bind is refused unless `api_token` is set."""

    api_port: int = Field(default=8000, ge=1, le=65535)
    """Bind port."""

    api_token: SecretStr | None = None
    """Bearer token required to expose the API off loopback. Never logged.

    A set-but-empty value is normalised to ``None`` by
    :meth:`_normalise_empty_token`, because an empty token would authorise every
    request while making the bind check believe authentication was configured.
    """

    cors_origins: tuple[str, ...] = (_DEFAULT_DEV_ORIGIN,)
    """Allowed browser origins. Never `*`: this API can spend money."""

    max_request_bytes: int = Field(default=4_194_304, ge=1024)
    """Cap on a request body, applied before parsing. 4 MiB."""

    # --- plugins ----------------------------------------------------------
    plugins_enabled: bool = False
    """Load third-party evaluator entry points. Off by default: they execute code."""

    plugins_allowed: frozenset[str] = frozenset()
    """Entry-point names permitted when `plugins_enabled`. Empty means none."""

    @field_validator("api_token", mode="after")
    @classmethod
    def _normalise_empty_token(cls, value: SecretStr | None) -> SecretStr | None:
        """Collapse a set-but-empty token to ``None``, so "unset" is ONE state.

        An environment variable that is set but empty - the result of
        ``export LLM_EVAL_API_TOKEN=$UNSET_VAR``, of a `.env` line reading
        ``LLM_EVAL_API_TOKEN=``, or of a secrets mount that produced an empty
        file - parses to ``SecretStr('')``, which is not ``None``. Two controls
        then failed open at once: the non-loopback bind refusal accepted the bind
        because a token "was set", and the bearer gate compared the empty
        presented value against the empty configured one and matched, so a
        request with NO ``Authorization`` header at all was authorised.

        Normalising here means both controls see exactly one condition, "there is
        no token", and both then do the safe thing: the bind is refused unless it
        is loopback, and the gate is a no-op on an API nothing else can reach.
        Rejecting the value instead would turn a mis-set variable into a process
        that will not start, which is worse for the operator and no safer, since
        the refused bind already stops the dangerous configuration.
        """
        if value is None:
            return None
        return value if value.get_secret_value().strip() else None

    @field_validator("cors_origins")
    @classmethod
    def _reject_wildcard_origin(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse `*` as an allowed origin.

        `POST /api/runs` spends money, so a wildcard origin lets any page a
        browser visits start a run against the user's credentials. There is no
        configuration in which that is the right answer, so it is refused here
        rather than left as a default nobody revisits.
        """
        if "*" in value:
            msg = (
                "cors_origins must not contain '*': the API can start runs that spend "
                "money. List the exact origins that may call it."
            )
            raise ValueError(msg)
        return value

    @property
    def api_bind_is_loopback(self) -> bool:
        """Report whether the configured bind address is loopback-only."""
        return self.api_host in _LOOPBACK_HOSTS

    def resolved_database_url(self) -> str:
        """Return the database URL, deriving a per-user SQLite path when unset."""
        if self.database_url is not None:
            return self.database_url
        return f"sqlite+aiosqlite:///{self.data_dir / 'llm-eval-lab.db'}"

    def evaluator_settings(self) -> EvaluatorSettings:
        """Project the evaluator-facing slice of this configuration.

        Evaluators receive this through ``EvaluationContext`` rather than
        importing this module, which is what keeps them testable without an
        environment.
        """
        return EvaluatorSettings(
            judge_concurrency=self.judge_concurrency,
            max_output_chars=self.max_output_chars,
            max_regex_match_s=self.max_regex_match_s,
            max_raw_bytes=self.max_raw_bytes,
        )


@lru_cache(maxsize=1)
def _load_dotenv_once() -> None:
    """Load the `.env` file into the process environment, at most once.

    ``load_dotenv`` mutates ``os.environ`` process-wide and, with
    ``override=False``, never replaces a variable that is already set, so a real
    environment variable always beats the file. Cached so that repeated calls
    from :func:`read_credential` and :func:`get_settings` cost nothing and
    cannot produce two different views of the environment. Tests that need to
    reload call ``_load_dotenv_once.cache_clear()`` (and, if the working
    directory changed too, ``_dotenv_path.cache_clear()``).

    Uses :func:`_dotenv_path`, the same cached resolution
    :func:`get_settings` passes to :class:`Settings` as ``_env_file`` - one
    resolution, shared, rather than two independent calls to
    :func:`~dotenv.find_dotenv` that could in principle disagree.
    """
    load_dotenv(_dotenv_path(), override=False)


def read_credential(env_var: str) -> SecretStr | None:
    """Read one vendor credential from the environment. The only place that happens.

    Returns ``None`` when the variable is unset or empty, which is what lets a
    caller raise :class:`~llm_eval_lab.models.MissingCredentialError` naming the
    variable. The value is wrapped in :class:`~pydantic.SecretStr` immediately so
    it cannot reach a log line or a persisted model through an accidental repr,
    and it is never stored on :class:`Settings`: credentials are fetched by name,
    on demand, and are not configuration.

    Loads `.env` first, so a resolver built before anything touches
    :func:`get_settings` still sees a key the user put in that file. That
    ordering was the bug this guards against: the two entry points into this
    module must not disagree about whether `.env` has been read.
    """
    _load_dotenv_once()
    raw = os.environ.get(env_var)
    return SecretStr(raw) if raw else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, reading the environment once.

    Loads a `.env` file first, through the same :func:`_load_dotenv_once` that
    :func:`read_credential` uses, so both entry points see one environment.
    That load is what puts vendor credentials where :func:`read_credential`
    looks for them: pydantic-settings would populate ``Settings`` fields only,
    and a vendor key is not a settings field, so without this a key in `.env`
    could never reach a resolver.

    Cached deliberately: the environment is read at first use and not again, so
    two components cannot observe different configuration. Tests that need a
    different configuration construct :class:`Settings` directly, or call
    ``get_settings.cache_clear()``.

    Passes :func:`_dotenv_path`'s resolution explicitly as ``_env_file``,
    rather than relying on the class's own ``model_config`` default (a plain
    relative ``".env"``, resolved by pydantic-settings against the working
    directory): this is the one call site responsible for using the SAME
    resolved path :func:`_load_dotenv_once` already loaded into the
    environment, so the two cannot disagree about which file was read.
    """
    _load_dotenv_once()
    return Settings(_env_file=_dotenv_path())
