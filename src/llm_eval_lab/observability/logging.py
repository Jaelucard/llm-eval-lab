"""structlog configuration, secret scrubbing and prompt gating.

Two processors here are load-bearing rather than cosmetic.

:func:`scrub_secrets` walks every event dictionary and replaces the value of
any key whose name looks like a credential, using the same rule the persistence
mappers apply (``llm_eval_lab.redaction``). It runs on every event, not only on events a
developer remembered to sanitize, because the failure mode it guards against is
precisely the log line nobody audited.

:func:`make_prompt_filter` drops prompt and output text unless the operator
explicitly enabled ``log_prompts``. Prompts are user data and frequently contain
the very material a benchmark is testing against; they are opt-in, at DEBUG
only, and off by default.
"""

import logging
import sys
from typing import Literal

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

from llm_eval_lab.redaction import (
    MAX_SCRUB_DEPTH,
    REDACTED,
    SECRET_KEY_PATTERN,
    is_secret_key,
    scrub_mapping,
)

PROMPT_KEYS: frozenset[str] = frozenset(
    {"prompt", "prompts", "messages", "system", "output_text", "response_text", "raw"}
)
"""Event keys carrying model input or output text, gated behind `log_prompts`."""


def scrub_secrets(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Redact every secret-shaped key in the event."""
    return scrub_mapping(dict(event_dict))


def make_prompt_filter(*, log_prompts: bool) -> Processor:
    """Build a processor that drops prompt and output text unless enabled."""

    def _filter(
        _logger: WrappedLogger,
        _method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        if log_prompts:
            return event_dict
        for key in PROMPT_KEYS:
            event_dict.pop(key, None)
        return event_dict

    return _filter


def configure_logging(
    *,
    level: str = "INFO",
    fmt: Literal["console", "json"] = "console",
    log_prompts: bool = False,
    colors: bool = True,
) -> None:
    """Configure structlog and the stdlib root logger for this process.

    Everything is written to **stderr**. stdout belongs to ``--json`` output,
    and a log line interleaved into it would break every caller that pipes the
    CLI into ``jq``.
    """
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level, force=True)

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=colors)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            make_prompt_filter(log_prompts=log_prompts),
            scrub_secrets,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level]),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound logger for `name`."""
    logger: structlog.BoundLogger = structlog.get_logger(name)
    return logger


__all__ = [
    "MAX_SCRUB_DEPTH",
    "PROMPT_KEYS",
    "REDACTED",
    "SECRET_KEY_PATTERN",
    "configure_logging",
    "get_logger",
    "is_secret_key",
    "make_prompt_filter",
    "scrub_mapping",
    "scrub_secrets",
]
