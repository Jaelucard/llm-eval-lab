"""Benchmark suite contract: authored form, resolved form and persisted snapshot.

Three shapes, deliberately distinct. :class:`BenchmarkSuite` is what an author
writes and what the loader validates. :class:`ResolvedSuite` is that suite with
defaults merged into every case; it is what executes and what is hashed.
:class:`BenchmarkSnapshot` is the content-addressed record persisted alongside
a run, so the run stays interpretable after its source file changes or is
deleted.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_eval_lab.models.common import ChatMessage, GenerationParams, JSONValue, UtcDatetime

CASE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
SUITE_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"


class EvaluatorSpec(BaseModel):
    """One evaluator to apply to a case, and how its outcome is treated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    id: str | None = None
    weight: float = Field(default=1.0, ge=0.0)
    required: bool = True
    on_error: Literal["error", "fail", "skip"] = "error"
    params: dict[str, JSONValue] = Field(default_factory=dict)


class BenchmarkCase(BaseModel):
    """One authored test case.

    Exactly one of ``input`` and ``messages`` is supplied: ``input`` is the
    single-turn shorthand, ``messages`` the multi-turn form.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=CASE_ID_PATTERN)
    input: str | None = None
    messages: tuple[ChatMessage, ...] | None = None
    system: str | None = None
    expected: JSONValue = None
    category: str | None = None
    tags: tuple[str, ...] = ()
    weight: float = Field(default=1.0, ge=0.0)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)
    params: GenerationParams | None = None
    evaluators: tuple[EvaluatorSpec, ...] = ()

    @model_validator(mode="after")
    def _exactly_one_prompt_form(self) -> "BenchmarkCase":
        """Require exactly one of `input` and `messages`."""
        supplied = (self.input is not None, self.messages is not None)
        if not any(supplied):
            msg = f"case {self.id!r} must supply exactly one of 'input' or 'messages', got neither"
            raise ValueError(msg)
        if all(supplied):
            msg = f"case {self.id!r} must supply exactly one of 'input' or 'messages', got both"
            raise ValueError(msg)
        return self


class SuiteDefaults(BaseModel):
    """Values every case in the suite inherits unless it overrides them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str | None = None
    params: GenerationParams = GenerationParams()
    evaluators: tuple[EvaluatorSpec, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class BenchmarkSuite(BaseModel):
    """A suite exactly as authored in YAML or JSON."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    name: str = Field(pattern=SUITE_NAME_PATTERN)
    version: str = "1"
    description: str | None = None
    defaults: SuiteDefaults = SuiteDefaults()
    cases: tuple[BenchmarkCase, ...] = Field(min_length=1)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _case_ids_are_unique(self) -> "BenchmarkSuite":
        """Reject duplicate case ids, naming every id that repeats."""
        seen: set[str] = set()
        duplicates: list[str] = []
        for case in self.cases:
            if case.id in seen and case.id not in duplicates:
                duplicates.append(case.id)
            seen.add(case.id)
        if duplicates:
            msg = f"duplicate case ids in suite {self.name!r}: {', '.join(sorted(duplicates))}"
            raise ValueError(msg)
        return self


class ResolvedCase(BaseModel):
    """A case with suite defaults merged in.

    This is what executes and what is hashed; nothing downstream re-reads the
    authored form.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    case_hash: str
    messages: tuple[ChatMessage, ...]
    system: str | None
    expected: JSONValue
    category: str | None
    tags: tuple[str, ...]
    weight: float
    metadata: dict[str, JSONValue]
    params: GenerationParams
    evaluators: tuple[EvaluatorSpec, ...]


class ResolvedSuite(BaseModel):
    """A whole suite in resolved, executable, content-addressed form."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    suite_hash: str
    cases: tuple[ResolvedCase, ...]
    metadata: dict[str, JSONValue]


class BenchmarkSnapshot(BaseModel):
    """The persisted, content-addressed record of a resolved suite.

    Deduplicated by digest, so a run stays interpretable after its source file
    changes or disappears.
    """

    model_config = ConfigDict(frozen=True)

    suite_hash: str
    name: str
    version: str
    n_cases: int
    created_at: UtcDatetime
    source_path: str | None
    body: dict[str, JSONValue]
