# Providers

`llm-eval-lab` reaches every model vendor through one `BaseProvider`
interface (`src/llm_eval_lab/providers/base.py`); the evaluation engine,
`runner`, `evaluators` and `services` never import a concrete vendor SDK.
Every provider integration is optional and lazily imported, so an unset
credential or an uninstalled extra only matters when you actually try to use
that provider.

**Record of record.** This document transcribes
[`docs/providers-sources.md`](providers-sources.md), which Lane A owns and
which records the exact documentation URLs, access dates, pinned SDK
versions and the specific facts confirmed against each. If the two ever
disagree, `docs/providers-sources.md` is the one that was actually checked
against the vendor — treat this page as a readable summary, that page as the
audit trail. Anthropic is documented here exactly like every other supported
model provider, as one vendor among five this product integrates with.

| Provider | Install extra | Default credential env var | SDK pinned |
|---|---|---|---|
| `openai` | `llm-eval-lab[openai]` | `OPENAI_API_KEY` | `openai==2.54.0` |
| `openai_compatible` | `llm-eval-lab[openai]` (same SDK, pointed at a different `base_url`) | none by default — pass `--api-key-env` naming your own variable if the endpoint requires auth | `openai==2.54.0` |
| `anthropic` | `llm-eval-lab[anthropic]` | `ANTHROPIC_API_KEY` | `anthropic==0.125.0` |
| `google` | `llm-eval-lab[google]` | `GOOGLE_API_KEY` | `google-genai==1.75.0` |
| `ollama` | none — reached over plain HTTP via the core `httpx` dependency | none required; supply a host with `--base-url` | n/a (no SDK) |

A `ProviderConfig` never carries a credential value — only the *name* of the
environment variable holding it (`api_key_env`), resolved on demand through
`read_credential()`. This is what makes a saved `RunConfig` safe to persist,
log and return over the API verbatim: `llm-eval config show` and `llm-eval
providers` can tell you exactly which variable a provider expects without
ever reading its value.

## OpenAI (`openai`)

Accessed 2026-09-07 against `openai==2.54.0`, from
<https://developers.openai.com/api/docs/guides/migrate-to-responses> and
<https://developers.openai.com/api/docs/api-reference/responses/object>.

- The adapter calls the **Responses API** (`client.responses.create`), not
  Chat Completions: OpenAI documents Responses as the recommended default
  for new projects and states Chat Completions receives no further feature
  development.
- `GenerationParams.stop` and `.seed` have **no equivalent** on
  `responses.create` in the pinned SDK version and are silently dropped for
  this provider — `responses.create`'s signature carries neither parameter.
- `instructions` carries the system prompt; `max_output_tokens` caps output;
  `text.format` is where JSON/structured output is requested, replacing Chat
  Completions' `response_format`.
- **Fields that commonly go missing / normalize:** `incomplete_details.reason`
  is a `Literal["max_output_tokens","content_filter"]` in the pinned SDK,
  while OpenAI's published reference also documents `max_messages` and
  `steered` — both map to an `UNKNOWN` finish reason rather than raising.
  Token usage is reported at `usage.input_tokens` / `.output_tokens` /
  `.total_tokens`, with `input_tokens_details.cached_tokens` and
  `output_tokens_details.reasoning_tokens` present only when the model
  supports them.
- `max_retries=0` is set on the client so the SDK's own retry loop (default
  2) never doubles up with this project's own bounded backoff in `runner`.

## OpenAI-compatible endpoints (`openai_compatible`)

Same SDK and access date (2026-09-07) as above, from
<https://raw.githubusercontent.com/openai/openai-python/main/README.md>'s
documented `base_url` override pattern.

- Reached with `AsyncOpenAI(base_url=...)`, calling **`chat.completions.create`**
  rather than `responses.create` — Chat Completions is the surface
  third-party and self-hosted servers actually implement; the Responses API
  is an OpenAI-hosted surface a compatible endpoint is not expected to
  provide.
- **No default credential env var.** Unlike the other four providers, the
  registry sets no default `credential_env` for `openai_compatible` — many
  self-hosted endpoints need no key at all. Pass `--api-key-env` naming your
  own variable when the specific endpoint you are pointing at does require
  one.
- **Fields that commonly go missing** on third-party endpoints, and which
  degrade to `None` rather than a fabricated value: the whole `usage`
  object; `usage.prompt_tokens_details.cached_tokens`;
  `usage.completion_tokens_details.reasoning_tokens`;
  `choices[].finish_reason` (mapped to `UNKNOWN` when absent); the response
  `id` (mapped to `provider_request_id=None`). The `model` a server echoes
  back is recorded separately from the model that was requested, because
  some servers return a different string than they were asked for.
- Has **no price table entries by design**: a third-party endpoint's price is
  set by whoever runs it, so runs against it record `priced=false` with a
  reason rather than a fabricated cost. Supply your own price table with
  `--prices` to cost them.

## Anthropic (`anthropic`)

Accessed 2026-09-07 against `anthropic==0.125.0`, from
<https://platform.claude.com/docs/en/api/messages> and
<https://platform.claude.com/docs/en/about-claude/models/overview>.

- Async client is `AsyncAnthropic`; the call path is `client.messages.create`.
- `max_tokens` is a **required** request parameter for this vendor — the
  adapter supplies a default whenever a benchmark case does not set one,
  since `GenerationParams.max_output_tokens` is optional in the neutral
  contract but Anthropic's API rejects a request without it.
- `system` is a top-level request parameter, not a message with a `system`
  role.
- Response `content` is a list of blocks; response text lives in blocks of
  `type: "text"`, in their `text` field.
- **Normalization note:** the pinned SDK's `StopReason` type is the seven-value
  set `end_turn, max_tokens, stop_sequence, tool_use, pause_turn, refusal,
  model_context_window_exceeded`, introspected directly from
  `anthropic.types.StopReason`. Anthropic's published reference page
  documents only three of those (`end_turn`, `stop_sequence`, `max_tokens`) —
  the installed SDK is treated as the wider and authoritative set, per the
  documented tie-break rule: when a vendor SDK and its prose documentation
  disagree about an enumerated value set, the SDK actually installed wins,
  because that is what the adapter receives at runtime.
- `usage` carries `input_tokens`, `output_tokens`,
  `cache_read_input_tokens`, `cache_creation_input_tokens` — there is **no**
  `total_tokens` field, so this provider's total is derived by summing the
  pair rather than read directly.
- `temperature` is **not sent** unless a benchmark explicitly set it: it is
  documented as deprecated on newer Anthropic models, and sending an
  unrequested default would put an opinion into every request this project
  did not ask for.

## Google (`google`)

Accessed 2026-09-07 against `google-genai==1.75.0`, from
<https://ai.google.dev/api/generate-content>.

- The package is **`google-genai`**, imported as `google.genai`; the legacy
  `google-generativeai` package is not used anywhere in this adapter.
- Async surface: `client.aio.models.generate_content(model=..., contents=...,
  config=...)`. System instruction, output cap, temperature, `top_p`,
  `stop_sequences`, `seed` and `response_mime_type` are all fields of
  `types.GenerateContentConfig`.
- Response text lives at `candidates[].content.parts[].text`; usage fields
  are `usage_metadata.prompt_token_count`, `.candidates_token_count`,
  `.total_token_count`, `.cached_content_token_count`,
  `.thoughts_token_count`.
- **Normalization note:** the pinned SDK's `types.FinishReason` is a
  17-value set (`FINISH_REASON_UNSPECIFIED, STOP, MAX_TOKENS, SAFETY,
  RECITATION, LANGUAGE, OTHER, BLOCKLIST, PROHIBITED_CONTENT, SPII,
  MALFORMED_FUNCTION_CALL, IMAGE_SAFETY, UNEXPECTED_TOOL_CALL,
  IMAGE_PROHIBITED_CONTENT, NO_IMAGE, IMAGE_RECITATION, IMAGE_OTHER`),
  introspected directly from the SDK; the published reference documents only
  the first six and notes that further values exist. Again, the installed
  SDK is the authoritative set.
- **A prompt rejected by a safety filter returns no candidates at all**,
  with `prompt_feedback.block_reason` set instead — a shape distinct from
  every other provider's error path, which the adapter has to check for
  explicitly rather than assuming a response always has at least one
  candidate.
- Errors are `google.genai.errors.APIError`, with `ClientError` (4xx) and
  `ServerError` (5xx) subclasses, each carrying `.code`, `.status` and
  `.message`.

## Ollama (`ollama`)

Accessed 2026-09-07 against
<https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md> and
<https://ollama.readthedocs.io/en/api/>. Has **no SDK extra** — the adapter
is plain `httpx` (already a core dependency) against `POST
{base_url}/api/chat`.

- Request fields: `model`, `messages`, `stream`, `options`, `format`,
  `keep_alive`, `tools`, `think`. This project always sends `stream: false`,
  so the reply is a single JSON object rather than a newline-delimited
  stream.
- Response fields: `model`, `created_at`, `message.role`, `message.content`,
  `done`, `done_reason`, `total_duration`, `load_duration`,
  `prompt_eval_count`, `prompt_eval_cached_count`, `prompt_eval_duration`,
  `eval_count`, `eval_duration`. `done_reason` values enumerated anywhere in
  the cited documentation are exactly `stop`, `load` and `unload`; the server
  also emits `length` when generation is cut off by the `num_predict` cap,
  but that value does **not** appear in the documentation and is recorded
  here as observed behavior, not documented behavior. The adapter maps all
  four and maps every other or absent value to `UNKNOWN` rather than
  assuming a clean stop.
- **Fields that commonly go missing:** the `load` and `unload` response
  examples in Ollama's own documentation contain **no** `prompt_eval_count`
  and **no** `eval_count` at all — the token-count keys are simply absent
  from some replies, not present-but-zero. Consistent with every other
  provider in this project, an absent count maps to `None`, never to `0`, so
  a missing count is never mistaken for a free request.
- Runs locally, so has no meaningful per-token price; the shipped price
  table treats Ollama as having no marginal cost to compute (see
  [`docs/providers-sources.md`](providers-sources.md) for the pricing source
  table).

## Cross-provider normalization principles

Two rules hold for every adapter above, stated once here rather than
repeated five times:

1. **A field a provider did not report is `None`, never a fabricated
   default.** This applies to token usage, finish reasons (mapped to an
   explicit `UNKNOWN` sentinel rather than guessed), request ids, and cost
   (`CostBreakdown.total_cost` is `None`, never `0`, when a price is
   unavailable). A consumer reading `AggregateMetrics.token_usage_coverage`
   or `.cost_coverage` can always tell what fraction of cases actually
   reported the figure being averaged.
2. **When a vendor's prose documentation and its installed SDK disagree
   about an enumerated value set (finish reasons, stop reasons, block
   reasons), the installed SDK at the pinned version wins**, because that is
   what the adapter actually receives at runtime — not what the vendor's
   docs page happened to say on the access date. Both readings are recorded
   in `docs/providers-sources.md` so the disagreement stays visible instead
   of being silently resolved one way.

Every SDK version above is pinned with a major-version cap in
`pyproject.toml`; a provider's shapes were verified against its
documentation and the installed SDK on the single date recorded per provider
above, and vendor SDKs and documentation can and do drift after that date.
Re-verifying is described in `docs/providers-sources.md`, "How to re-verify."
