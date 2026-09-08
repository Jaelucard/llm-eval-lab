# Provider documentation sources

Every real provider adapter in this repository was written against the official documentation
listed below, re-read on the access date shown, and against the SDK version pinned in `uv.lock`.
No request or response shape in `src/llm_eval_lab/providers/` was written from memory.

This file is the primary record. `docs/providers.md` transcribes it for readers; if the two ever
disagree, this file is the one that was checked against the vendor.

When a vendor SDK and the vendor's prose documentation disagree about an enumerated value set, the
**installed SDK at the pinned version wins**, because that is what the adapter actually receives.
Both readings are recorded below so the disagreement is visible rather than silently resolved.

## Sources

| provider | source URL | accessed | SDK version pinned in uv.lock | facts confirmed |
|---|---|---|---|---|
| `openai` | <https://developers.openai.com/api/docs/guides/migrate-to-responses>, <https://raw.githubusercontent.com/openai/openai-python/main/README.md>, <https://developers.openai.com/api/docs/api-reference/responses/object> | 2026-09-07 | `openai==2.54.0` | Responses is the recommended default for new projects and Chat Completions receives no new feature development, so `responses.create` is the default path. Request parameters: `instructions` carries the system instruction, `max_output_tokens` caps output, `text.format` replaces Chat Completions' `response_format` for JSON and structured output. There is no `stop` and no `seed` parameter on `responses.create` in 2.54.0 (confirmed by introspecting the method signature), so `GenerationParams.stop` and `.seed` are dropped for this provider. Response object: `status` is `Optional[Literal["completed","failed","in_progress","cancelled","queued","incomplete"]]`; `incomplete_details.reason` is `Optional[Literal["max_output_tokens","content_filter"]]` in the pinned SDK, while the published reference also lists `max_messages` and `steered` (both mapped to `UNKNOWN`). `usage` is `input_tokens`, `output_tokens`, `total_tokens`, `input_tokens_details.cached_tokens`, `input_tokens_details.cache_write_tokens`, `output_tokens_details.reasoning_tokens`. Text lives in `output[].content[]` items of type `output_text`. Exception hierarchy: `APIError` → `APIStatusError` (`BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `UnprocessableEntityError`, `RateLimitError`, `InternalServerError`) and `APIError` → `APIConnectionError` → `APITimeoutError`. `max_retries=0` on the client disables the SDK's own retry loop (default 2). |
| `openai_compatible` | <https://raw.githubusercontent.com/openai/openai-python/main/README.md> | 2026-09-07 | `openai==2.54.0` | `AsyncOpenAI(base_url=...)` is the documented, supported pattern for third-party and self-hosted endpoints (`OPENAI_BASE_URL` is the environment equivalent). The adapter calls `chat.completions.create` rather than `responses.create`: Chat Completions is the surface third-party servers implement, and the Responses API is an OpenAI-hosted surface a compatible endpoint is not expected to provide. Fields that commonly go missing on third-party endpoints, and which therefore degrade to `None` rather than to a fabricated value: the whole `usage` object; `usage.prompt_tokens_details.cached_tokens`; `usage.completion_tokens_details.reasoning_tokens`; `choices[].finish_reason` (mapped to `UNKNOWN` when absent); `id` (mapped to `provider_request_id=None`); and `model`, which some servers echo back differently from the requested id, so both the returned and the requested id are recorded. |
| `anthropic` | <https://platform.claude.com/docs/en/api/messages>, <https://platform.claude.com/docs/en/about-claude/models/overview> | 2026-09-07 | `anthropic==0.125.0` | Async client is `AsyncAnthropic`; the call path is `client.messages.create`. `max_tokens` is a **required** request parameter, so the adapter supplies a default when the benchmark does not. `system` is a top-level parameter, not a message. Response `content` is a list of blocks; text is in blocks of `type: "text"` with a `text` field. `stop_reason` in the pinned SDK is `Literal["end_turn","max_tokens","stop_sequence","tool_use","pause_turn","refusal","model_context_window_exceeded"]` (introspected from `anthropic.types.StopReason`); the published reference page enumerates only `end_turn`, `stop_sequence` and `max_tokens`, so the SDK is the wider and authoritative set. `usage` carries `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`; there is no `total_tokens`, so the total is derived from the pair. Exceptions: `APIStatusError` subclasses `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `ConflictError`, `UnprocessableEntityError`, `RateLimitError`, `RequestTooLargeError`, `InternalServerError`, `ServiceUnavailableError`, `OverloadedError`, `DeadlineExceededError`; plus `APIConnectionError` → `APITimeoutError`. `max_retries=0` disables SDK-internal retries. `temperature` is not sent unless the benchmark set it explicitly, because it is documented as deprecated on newer models. |
| `google` | <https://ai.google.dev/api/generate-content>, <https://ai.google.dev/gemini-api/docs/pricing> | 2026-09-07 | `google-genai==1.75.0` | The package is `google-genai`, imported as `google.genai`; the legacy `google-generativeai` package is not used anywhere in this repository. The async surface is `client.aio.models.generate_content(model=..., contents=..., config=...)`. System instruction, output cap, temperature, `top_p`, `stop_sequences`, `seed` and `response_mime_type` are all fields of `types.GenerateContentConfig`. `types.HttpOptions` carries `timeout` and `retry_options`; `types.HttpRetryOptions(attempts=1)` is what disables the SDK's own retry loop. Response: text is `candidates[].content.parts[].text`; `usage_metadata` fields are `prompt_token_count`, `candidates_token_count`, `total_token_count`, `cached_content_token_count`, `thoughts_token_count`. `types.FinishReason` at 1.75.0 is the 17-value set `FINISH_REASON_UNSPECIFIED, STOP, MAX_TOKENS, SAFETY, RECITATION, LANGUAGE, OTHER, BLOCKLIST, PROHIBITED_CONTENT, SPII, MALFORMED_FUNCTION_CALL, IMAGE_SAFETY, UNEXPECTED_TOOL_CALL, IMAGE_PROHIBITED_CONTENT, NO_IMAGE, IMAGE_RECITATION, IMAGE_OTHER` (introspected; the published reference lists only the first six and says newer ones exist). A prompt rejected by a safety filter comes back with **no candidates** and `prompt_feedback.block_reason` set, drawn from `types.BlockedReason`: `BLOCKED_REASON_UNSPECIFIED, SAFETY, OTHER, BLOCKLIST, PROHIBITED_CONTENT, IMAGE_SAFETY, MODEL_ARMOR, JAILBREAK`. Errors are `google.genai.errors.APIError` with `ClientError` (4xx) and `ServerError` (5xx) subclasses, each carrying `.code` (HTTP status), `.status` (the vendor status string) and `.message`. |
| `ollama` | <https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md>, <https://ollama.readthedocs.io/en/api/> | 2026-09-07 | none — `httpx==0.28.1` (a core dependency; Ollama has no SDK extra) | `POST {base_url}/api/chat` with `model`, `messages`, `stream`, `options`, `format`, `keep_alive`, `tools`, `think`. With `stream: false` the reply is a single JSON object with `model`, `created_at`, `message.role`, `message.content`, `done`, `done_reason`, `total_duration`, `load_duration`, `prompt_eval_count`, `prompt_eval_cached_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`. The `done_reason` values enumerated anywhere in the cited page are exactly `stop`, `load` and `unload`. The server also emits `length` when generation is cut off by the `num_predict` cap, but that value does **not** appear in `api.md` and is recorded here as observed behaviour, not as documented behaviour. The adapter maps all four and maps every other value, and an absent one, to `UNKNOWN` rather than assuming a clean stop. **Correction to the brief:** the current documentation no longer carries the sentence about `prompt_eval_count` being absent or wrong when the prompt exceeds `num_ctx`; that claim was not confirmed on this date and is not asserted here. What *is* documented and load-bearing for this adapter is that the token-count keys are simply absent from some replies — the `load` and `unload` examples in `api.md` contain no `prompt_eval_count` and no `eval_count` at all. Absent counts therefore map to `None`, never to `0`, so a missing count can never be mistaken for a free request. |

## Pricing sources

`src/llm_eval_lab/pricing/data/prices.yaml` is transcribed from the official vendor pricing pages
below, all retrieved on **2026-09-07**. That date is recorded in the table's `source` field, and the
table's `version` is the retrieval date, so a run's recorded price-table version says exactly which
day's published prices costed it.

| vendor | pricing page | accessed |
|---|---|---|
| OpenAI | <https://developers.openai.com/api/docs/pricing> | 2026-09-07 |
| Anthropic | <https://platform.claude.com/docs/en/about-claude/pricing> | 2026-09-07 |
| Google | <https://ai.google.dev/gemini-api/docs/pricing> | 2026-09-07 |
| Ollama | not applicable — models run locally, so the marginal token price is zero | 2026-09-07 |

Prices are standard (non-batch), synchronous, global-routing, per **million** tokens, in USD, and
are written as quoted strings so they parse to exact `Decimal` values. Batch discounts, cache-write
premiums, long-context tiers above a model's first pricing tier, and data-residency multipliers are
**not** modelled: a single number per direction is the only figure this project can compute honestly
from the token counts a provider returns. Where a vendor tiers input price by context length, the
table quotes the base (smaller-context) tier and says so in the entry's `notes`.

`openai_compatible` deliberately has **no** price entries. A third-party or self-hosted endpoint's
price is set by whoever runs it, so the shipped table cannot know it; those runs record
`priced=false` with a reason rather than a fabricated cost. Supply a `--prices` file to cost them.

## How to re-verify

The SDK-side facts above are reproducible without a network:

```
uv run python -c "from anthropic.types import StopReason; print(StopReason)"
uv run python -c "from google.genai import types; print([m.value for m in types.FinishReason])"
uv run python -c "from openai.types.responses.response import IncompleteDetails; print(IncompleteDetails.model_fields)"
```

The vendor-side facts require re-reading the URLs above. Re-read them whenever a pinned SDK version
changes, and update both this file's access date and `prices.yaml`'s `version` and `source`.
