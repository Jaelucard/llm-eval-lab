/**
 * Helpers for the optional record fields the API returns.
 *
 * A field like `JudgeScale.labels` or `JudgeVerdict.per_criterion` is optional
 * in the schema, so reading it yields `undefined` when the backend omitted it.
 * Handling that at the point of use forces a nullish coalesce on every call
 * site; handling it here means the absence is described once, in a signature
 * that says the value may not be there.
 */

/** Entries of a record that may be absent, as an empty list when it is. */
export function entriesOf<T>(
  record: Record<string, T> | undefined,
): [string, T][] {
  return record === undefined ? [] : Object.entries(record);
}
