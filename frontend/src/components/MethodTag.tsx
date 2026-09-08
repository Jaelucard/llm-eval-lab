/**
 * The `method` tag a similarity evaluator records in its result metadata.
 *
 * `lexical_similarity` and `semantic_similarity` are different measurements
 * wearing similar numbers, so the method is shown next to every similarity
 * score rather than left in a metadata blob nobody opens.
 */

const METHOD_KEY = "method";

/** Pull the `method` metadata value off an evaluation result, if it has one. */
function methodOf(metadata: Record<string, unknown> | undefined): string | null {
  if (!metadata) return null;
  const value = metadata[METHOD_KEY];
  return typeof value === "string" && value !== "" ? value : null;
}

export interface MethodTagProps {
  /** The evaluation result's metadata object, straight off the API. */
  metadata?: Record<string, unknown> | undefined;
  /** An explicit method string, when the caller already extracted one. */
  method?: string | null;
}

export function MethodTag({ metadata, method }: MethodTagProps) {
  const value = method ?? methodOf(metadata);
  if (value === null) return null;
  return (
    <span
      className="method-tag"
      title="How this similarity was measured. Lexical overlap and embedding distance are not interchangeable."
    >
      method: {value}
    </span>
  );
}
