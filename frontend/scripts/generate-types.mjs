/**
 * Generate `src/api/generated-types.ts` from the checked-in OpenAPI snapshot.
 *
 * This is a script rather than a bare `openapi-typescript` invocation for one
 * reason. The API's `JSONValue` schema is recursive — a JSON value contains
 * JSON values — and the type alias openapi-typescript emits for it is a
 * self-referential alias that TypeScript rejects outright with TS2502. The
 * generator's own `transform` hook is the supported place to intervene, so the
 * two `JSONValue` schemas are emitted as `unknown`, which is what an arbitrary
 * JSON value is anyway: every consumer of `metadata`, `raw` and `extra` has to
 * narrow before using them regardless.
 *
 * Nothing else is altered, and the output is a pure function of the input, so
 * regenerating over an up-to-date file produces a byte-identical result and the
 * staleness check stays meaningful.
 */

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import openapiTS, { astToString } from "openapi-typescript";
import ts from "typescript";

const ROOT = new URL("../", import.meta.url);
const INPUT = new URL("openapi.json", ROOT);
const OUTPUT = fileURLToPath(new URL("src/api/generated-types.ts", ROOT));

/** Schema paths whose emitted type is replaced with `unknown`. */
const RECURSIVE_JSON_SCHEMAS = new Set([
  "#/components/schemas/JSONValue-Input",
  "#/components/schemas/JSONValue-Output",
]);

const ast = await openapiTS(INPUT, {
  transform(_schemaObject, options) {
    if (RECURSIVE_JSON_SCHEMAS.has(options.path ?? "")) {
      return ts.factory.createKeywordTypeNode(ts.SyntaxKind.UnknownKeyword);
    }
    return undefined;
  },
});

writeFileSync(OUTPUT, astToString(ast), "utf8");
