import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import tseslint from "typescript-eslint";

/**
 * Two project-specific bans carry real weight here and are errors, not warnings.
 *
 * `dangerouslySetInnerHTML` is banned because every string this dashboard
 * renders — model output, benchmark text, judge reasoning, provider error
 * messages — is untrusted content that arrived from a model or a shared
 * benchmark file. It is rendered as text, always.
 *
 * The `style` JSX attribute is banned on the same footing because the backend
 * serves this bundle under `style-src 'self'`, which blocks inline style
 * attributes. A component that positions itself with an inline style works in
 * the Vite dev server and silently breaks in production. Geometry belongs in
 * SVG presentation attributes or in a class.
 */
export default tseslint.config(
  { ignores: ["dist", "coverage", "src/api/generated-types.ts"] },
  js.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2022,
      globals: { ...globals.browser },
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
      "no-restricted-syntax": [
        "error",
        {
          selector: "JSXAttribute[name.name='dangerouslySetInnerHTML']",
          message:
            "Model output and benchmark strings are untrusted. Render them as text.",
        },
        {
          selector: "JSXAttribute[name.name='style']",
          message:
            "The served content-security-policy is style-src 'self'; inline styles are blocked. Use a class or an SVG presentation attribute.",
        },
      ],
      "@typescript-eslint/consistent-type-imports": [
        "error",
        { prefer: "type-imports", fixStyle: "inline-type-imports" },
      ],
      "@typescript-eslint/restrict-template-expressions": [
        "error",
        { allowNumber: true },
      ],
    },
  },
  {
    files: ["vite.config.ts"],
    languageOptions: { globals: { ...globals.node } },
  },
  {
    files: ["src/test/**/*.{ts,tsx}"],
    rules: {
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/no-unsafe-assignment": "off",
    },
  },
  {
    // Plain JavaScript config and tooling files are not part of the TypeScript
    // program, so the type-aware rules have nothing to read and must be off.
    files: ["eslint.config.js", "scripts/**/*.mjs"],
    ...tseslint.configs.disableTypeChecked,
    languageOptions: {
      ...tseslint.configs.disableTypeChecked.languageOptions,
      globals: { ...globals.node },
    },
  },
);
