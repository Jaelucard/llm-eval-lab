/**
 * A scrollable block for untrusted text.
 *
 * Model output, judge reasoning, rendered prompts and provider error messages
 * all arrive from outside this application's trust boundary. They are placed in
 * the DOM as text children, never as markup, so nothing in a model response can
 * become an element. The copy button exists because reading a long completion in
 * a 340-pixel box is not how anyone actually inspects one. The block itself is
 * focusable so a keyboard-only reader can scroll it at all.
 */

import { useCallback, useState } from "react";

export interface OutputBlockProps {
  label: string;
  text: string;
  /** Extra detail shown to the right of the label, such as a token count. */
  meta?: string | undefined;
}

export function OutputBlock({ label, text, meta }: OutputBlockProps) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(() => {
    void navigator.clipboard
      .writeText(text)
      .then(() => {
        setCopied(true);
        setTimeout(() => {
          setCopied(false);
        }, 1500);
      })
      .catch(() => {
        setCopied(false);
      });
  }, [text]);

  return (
    <div className="output-block">
      <div className="output-head">
        <span>{label}</span>
        {meta === undefined ? null : <span className="dim">{meta}</span>}
        <span className="spacer" />
        <span className="dim">{text.length} chars</span>
        <button type="button" onClick={copy}>
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <pre tabIndex={0} role="region" aria-label={label}>
        {text}
      </pre>
    </div>
  );
}
