/**
 * Everything that went into a model-graded verdict, kept in one place.
 *
 * A judge score is an opinion produced by another language model, and the only
 * way a reader can weigh it is by seeing what produced it: which judge models,
 * which rubric hash, which template, how far apart repeated judgments landed,
 * and how many outputs failed to parse. Two flags get their own badges because
 * they change how much the number is worth — `self_preference_risk`, where the
 * judge is grading its own family, and `high_judge_variance`, where the panel
 * did not agree. Both are advisory and neither is blocking.
 */

import { Badge } from "./Badge";
import { OutputBlock } from "./OutputBlock";
import { ScrollRegion } from "./ScrollRegion";
import type { JudgeProvenance } from "../api/client";
import { EMPTY, formatCost, formatCount, formatNumber, formatScore } from "../lib/format";
import { entriesOf } from "../lib/records";

export interface JudgeVerdictPanelProps {
  provenance: JudgeProvenance;
  evaluatorId: string;
}

export function JudgeVerdictPanel({ provenance, evaluatorId }: JudgeVerdictPanelProps) {
  const scale = provenance.scale;

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Judge provenance</h2>
        <span className="mono dim">{evaluatorId}</span>
        <span className="spacer" />
        {provenance.self_preference_risk ? (
          <Badge
            tone="warn"
            title="The judge model matches the model under test. A model grading itself tends to grade itself well."
          >
            self preference risk
          </Badge>
        ) : null}
        {provenance.high_judge_variance ? (
          <Badge
            tone="warn"
            title="Repeated or panel judgments disagreed by more than the configured threshold. Treat the aggregate as soft."
          >
            high judge variance
          </Badge>
        ) : null}
        {provenance.parse_failures > 0 ? (
          <Badge tone="warn" title="Judge outputs that could not be parsed into a verdict.">
            {provenance.parse_failures} parse failures
          </Badge>
        ) : null}
      </div>

      <div className="panel-body">
        <dl className="deflist">
          <dt>Judge models</dt>
          <dd>{provenance.judge_models.join(", ") || EMPTY}</dd>
          <dt>Template</dt>
          <dd>{provenance.template_id}</dd>
          <dt>Rubric hash</dt>
          <dd>{provenance.rubric_hash}</dd>
          <dt>Scale</dt>
          <dd>
            {scale.kind} {formatNumber(scale.minimum)}..{formatNumber(scale.maximum)}
          </dd>
          <dt>Scale anchors</dt>
          <dd>
            {entriesOf(scale.labels)
              .map(([point, label]) => `${point}: ${label}`)
              .join("  |  ") || EMPTY}
          </dd>
          <dt>Aggregation</dt>
          <dd>{provenance.aggregation}</dd>
          <dt>Dispersion</dt>
          <dd>
            {provenance.dispersion === null ? (
              <span className="dim">not computed (single judgment)</span>
            ) : (
              formatNumber(provenance.dispersion, 3)
            )}
          </dd>
          <dt>Agreement</dt>
          <dd>
            {provenance.agreement === null ? (
              <span className="dim">not computed (single judgment)</span>
            ) : (
              formatNumber(provenance.agreement, 3)
            )}
          </dd>
          <dt>Judge tokens</dt>
          <dd>
            {formatCount(provenance.usage.input_tokens)} in /{" "}
            {formatCount(provenance.usage.output_tokens)} out
          </dd>
          <dt>Judge cost</dt>
          <dd>{formatCost(provenance.cost_usd)}</dd>
        </dl>
      </div>

      <ScrollRegion label="Judge verdicts table">
        <table className="data">
          <caption className="sr-only">
            One row per judgment, with its raw score, confidence, per-criterion scores and
            recorded violations.
          </caption>
          <thead>
            <tr>
              <th scope="col" className="num">
                #
              </th>
              <th scope="col" className="num">
                Raw score
              </th>
              <th scope="col" className="num">
                Confidence
              </th>
              <th scope="col">Per criterion</th>
              <th scope="col">Violations</th>
            </tr>
          </thead>
          <tbody>
            {provenance.verdicts.map((verdict, index) => (
              <tr key={`${String(index)}-${String(verdict.score)}`}>
                <td className="num">{index + 1}</td>
                <td className="num">{formatNumber(verdict.score, 2)}</td>
                <td className="num">
                  {verdict.confidence === null || verdict.confidence === undefined ? (
                    <span className="dim">—</span>
                  ) : (
                    formatScore(verdict.confidence, 2)
                  )}
                </td>
                <td className="mono">
                  {entriesOf(verdict.per_criterion)
                    .map(([name, value]) => `${name}=${formatNumber(value, 2)}`)
                    .join("  ") || EMPTY}
                </td>
                <td className="mono">{verdict.violations.join(", ") || EMPTY}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollRegion>

      <div className="panel-body grid grid-2">
        {provenance.verdicts.map((verdict, index) => (
          <OutputBlock
            key={`reasoning-${String(index)}`}
            label={`Judge reasoning ${String(index + 1)}`}
            text={verdict.reasoning}
          />
        ))}
        {provenance.rendered_prompts.map((prompt, index) => (
          <OutputBlock
            key={`prompt-${String(index)}`}
            label={`Rendered prompt ${String(index + 1)}`}
            text={prompt}
          />
        ))}
        {provenance.raw_outputs.map((raw, index) => (
          <OutputBlock
            key={`raw-${String(index)}`}
            label={`Raw judge output ${String(index + 1)}`}
            text={raw}
          />
        ))}
      </div>

      <p className="panel-note">
        A judge score is one model&apos;s opinion of another model&apos;s answer. It is
        reported with the prompt, the rubric hash and the spread across judgments so it can
        be audited, not because the number is authoritative on its own.
      </p>
    </div>
  );
}
