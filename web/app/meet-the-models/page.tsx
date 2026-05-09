import fs from "node:fs";
import path from "node:path";

import ModelSprite from "@/components/ModelSprite";
import { MODEL_REGISTRY } from "@/lib/generated/model-manifest";
import { displayPredictionModel } from "@/lib/predictions-report";
import { loadEvidenceStatus } from "@/lib/server/research-evidence-status";
import type { CurrentBestTopModelSummary, ResearchDeskResponse, TableRow } from "@/lib/types";
import styles from "./styles.module.css";

type ModelCard = {
  key: string;
  shortName: string;
  plain: string;
  helps: string;
  fooled: string;
  role: string;
};

type ModelGroup = {
  title: string;
  lane: string;
  intro: string;
  models: ModelCard[];
};

type ModelEvidence = {
  label: string;
  tone: "ready" | "review" | "blocked" | "research" | "quiet";
  copy: string;
  detail?: string;
};

type ValidationFitStatus = {
  status?: string | null;
  feature_count?: number | null;
};

type EvidencePayload = Pick<
  ResearchDeskResponse,
  | "source_kind"
  | "source_status"
  | "evidence_stage"
  | "latest_artifact_role"
  | "promotion_eligible"
  | "production_ready"
  | "evidence_status"
  | "current_best_top_models"
>;

function repoRootFromCwd(): string {
  return path.basename(process.cwd()) === "web" ? path.resolve(process.cwd(), "..") : process.cwd();
}

function readJsonRecord(filePath: string): TableRow | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    const parsed = JSON.parse(fs.readFileSync(filePath, "utf8")) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as TableRow) : null;
  } catch {
    return null;
  }
}

function readValidationFitStatuses(): Record<string, ValidationFitStatus> {
  const metadata = readJsonRecord(path.join(repoRootFromCwd(), "artifacts", "validation", "mlb", "validation_run_metadata.json"));
  const fitStatus = metadata?.primary_model_resolution;
  if (!fitStatus || typeof fitStatus !== "object" || Array.isArray(fitStatus)) {
    return {};
  }
  const byModel = (fitStatus as TableRow).model_fit_status;
  if (!byModel || typeof byModel !== "object" || Array.isArray(byModel)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(byModel as Record<string, unknown>)
      .filter(([, value]) => Boolean(value) && typeof value === "object" && !Array.isArray(value))
      .map(([modelName, value]) => {
        const row = value as TableRow;
        return [
          modelName,
          {
            status: typeof row.status === "string" ? row.status : null,
            feature_count: typeof row.feature_count === "number" && Number.isFinite(row.feature_count) ? row.feature_count : null,
          },
        ];
      })
  );
}

function titleCaseToken(value?: string | null): string {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatEvidenceStage(value?: string | null): string {
  if (value === "production_ready") return "Production ready";
  if (value === "promotion_eligible") return "Promotion eligible";
  if (value === "fixture_demo_smoke") return "Fixture/demo smoke";
  if (value === "research_only") return "Research only";
  return value ? titleCaseToken(value) : "Unknown";
}

function formatArtifactRole(value?: string | null): string {
  if (value === "blocked_promotion_evidence") return "Blocked promotion evidence";
  if (value === "latest_research_recommendation") return "Latest research recommendation";
  if (value === "latest_promotion_eligible_evidence") return "Promotion-eligible evidence";
  return value ? titleCaseToken(value) : "No current artifact";
}

function formatReason(value?: string | null): string {
  const normalized = titleCaseToken(value).replace(/\bMlb\b/g, "MLB").replace(/\bCv\b/g, "CV").replace(/\bGlm\b/g, "GLM");
  return normalized || "Evidence gate still open";
}

function firstReason(evidence: EvidencePayload): string {
  const readinessReasons = evidence.evidence_status?.readiness_blocked_reasons || [];
  const blockedReasons = evidence.evidence_status?.blocked_reasons || [];
  return formatReason(readinessReasons[0] || blockedReasons[0]);
}

function indexTopModels(rows?: CurrentBestTopModelSummary[]): Map<string, CurrentBestTopModelSummary> {
  return new Map(
    (rows || [])
      .filter((row) => row.model_name)
      .map((row) => [String(row.model_name), row])
  );
}

function modelRegistryLane(modelKey: string): string {
  const row = (MODEL_REGISTRY as Record<string, { lane?: string }>)[modelKey];
  return row?.lane || "experimental";
}

function buildModelEvidence(
  modelKey: string,
  evidence: EvidencePayload,
  topModels: Map<string, CurrentBestTopModelSummary>,
  fitStatuses: Record<string, ValidationFitStatus>
): ModelEvidence {
  const topModel = topModels.get(modelKey);
  const fitStatus = fitStatuses[modelKey];
  const lane = modelRegistryLane(modelKey);
  const artifactRole = formatArtifactRole(evidence.latest_artifact_role);

  if (modelKey === "ensemble") {
    if (evidence.production_ready) {
      return {
        label: "Production ready",
        tone: "ready",
        copy: "The desk blend can only claim approval when its source champion clears the full-ledger gate.",
      };
    }
    return {
      label: "No approved champion",
      tone: "blocked",
      copy: "The final call stays gated until a model family clears full-ledger promotion evidence.",
      detail: artifactRole,
    };
  }

  if (topModel && evidence.production_ready) {
    return {
      label: "Production ready",
      tone: "ready",
      copy: `${topModel.display_name || displayPredictionModel(modelKey)} is present in the current production-ready packet.`,
      detail: topModel.rank ? `Rank ${topModel.rank}` : undefined,
    };
  }

  if (topModel && evidence.promotion_eligible) {
    return {
      label: "Promotion review",
      tone: "review",
      copy: "This family is in the current full-ledger review packet but still needs production-readiness gates.",
      detail: topModel.rank ? `Rank ${topModel.rank}` : artifactRole,
    };
  }

  if (topModel) {
    if (evidence.latest_artifact_role === "latest_research_recommendation") {
      return {
        label: "Research-ranked",
        tone: "research",
        copy: "Current research ranks this family, but the artifact is not promotion-eligible evidence.",
        detail: topModel.rank ? `Rank ${topModel.rank} in ${artifactRole.toLowerCase()}` : artifactRole,
      };
    }

    return {
      label: "Promotion blocked",
      tone: "blocked",
      copy: `Current evidence ranks it, but promotion is blocked by ${firstReason(evidence).toLowerCase()}.`,
      detail: topModel.rank ? `Rank ${topModel.rank} in ${artifactRole.toLowerCase()}` : artifactRole,
    };
  }

  if (fitStatus?.status === "fit_ok") {
    return {
      label: "Fit diagnostics only",
      tone: "research",
      copy: "A validation fit exists, but this card is not in the current promotion packet.",
      detail: fitStatus.feature_count ? `${fitStatus.feature_count} active features` : undefined,
    };
  }

  if (lane === "core") {
    return {
      label: "Core, not promoted",
      tone: "research",
      copy: "Theory-core status does not make this family promotion-ready without a current evidence packet.",
      detail: artifactRole,
    };
  }

  if (lane === "extension") {
    return {
      label: "Challenger only",
      tone: "quiet",
      copy: "Theory-compatible challenger lane; kept separate from the core champion path until evidence catches up.",
      detail: "Not promotion-ready",
    };
  }

  return {
    label: "Research-only",
    tone: "quiet",
    copy: "Opt-in comparison family. Useful as a scout, not eligible for automatic promotion.",
    detail: "Not promotion-ready",
  };
}

const groups: ModelGroup[] = [
  {
    title: "The final desk call",
    lane: "Desk blend",
    intro: "This is the model you usually want first. It listens to the room instead of trusting one voice.",
    models: [
      {
        key: "ensemble",
        shortName: "The table captain",
        role: "Default forecast",
        plain: "Combines the model stack into one calmer probability.",
        helps: "Best when no single model has earned the right to dominate the board.",
        fooled: "Can still share the same blind spot if every model is missing the same baseball context.",
      },
    ],
  },
  {
    title: "The steady core",
    lane: "Core models",
    intro: "These are the workhorse actuarial models. They prefer simple relationships and punish noise.",
    models: [
      {
        key: "glm_ridge",
        shortName: "The stabilizer",
        role: "Core GLM",
        plain: "Keeps every input in the conversation, but quiets the loud ones.",
        helps: "Best when a lot of signals are useful but none should get to shout.",
        fooled: "Can keep too many weak signals alive and make the forecast feel safer than it is.",
      },
      {
        key: "glm_elastic_net",
        shortName: "The compromise",
        role: "Core GLM",
        plain: "Blends ridge's patience with lasso's willingness to cut clutter.",
        helps: "Best when related baseball signals travel in packs, like pitching, bullpen, and lineup strength.",
        fooled: "Can mute a real edge if the useful signals are tangled together.",
      },
      {
        key: "glm_lasso",
        shortName: "The pruner",
        role: "Core GLM",
        plain: "Deletes weak inputs so the forecast can focus on the strongest evidence.",
        helps: "Best when the feature set is crowded and the model needs discipline.",
        fooled: "Can cut a subtle signal that only matters in certain matchups.",
      },
      {
        key: "glm_vanilla",
        shortName: "The baseline",
        role: "Core GLM",
        plain: "The simple no-shrinkage version. It shows what the raw GLM wants to say.",
        helps: "Best as a sanity check against the regularized models.",
        fooled: "Most likely to overreact to small samples or noisy features.",
      },
    ],
  },
  {
    title: "Credibility models",
    lane: "Core models, opt-in",
    intro: "These start with another opinion, then ask how much the data should move away from it.",
    models: [
      {
        key: "glm_lasso_market_credibility",
        shortName: "Market-aware credibility",
        role: "Opt-in core",
        plain: "Starts with a clean market opinion and lets lasso decide what the model adds.",
        helps: "Best when the betting market is a useful reference, but not the whole answer.",
        fooled: "Blocked if the market complement is stale, missing, or too close to copying the board.",
      },
      {
        key: "glm_lasso_prior_credibility",
        shortName: "Prior-aware credibility",
        role: "Opt-in core",
        plain: "Starts with a prior model opinion and learns what today's data should change.",
        helps: "Best when there is a trustworthy pregame prior to build from.",
        fooled: "Can inherit old mistakes if the prior is not audited and frozen correctly.",
      },
    ],
  },
  {
    title: "Theory-compatible challengers",
    lane: "Theory-compatible challengers",
    intro: "These are still actuarial-adjacent, but they are kept out of the default champion lane until evidence catches up.",
    models: [
      {
        key: "gam_spline",
        shortName: "The curve reader",
        role: "Extension",
        plain: "Lets some relationships bend instead of forcing every input into a straight line.",
        helps: "Best when a baseball signal has a sweet spot or threshold.",
        fooled: "Can draw a beautiful curve through noise and call it signal.",
      },
      {
        key: "glmm_logit",
        shortName: "The team-context model",
        role: "Extension",
        plain: "Keeps the main inputs, then gives teams their own background tendencies.",
        helps: "Best when team identity matters beyond the measured features.",
        fooled: "Can confuse team reputation with real current strength.",
      },
      {
        key: "dglm_margin",
        shortName: "The margin translator",
        role: "Extension",
        plain: "Thinks about expected run margin first, then translates that into win chance.",
        helps: "Best when spread shape says more than winner-only data.",
        fooled: "Can drift if the run-margin variance is wrong.",
      },
      {
        key: "goals_poisson",
        shortName: "The run counter",
        role: "Extension",
        plain: "Treats the game like two run-scoring processes and turns that into a win probability.",
        helps: "Best for normal scoring environments where run rates are well behaved.",
        fooled: "Messy pitching changes, weather, or unusual scoring environments can throw it off.",
      },
    ],
  },
  {
    title: "Experimental scouts",
    lane: "Not promotion-ready",
    intro: "These are useful second opinions. They can spot patterns, but they do not get automatic trust.",
    models: [
      {
        key: "mars_hinge",
        shortName: "The bend finder",
        role: "Experimental",
        plain: "Looks for sharp turning points where the relationship changes direction.",
        helps: "Best for discovering thresholds worth testing later.",
        fooled: "Can mistake random wiggles for real turning points.",
      },
      {
        key: "two_stage",
        shortName: "The assembly line",
        role: "Experimental",
        plain: "Solves the problem in two steps instead of one big leap.",
        helps: "Best when one model should create an intermediate baseball view before the final call.",
        fooled: "A bad first stage can poison the second stage.",
      },
      {
        key: "rf",
        shortName: "The voting forest",
        role: "Experimental",
        plain: "Lets many small decision trees vote on the game.",
        helps: "Best for rough interaction hunting.",
        fooled: "Can look confident because many trees repeated the same noisy split.",
      },
      {
        key: "gbdt",
        shortName: "The booster",
        role: "Experimental",
        plain: "Builds a chain of small trees, each trying to fix the last one's misses.",
        helps: "Best when there are repeated patterns the GLMs are too simple to catch.",
        fooled: "Can chase yesterday's mistakes too aggressively.",
      },
      {
        key: "nn_mlp",
        shortName: "The pattern sponge",
        role: "Experimental",
        plain: "A neural model that absorbs many signals and learns flexible combinations.",
        helps: "Best as a broad pattern detector when there is enough clean data.",
        fooled: "Can become a black box that remembers noise instead of baseball truth.",
      },
      {
        key: "bayes_goals",
        shortName: "The uncertainty counter",
        role: "Experimental",
        plain: "A Bayesian run model that tries to carry uncertainty along with the score view.",
        helps: "Best when the size of uncertainty matters as much as the point estimate.",
        fooled: "Can be only as good as the assumptions behind its run process.",
      },
      {
        key: "bayes_bt_state_space",
        shortName: "The form tracker",
        role: "Experimental",
        plain: "Tracks team strength as something that moves over time.",
        helps: "Best when teams are changing and stale season averages lag behind.",
        fooled: "Can overread short hot streaks or cold streaks.",
      },
    ],
  },
];

export default function MeetTheModelsPage() {
  const evidence = loadEvidenceStatus("MLB");
  const topModels = indexTopModels(evidence.current_best_top_models);
  const fitStatuses = readValidationFitStatuses();
  const evidenceLabel = evidence.evidence_status?.label || formatArtifactRole(evidence.latest_artifact_role);

  return (
    <div className={styles.page}>
      <div className={styles.overviewGrid}>
        <section className={styles.intro} aria-labelledby="meet-models-title">
          <div>
            <h2 id="meet-models-title">The model room, without the math fog.</h2>
            <p>
              Each model is a different kind of scout. The badges below come from current MLB evidence artifacts, so
              a good research showing stays separate from promotion readiness.
            </p>
          </div>
          <div className={styles.legend} aria-label="How to read the model cards">
            <span>Plain idea</span>
            <span>When it helps</span>
            <span>How it gets fooled</span>
          </div>
        </section>

        <section className={styles.evidenceStrip} aria-label="Current MLB evidence status">
          <div>
            <span className={styles.evidenceEyebrow}>Live MLB evidence</span>
            <strong>{evidenceLabel}</strong>
            <p>{evidence.evidence_status?.pointer_semantics || "Current model evidence is loaded from the MLB report artifacts."}</p>
          </div>
          <div className={styles.evidenceFacts}>
            <div>
              <span>Stage</span>
              <strong>{formatEvidenceStage(evidence.evidence_stage || evidence.evidence_status?.evidence_stage)}</strong>
            </div>
            <div>
              <span>Artifact role</span>
              <strong>{formatArtifactRole(evidence.latest_artifact_role)}</strong>
            </div>
            <div>
              <span>Promotion-ready</span>
              <strong>{evidence.production_ready ? "Yes" : "No"}</strong>
            </div>
          </div>
        </section>
      </div>

      <div className={styles.groupStack}>
        {groups.map((group) => (
          <section className={styles.group} key={group.title} aria-labelledby={`${group.title}-heading`}>
            <div className={styles.groupHeader}>
              <span className={styles.groupLane}>{group.lane}</span>
              <h3 id={`${group.title}-heading`}>{group.title}</h3>
              <p>{group.intro}</p>
            </div>
            <div className={styles.modelGrid}>
              {group.models.map((model) => {
                const modelEvidence = buildModelEvidence(model.key, evidence, topModels, fitStatuses);
                return (
                  <article className={styles.modelCard} key={model.key}>
                    <div className={styles.modelTop}>
                      <ModelSprite model={model.key} className={styles.sprite} />
                      <div>
                        <div className={styles.roleRow}>
                          <span className={styles.role}>{model.role}</span>
                          <span className={`${styles.statusBadge} ${styles[modelEvidence.tone]}`}>{modelEvidence.label}</span>
                        </div>
                        <h4>{displayPredictionModel(model.key)}</h4>
                        <p className={styles.shortName}>{model.shortName}</p>
                      </div>
                    </div>
                    <div className={styles.statusCallout}>
                      <span>Evidence status</span>
                      <p>{modelEvidence.copy}</p>
                      {modelEvidence.detail ? <small>{modelEvidence.detail}</small> : null}
                    </div>
                    <p className={styles.plain}>{model.plain}</p>
                    <div className={styles.explainGrid}>
                      <div>
                        <span>Helps when</span>
                        <p>{model.helps}</p>
                      </div>
                      <div>
                        <span>Gets fooled by</span>
                        <p>{model.fooled}</p>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
