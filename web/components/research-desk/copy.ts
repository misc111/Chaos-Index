import type { ResearchDeskResponse, TableRow } from "@/lib/types";

type GateChip = {
  label: string;
  description: string;
  passed: boolean;
};

function titleCaseToken(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export function displayModelName(value?: string | null): string {
  const raw = String(value || "").trim();
  if (!raw) return "candidate";
  return raw
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((token) => {
      const lower = token.toLowerCase();
      if (["glm", "gam", "dglm", "mars"].includes(lower)) return lower.toUpperCase();
      return token.charAt(0).toUpperCase() + token.slice(1);
    })
    .join(" ");
}

export function plainCheckLabel(value: string): string {
  const labels: Record<string, string> = {
    calibration_guardrail: "The predicted chances need to be closer to real results",
    materializable_candidate: "The model needs to be buildable from saved data",
    minimum_bet_count: "The test needs more bets before it counts",
    minimum_profitable_folds: "The model needs to win in more test periods",
    missing_ledger_audit: "The pregame prediction log still needs an audit",
    not_full_immutable_pregame_mlb_ledger: "The pregame prediction log is incomplete",
    not_production_grade: "The evidence is not strong enough for live use",
    research_backtest_eligible: "The historical test is not complete enough",
    theory_core_candidate: "The model needs to match the approved modeling rules",
    validation_contract_complete: "The required validation report is incomplete",
  };
  return labels[value] || titleCaseToken(value);
}

export function plainCheckDescription(value: string): string {
  const descriptions: Record<string, string> = {
    calibration_guardrail: "A beginner version: when the model says 60%, teams like that should win close to 60% of the time.",
    materializable_candidate: "The dashboard must be able to rebuild the same model later, not just show a one-time experiment.",
    minimum_bet_count: "A tiny sample can look good by luck. We need enough picks to judge it fairly.",
    minimum_profitable_folds: "The model should work across multiple slices of history, not just one favorable run.",
    missing_ledger_audit: "Every prediction needs proof that it was made before the game started.",
    not_full_immutable_pregame_mlb_ledger: "The system is still missing the complete locked record of pregame predictions.",
    not_production_grade: "This is useful for learning, but not strong enough to be the official model.",
    research_backtest_eligible: "The model still needs a complete replay against older games.",
    theory_core_candidate: "The model must follow the approved statistics rules for this project.",
    validation_contract_complete: "The review packet is missing one or more required checks.",
  };
  return descriptions[value] || "This check still needs more proof before the model can be approved.";
}

export function gateChips(policy?: TableRow | null): GateChip[] {
  const gates = policy?.gates;
  if (!gates || typeof gates !== "object" || Array.isArray(gates)) return [];
  return Object.entries(gates)
    .filter(([, value]) => value === false)
    .map(([key, value]) => ({
      label: plainCheckLabel(key),
      description: plainCheckDescription(key),
      passed: Boolean(value),
    }));
}

export function promotionSummary(promotion: ResearchDeskResponse["latest_promotion"]): string {
  if (!promotion) return "No promotion review yet.";
  const candidate = displayModelName(promotion.candidate_model_name);
  return promotion.promoted
    ? `${candidate} became the approved model.`
    : `${candidate} was tested, but it was not approved for full use.`;
}
