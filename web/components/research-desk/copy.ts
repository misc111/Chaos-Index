import type { ResearchDeskResponse, TableRow } from "@/lib/types";

type GateChip = {
  label: string;
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

function gateLabel(value: string): string {
  const labels: Record<string, string> = {
    calibration_guardrail: "Calibration",
    materializable_candidate: "Buildable model",
    minimum_bet_count: "Bet volume",
    minimum_profitable_folds: "Profitable folds",
    research_backtest_eligible: "Backtest",
    theory_core_candidate: "Theory lane",
    validation_contract_complete: "Validation",
  };
  return labels[value] || titleCaseToken(value);
}

export function gateChips(policy?: TableRow | null): GateChip[] {
  const gates = policy?.gates;
  if (!gates || typeof gates !== "object" || Array.isArray(gates)) return [];
  return Object.entries(gates)
    .filter(([, value]) => value === false)
    .map(([key, value]) => ({
      label: gateLabel(key),
      passed: Boolean(value),
    }));
}

export function promotionSummary(promotion: ResearchDeskResponse["latest_promotion"]): string {
  if (!promotion) return "No promotion review yet.";
  const candidate = displayModelName(promotion.candidate_model_name);
  return promotion.promoted ? `${candidate} is champion.` : `${candidate} stayed in research. Promotion gates did their job.`;
}
