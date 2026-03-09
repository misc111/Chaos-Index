import { Suspense } from "react";
import BetSizingExperience from "@/components/bet-sizing/BetSizingExperience";

export default function BetSizingPage() {
  return (
    <Suspense fallback={<p className="small">Loading bet sizing view...</p>}>
      <BetSizingExperience />
    </Suspense>
  );
}
