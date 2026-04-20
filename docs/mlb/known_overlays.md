# MLB Known Overlays

This file lists items that are intentionally outside direct CAS monograph theory.

## Engineering Overlays

- registry/codegen manifests
- SQLite schema shape
- immutable pregame ledger implementation
- staging snapshot generation
- dashboard API contracts
- background orchestration commands
- release runbooks

## Betting Product Overlays

- fair-odds conversion
- vig-removal implementation details
- edge thresholds
- EV thresholds
- stake-sizing policy
- bankroll replay
- ROI by edge bucket
- realized-vs-expected value by bucket
- bet/pass reason codes

## Data-Source Overlays

- public or paid MLB data providers
- odds-feed provider conventions
- weather feed selection
- lineup-feed source selection
- park-factor provider choice

## Operator UX Overlays

- dashboard route design
- model-card presentation
- diagnostics layout
- research-admin controls
- staging-site formatting

## Rule

An overlay can be important, even mission-critical, without being presented as monograph theory. The repo should preserve that distinction in docs, model cards, and promotion reports.
