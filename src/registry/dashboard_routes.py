"""Canonical dashboard route registry for API, staging, and docs."""

from __future__ import annotations

from src.registry.types import DashboardRouteRegistryEntry


DASHBOARD_ROUTE_REGISTRY: tuple[DashboardRouteRegistryEntry, ...] = (
    DashboardRouteRegistryEntry(
        key="actualVsExpected",
        summary="Actual-vs-expected trend payload for residual-style dashboard views.",
        module_path="app/api/actual-vs-expected/route.ts",
        api_path="/api/actual-vs-expected",
        page_path="/actual-vs-expected",
        staging_file_name="actual-vs-expected.json",
        payload_contract="actualVsExpected",
    ),
    DashboardRouteRegistryEntry(
        key="betHistory",
        summary="Bet history recap payload for bankroll, replay, and bet/no-bet explanations.",
        module_path="app/api/bet-history/route.ts",
        api_path="/api/bet-history",
        page_path="/bet-history",
        staging_file_name="bet-history.json",
        payload_contract="betHistory",
    ),
    DashboardRouteRegistryEntry(
        key="gamesToday",
        summary="Upcoming slate payload for the games-today dashboard view.",
        module_path="app/api/games-today/route.ts",
        api_path="/api/games-today",
        page_path="/games-today",
        staging_file_name="games-today.json",
        payload_contract="gamesToday",
    ),
    DashboardRouteRegistryEntry(
        key="marketBoard",
        summary="Market board payload for edge screens and pricing comparisons.",
        module_path="app/api/market-board/route.ts",
        api_path="/api/market-board",
        page_path="/market-board",
        staging_file_name="market-board.json",
        payload_contract="marketBoard",
    ),
    DashboardRouteRegistryEntry(
        key="metrics",
        summary="Top-level metrics payload used by the overview surfaces.",
        module_path="app/api/metrics/route.ts",
        api_path="/api/metrics",
        staging_file_name="metrics.json",
        payload_contract="metrics",
    ),
    DashboardRouteRegistryEntry(
        key="performance",
        summary="Performance payload including experiment-aware replay variants for staging snapshots.",
        module_path="app/api/performance/route.ts",
        api_path="/api/performance",
        page_path="/performance",
        staging_file_name="performance.json",
        payload_contract="performance",
        supports_experiments=True,
    ),
    DashboardRouteRegistryEntry(
        key="predictions",
        summary="Prediction board payload for daily forecast delivery.",
        module_path="app/api/predictions/route.ts",
        api_path="/api/predictions",
        page_path="/predictions",
        staging_file_name="predictions.json",
        payload_contract="predictions",
    ),
    DashboardRouteRegistryEntry(
        key="validation",
        summary="Validation artifact digest payload served to the dashboard and sanitized staging data.",
        module_path="app/api/validation/route.ts",
        api_path="/api/validation",
        page_path="/validation",
        staging_file_name="validation.json",
        payload_contract="validation",
    ),
)


def dashboard_routes() -> tuple[DashboardRouteRegistryEntry, ...]:
    """Return all canonical dashboard route definitions."""

    return DASHBOARD_ROUTE_REGISTRY


def staging_dashboard_routes() -> tuple[DashboardRouteRegistryEntry, ...]:
    """Return dashboard routes that participate in staging snapshot generation."""

    return tuple(route for route in DASHBOARD_ROUTE_REGISTRY if route.include_in_staging)


def dashboard_route_manifest_payload() -> dict[str, object]:
    """Render the deterministic dashboard route manifest payload."""

    return {
        "version": 1,
        "source": "code_registry",
        "routes": [
            {
                "key": route.key,
                "summary": route.summary,
                "module_path": route.module_path,
                "api_path": route.api_path,
                "page_path": route.page_path,
                "staging_file_name": route.staging_file_name,
                "payload_contract": route.payload_contract,
                "include_in_staging": route.include_in_staging,
                "public": route.public,
                "supports_experiments": route.supports_experiments,
            }
            for route in DASHBOARD_ROUTE_REGISTRY
        ],
    }
