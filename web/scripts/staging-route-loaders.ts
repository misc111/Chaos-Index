import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { DASHBOARD_STAGING_ROUTES } from "../lib/generated/dashboard-routes";

export type JsonRouteHandler = (request: Request) => Promise<Response>;

type RouteModule = {
  GET?: JsonRouteHandler;
  default?: {
    GET?: JsonRouteHandler;
  };
};

function resolveGetHandler(path: string, mod: RouteModule): JsonRouteHandler {
  const handler = mod.GET || mod.default?.GET;
  if (!handler) {
    throw new Error(`Route module ${path} did not export GET`);
  }
  return handler;
}

async function loadRouteModule(routePath: string): Promise<RouteModule> {
  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const modulePath = path.resolve(scriptDir, "..", routePath);
  return import(pathToFileURL(modulePath).href);
}

export const STAGING_ROUTE_LOADERS: Record<string, () => Promise<JsonRouteHandler>> = Object.fromEntries(
  DASHBOARD_STAGING_ROUTES.map((route) => [
    route.modulePath,
    async () => resolveGetHandler(route.modulePath, await loadRouteModule(route.modulePath)),
  ])
) as Record<string, () => Promise<JsonRouteHandler>>;
