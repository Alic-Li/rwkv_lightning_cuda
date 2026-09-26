import { useEffect, useState } from "react";

export const ROUTES = [
  "nodes",
  "chat",
  "translate",
  "runtime",
  "training",
  "quant",
  "settings",
] as const;

export type Route = (typeof ROUTES)[number];

/** Deep links kept from the previous console. */
const ALIASES: Record<string, Route> = {
  "": "nodes",
  "state-tuning": "training",
  quantization: "quant",
};

/** Exported for tests: the single place a hash becomes a Route. */
export function parseRoute(hash: string): Route {
  const raw = hash.replace(/^#\/?/, "").split("?")[0].trim();
  // hasOwn, not `in`: `#/toString` would otherwise resolve to Object.prototype
  // and hand back a function as the route.
  if (Object.hasOwn(ALIASES, raw)) return ALIASES[raw];
  return (ROUTES as readonly string[]).includes(raw) ? (raw as Route) : "nodes";
}

export function navigate(route: Route) {
  if (parseRoute(location.hash) === route) return;
  location.hash = `/${route}`;
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseRoute(location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}
