import { nodeRequest } from "./http";
import type { FsResponse, MetricsResponse, NodeInfo } from "./types";

/** Agent node introspection, GPU metrics and remote directory browsing. */
export const nodeApi = {
  info: (backendId: string, signal?: AbortSignal) =>
    nodeRequest<NodeInfo>(backendId, "/api/v1/node", { signal }),

  metrics: (backendId: string, signal?: AbortSignal) =>
    nodeRequest<MetricsResponse>(backendId, "/api/v1/node/metrics", { signal }),

  /**
   * `{}` returns `{roots:[…]}`; `{path}` returns a directory listing.
   * Out-of-whitelist paths answer 403 without echoing the path back.
   */
  fs: (backendId: string, path?: string, signal?: AbortSignal) =>
    nodeRequest<FsResponse>(backendId, "/api/v1/node/fs", {
      body: path === undefined ? {} : { path },
      signal,
    }),

  /** Host-local only: Agent role + loopback origin, otherwise 400. */
  pickFile: (backendId: string) =>
    nodeRequest<{ path: string }>(backendId, "/api/v1/node/dialog/file", {
      body: {},
    }),

  pickDirectory: (backendId: string) =>
    nodeRequest<{ path: string }>(backendId, "/api/v1/node/dialog/directory", {
      body: {},
    }),

  reveal: (backendId: string) =>
    nodeRequest<{ ok: boolean }>(backendId, "/api/v1/node/dialog/reveal", {
      body: {},
    }),
};
