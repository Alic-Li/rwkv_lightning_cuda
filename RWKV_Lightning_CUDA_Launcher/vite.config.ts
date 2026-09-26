import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type HttpProxy } from "vite";

const LAUNCHER = "http://127.0.0.1:10721";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "./",
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          markdown: ["react-markdown", "remark-gfm", "rehype-highlight"],
        },
      },
    },
  },
  server: {
    // The launcher rejects any request whose Origin is not its own host, so
    // the dev server must present itself as the launcher: Host via
    // changeOrigin, Origin rewritten by hand (changeOrigin leaves it alone).
    // Without this every POST from the dev page answers 403.
    proxy: Object.fromEntries(
      ["/api", "/v1"].map((prefix) => [
        prefix,
        {
          target: LAUNCHER,
          changeOrigin: true,
          configure: (proxy: HttpProxy.Server) => {
            proxy.on("proxyReq", (request) => {
              if (request.getHeader("origin"))
                request.setHeader("origin", LAUNCHER);
            });
          },
        },
      ]),
    ),
  },
});
