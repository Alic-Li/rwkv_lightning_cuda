import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  base: "./",
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
    proxy: {
      "/api": "http://127.0.0.1:8088",
      "/v1": "http://127.0.0.1:8088",
      "/logs": "http://127.0.0.1:8088",
    },
  },
});
