import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

declare const process: { cwd: () => string };

// The frontend calls relative `/api/...` URLs; in dev Vite proxies them to the
// selected backend. Defaults to the Rust/axum port; set
// VITE_API_PROXY_TARGET=http://localhost:8000 to target the Python/FastAPI port.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiTarget = env.VITE_API_PROXY_TARGET || "http://localhost:8001";
  return {
    base: env.VITE_BASE_PATH || "/",
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
    build: { outDir: "dist" },
  };
});
