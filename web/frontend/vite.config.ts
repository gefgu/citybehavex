import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

declare const process: { cwd: () => string };

// The frontend calls relative `/api/...` URLs; in dev Vite proxies them to the
// FastAPI backend (see web/backend). Keep this target in sync with the uvicorn
// port used in the README / run command.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    base: env.VITE_BASE_PATH || "/",
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target: "http://localhost:8000",
          changeOrigin: true,
        },
      },
    },
    build: { outDir: "dist" },
  };
});
