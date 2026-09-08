/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

/**
 * Dev runs on 5173 and proxies `/api` to the backend on 127.0.0.1:8000, so the
 * browser only ever talks to one origin and CORS stays out of the normal loop.
 * Production is served by the backend's static mount from the same origin, so
 * the same relative `/api` paths work unchanged.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: false,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    // The backend serves this bundle under a content-security-policy of
    // `style-src 'self'`, which forbids inline <style> blocks. Vite emits a
    // linked stylesheet by default; this keeps it that way even for a small one.
    cssCodeSplit: false,
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/test/**/*.test.{ts,tsx}"],
    css: false,
    restoreMocks: true,
  },
});
