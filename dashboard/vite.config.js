import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Stable filenames so `include_str!` in crates/runa can embed the build.
export default defineConfig({
  base: "/dashboard/",
  plugins: [react()],
  server: {
    proxy: {
      "/dashboard/snapshot": "http://127.0.0.1:8080",
      "/dashboard/ws": { target: "ws://127.0.0.1:8080", ws: true },
    },
  },
  build: {
    sourcemap: false,
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
        entryFileNames: "dashboard.js",
        assetFileNames: "dashboard.css",
      },
    },
  },
});
