import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Stable filenames so `include_str!` in crates/runa can embed the build.
export default defineConfig({
  base: "/",
  plugins: [react()],
  server: {
    port: 3938,
    proxy: {
      "/api": "http://127.0.0.1:3939",
      "/health": "http://127.0.0.1:3939",
      "/ws": { target: "ws://127.0.0.1:3939", ws: true },
    },
  },
  build: {
    sourcemap: false,
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
        entryFileNames: "stats.js",
        assetFileNames: "stats.css",
      },
    },
  },
});
