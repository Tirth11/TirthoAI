import { defineConfig, loadEnv } from "vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import viteReact from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsConfigPaths from "vite-tsconfig-paths";

// Hand-written TanStack Start config (replaces @lovable.dev/vite-tanstack-config).
// Bundles the same underlying plugins the Lovable wrapper used — tanstackStart,
// React, Tailwind v4, tsconfig path aliases — and, on build, the Nitro deploy
// plugin targeting Vercel (override with NITRO_PRESET, e.g. "node-server").
export default defineConfig(async ({ mode, command }) => {
  // Vite only auto-exposes VITE_* vars to import.meta.env. Server-side code
  // (SSR + /api routes run in-process during dev) reads non-VITE secrets via
  // process.env — SUPABASE_SERVICE_ROLE_KEY, GROQ_API_KEY, etc. Load every var
  // from .env / .env.local into process.env without overriding the real shell.
  const env = loadEnv(mode, process.cwd(), "");
  for (const [key, value] of Object.entries(env)) {
    if (process.env[key] === undefined) process.env[key] = value;
  }

  const plugins = [
    tailwindcss(),
    tsConfigPaths({ projects: ["./tsconfig.json"] }),
    tanstackStart({
      importProtection: {
        behavior: "error",
        client: { files: ["**/server/**"], specifiers: ["server-only"] },
      },
      server: { entry: "server" },
    }),
  ];

  // On build, add Nitro so `vite build` emits a deployable server bundle.
  // Vercel preset writes the Build Output API folder (.vercel/output).
  if (command === "build") {
    const { nitro } = await import("nitro/vite");
    plugins.push(nitro({ preset: process.env.NITRO_PRESET ?? "vercel" }));
  }

  plugins.push(viteReact());

  return {
    plugins,
    resolve: {
      alias: { "@": `${process.cwd()}/src` },
      dedupe: [
        "react",
        "react-dom",
        "react/jsx-runtime",
        "react/jsx-dev-runtime",
        "@tanstack/react-query",
        "@tanstack/query-core",
      ],
    },
    server: { host: "::", port: 8080 },
  };
});
