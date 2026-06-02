import { supabaseAdmin } from "@/integrations/supabase/client.server";
import { createFileRoute } from "@tanstack/react-router";

type StatusResponse = {
  lovable: {
    configured: boolean;
    source: "env" | "missing";
  };
  nvidia: {
    envConfigured: boolean;
    userOverride: boolean;
    activeSource: "user" | "env" | "missing";
    maskedKey: string | null;
    testOk: boolean | null;
    testError?: string;
  };
};

function maskKey(key: string | null | undefined): string | null {
  if (!key) return null;
  if (key.length <= 8) return "••••";
  return `${key.slice(0, 6)}…${key.slice(-4)}`;
}

async function testNvidiaKey(key: string): Promise<{ ok: boolean; error?: string }> {
  try {
    const res = await fetch("https://integrate.api.nvidia.com/v1/models", {
      headers: { Authorization: `Bearer ${key}` },
    });
    if (res.ok) return { ok: true };
    return { ok: false, error: `HTTP ${res.status}` };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : "network error" };
  }
}

async function authenticate(request: Request): Promise<string | null> {
  const authHeader = request.headers.get("authorization") ?? "";
  const token = authHeader.toLowerCase().startsWith("bearer ")
    ? authHeader.slice(7).trim()
    : "";
  if (!token) return null;
  const { data, error } = await supabaseAdmin.auth.getUser(token);
  if (error || !data.user) return null;
  return data.user.id;
}

export const Route = createFileRoute("/api/provider-status")({
  server: {
    handlers: {
      GET: async ({ request }) => {
        const userId = await authenticate(request);
        if (!userId) return new Response("Unauthorized", { status: 401 });

        const envLovable = process.env.LOVABLE_API_KEY;
        const envNvidia = process.env.NVIDIA_API_KEY;

        const { data: keyRow } = await supabaseAdmin
          .from("user_api_keys")
          .select("nvidia_api_key")
          .eq("user_id", userId)
          .maybeSingle();
        const userNvidia = keyRow?.nvidia_api_key?.trim() || null;

        const activeNvidia = userNvidia || envNvidia || null;
        const test = activeNvidia ? await testNvidiaKey(activeNvidia) : null;

        const payload: StatusResponse = {
          lovable: {
            configured: !!envLovable,
            source: envLovable ? "env" : "missing",
          },
          nvidia: {
            envConfigured: !!envNvidia,
            userOverride: !!userNvidia,
            activeSource: userNvidia ? "user" : envNvidia ? "env" : "missing",
            maskedKey: maskKey(activeNvidia),
            testOk: test ? test.ok : null,
            testError: test?.error,
          },
        };

        return Response.json(payload);
      },

      POST: async ({ request }) => {
        const userId = await authenticate(request);
        if (!userId) return new Response("Unauthorized", { status: 401 });

        let body: { nvidiaKey?: string | null };
        try {
          body = (await request.json()) as { nvidiaKey?: string | null };
        } catch {
          return new Response("Invalid JSON", { status: 400 });
        }

        const raw = body.nvidiaKey;
        const next = typeof raw === "string" ? raw.trim() : null;

        if (next && (next.length < 20 || next.length > 512)) {
          return new Response(
            JSON.stringify({ error: "invalid_key", message: "Key looks malformed." }),
            { status: 400, headers: { "Content-Type": "application/json" } },
          );
        }

        const value = next && next.length > 0 ? next : null;

        const { error } = await supabaseAdmin
          .from("user_api_keys")
          .upsert(
            { user_id: userId, nvidia_api_key: value, updated_at: new Date().toISOString() },
            { onConflict: "user_id" },
          );

        if (error) {
          return new Response(
            JSON.stringify({ error: "save_failed", message: error.message }),
            { status: 500, headers: { "Content-Type": "application/json" } },
          );
        }

        // Test the active key after save so the UI gets immediate feedback.
        const activeNvidia = value || process.env.NVIDIA_API_KEY || null;
        const test = activeNvidia ? await testNvidiaKey(activeNvidia) : null;

        return Response.json({
          ok: true,
          cleared: value === null,
          testOk: test ? test.ok : null,
          testError: test?.error,
        });
      },
    },
  },
});
