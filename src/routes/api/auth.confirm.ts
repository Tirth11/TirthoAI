import { createFileRoute } from "@tanstack/react-router";
import { supabaseAdmin } from "@/integrations/supabase/client.server";

export const Route = createFileRoute("/api/auth/confirm")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        try {
          const { userId } = await request.json() as { userId: string };
          if (!userId) {
            return new Response(JSON.stringify({ error: "Missing userId" }), {
              status: 400,
              headers: { "Content-Type": "application/json" },
            });
          }

          const { data, error } = await supabaseAdmin.auth.admin.updateUserById(userId, {
            email_confirm: true,
          });

          if (error) {
            console.error("[auth-confirm] Failed to update user:", error);
            return new Response(JSON.stringify({ error: error.message }), {
              status: 500,
              headers: { "Content-Type": "application/json" },
            });
          }

          return new Response(JSON.stringify({ success: true, user: data.user }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          });
        } catch (err) {
          console.error("[auth-confirm] Unexpected error:", err);
          return new Response(JSON.stringify({ error: "Internal server error" }), {
            status: 500,
            headers: { "Content-Type": "application/json" },
          });
        }
      },
    },
  },
});
