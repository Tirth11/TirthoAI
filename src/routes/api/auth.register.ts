import { createFileRoute } from "@tanstack/react-router";
import { supabaseAdmin } from "@/integrations/supabase/client.server";

// Simple, friction-free registration.
//
// The public supabase.auth.signUp() flow sends a confirmation email, which is
// subject to the project's built-in email rate limit ("Too many attempts").
// Instead we create an ALREADY-CONFIRMED user with the Admin API — this sends
// no email, has no email rate limit, and the signup DB triggers still fire
// (500 credits + profile). The client then signs in with the password.
export const Route = createFileRoute("/api/auth/register")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        try {
          const { email, password } = (await request.json()) as {
            email?: string;
            password?: string;
          };

          if (!email || !password) {
            return Response.json(
              { error: "missing_fields", message: "Email and password are required." },
              { status: 400 },
            );
          }
          if (password.length < 8) {
            return Response.json(
              { error: "weak_password", message: "Password must be at least 8 characters." },
              { status: 400 },
            );
          }

          const { data, error } = await supabaseAdmin.auth.admin.createUser({
            email,
            password,
            email_confirm: true,
          });

          if (!error && data?.user) {
            return Response.json({ success: true, userId: data.user.id }, { status: 200 });
          }

          const msg = (error?.message || "").toLowerCase();
          const code = (error as { code?: string } | null)?.code ?? "";
          const alreadyExists =
            code === "email_exists" ||
            msg.includes("already") ||
            msg.includes("exists") ||
            msg.includes("registered");

          if (alreadyExists) {
            // Account already exists (e.g. from an earlier attempt). Make sure
            // it's confirmed so the user can sign in, then tell the client to
            // proceed to sign-in with the password they entered.
            try {
              const { data: list } = await supabaseAdmin.auth.admin.listUsers({
                page: 1,
                perPage: 200,
              });
              const existing = list?.users?.find(
                (u) => (u.email || "").toLowerCase() === email.toLowerCase(),
              );
              if (existing) {
                await supabaseAdmin.auth.admin.updateUserById(existing.id, { email_confirm: true });
              }
            } catch {
              /* best-effort */
            }
            return Response.json(
              {
                error: "user_exists",
                message: "An account with this email already exists — signing you in.",
              },
              { status: 200 },
            );
          }

          return Response.json(
            { error: "register_failed", message: error?.message || "Could not create your account." },
            { status: 500 },
          );
        } catch {
          return Response.json(
            { error: "server_error", message: "Internal server error during registration." },
            { status: 500 },
          );
        }
      },
    },
  },
});
