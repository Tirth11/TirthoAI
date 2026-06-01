import { useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Sparkles, Loader2, Mail, Lock, Eye, EyeOff, Gift } from "lucide-react";
import { toast } from "sonner";
import { setRemember } from "@/lib/remember-me";
import { ForgotPasswordModal } from "@/components/ForgotPasswordModal";

export function AuthScreen() {
  const [mode, setMode] = useState<"signin" | "signup">("signup");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [remember, setRememberState] = useState(true);
  const [forgotOpen, setForgotOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<{
    email?: string;
    password?: string;
    confirm?: string;
  }>({});

  const validate = (): boolean => {
    const next: typeof errors = {};
    if (!email) next.email = "Email is required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) next.email = "Enter a valid email";
    if (!password) next.password = "Password is required";
    else if (password.length < 6) next.password = "At least 6 characters";
    if (mode === "signup") {
      if (!confirm) next.confirm = "Please confirm your password";
      else if (confirm !== password) next.confirm = "Passwords do not match";
    }
    setErrors(next);
    return Object.keys(next).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;
    setLoading(true);
    try {
      // Persist remember-me preference BEFORE the auth call so the boot guard
      // doesn't immediately sign the user out on next page load.
      setRemember(remember);

      if (mode === "signup") {
        const { error } = await supabase.auth.signUp({
          email,
          password,
          options: { emailRedirectTo: `${window.location.origin}/` },
        });
        if (error) throw error;
        const { data: sess } = await supabase.auth.getSession();
        if (!sess.session) {
          const { error: signInErr } = await supabase.auth.signInWithPassword({
            email,
            password,
          });
          if (signInErr) throw signInErr;
        }
        toast.success("Welcome! 500 free credits added to your account 🎉");
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        toast.success("Welcome back!");
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4 py-10">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex flex-col items-center">
          <div
            className="mb-3 flex h-14 w-14 items-center justify-center rounded-2xl text-white shadow-xl"
            style={{ background: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
          >
            <Sparkles className="h-7 w-7" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight">TirthoAI</h1>
          <p className="mt-1 text-xs uppercase tracking-wider text-muted-foreground">
            Your multi-model AI workspace
          </p>
          <div className="mt-4 inline-flex items-center gap-1.5 rounded-full border border-primary/30 bg-primary/5 px-3 py-1 text-[11px] font-medium text-primary">
            <Gift className="h-3 w-3" />
            500 free credits on signup
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
          <h2 className="mb-4 text-center text-sm font-semibold text-foreground">
            {mode === "signin" ? "Sign in to continue" : "Create your free account"}
          </h2>

          <form onSubmit={handleSubmit} className="space-y-3" noValidate>
            <div>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value);
                    if (errors.email) setErrors((p) => ({ ...p, email: undefined }));
                  }}
                  placeholder="you@example.com"
                  required
                  autoComplete="email"
                  aria-invalid={!!errors.email}
                  className="w-full rounded-lg border border-border bg-background py-2.5 pl-9 pr-3 text-sm placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20 aria-invalid:border-destructive"
                />
              </div>
              {errors.email && (
                <p className="mt-1 text-[11px] text-destructive">{errors.email}</p>
              )}
            </div>

            <div>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                    if (errors.password) setErrors((p) => ({ ...p, password: undefined }));
                  }}
                  placeholder="Password (min 6 characters)"
                  required
                  minLength={6}
                  autoComplete={mode === "signin" ? "current-password" : "new-password"}
                  aria-invalid={!!errors.password}
                  className="w-full rounded-lg border border-border bg-background py-2.5 pl-9 pr-10 text-sm placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20 aria-invalid:border-destructive"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {errors.password && (
                <p className="mt-1 text-[11px] text-destructive">{errors.password}</p>
              )}
            </div>

            {mode === "signup" && (
              <div>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <input
                    type={showPassword ? "text" : "password"}
                    value={confirm}
                    onChange={(e) => {
                      setConfirm(e.target.value);
                      if (errors.confirm) setErrors((p) => ({ ...p, confirm: undefined }));
                    }}
                    placeholder="Confirm password"
                    required
                    minLength={6}
                    autoComplete="new-password"
                    aria-invalid={!!errors.confirm}
                    className="w-full rounded-lg border border-border bg-background py-2.5 pl-9 pr-3 text-sm placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20 aria-invalid:border-destructive"
                  />
                </div>
                {errors.confirm && (
                  <p className="mt-1 text-[11px] text-destructive">{errors.confirm}</p>
                )}
              </div>
            )}

            <div className="flex items-center justify-between pt-1">
              <label className="flex cursor-pointer items-center gap-2 text-[12px] text-muted-foreground select-none">
                <input
                  type="checkbox"
                  checked={remember}
                  onChange={(e) => setRememberState(e.target.checked)}
                  className="h-3.5 w-3.5 rounded border-border text-primary focus:ring-primary/30"
                />
                Remember me
              </label>
              {mode === "signin" && (
                <button
                  type="button"
                  onClick={() => setForgotOpen(true)}
                  className="text-[12px] font-medium text-primary hover:underline"
                >
                  Forgot password?
                </button>
              )}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold text-primary-foreground shadow-md transition hover:opacity-95 disabled:opacity-60"
              style={{ background: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
            >
              {loading && <Loader2 className="h-4 w-4 animate-spin" />}
              {mode === "signin" ? "Sign in" : "Create account"}
            </button>
          </form>

          <p className="mt-4 text-center text-xs text-muted-foreground">
            {mode === "signin" ? "New here? " : "Already have an account? "}
            <button
              type="button"
              onClick={() => {
                setMode(mode === "signin" ? "signup" : "signin");
                setErrors({});
                setConfirm("");
              }}
              className="font-semibold text-primary hover:underline"
            >
              {mode === "signin" ? "Create an account" : "Sign in"}
            </button>
          </p>
        </div>
        <p className="mt-4 text-center text-[11px] text-muted-foreground">
          Free forever up to 500 messages · No credit card required
        </p>
      </div>

      <ForgotPasswordModal
        open={forgotOpen}
        initialEmail={email}
        onClose={() => setForgotOpen(false)}
      />
    </div>
  );
}
