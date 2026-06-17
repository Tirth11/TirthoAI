import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Sparkles, Loader2, Mail, Lock, Eye, EyeOff, Gift } from "lucide-react";
import { toast } from "sonner";
import { setRemember } from "@/lib/remember-me";
import { ForgotPasswordModal } from "@/components/ForgotPasswordModal";

interface AuthScreenProps {
  initialMode?: "signin" | "signup";
  onContinueAsGuest?: () => void;
}

export function AuthScreen({ initialMode = "signup", onContinueAsGuest }: AuthScreenProps = {}) {
  const [mode, setMode] = useState<"signin" | "signup">(initialMode);
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
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitErrorHint, setSubmitErrorHint] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [showResend, setShowResend] = useState(false);
  const [resending, setResending] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(0);

  useEffect(() => {
    if (resendCooldown <= 0) return;
    const t = setInterval(() => {
      setResendCooldown((s) => (s <= 1 ? 0 : s - 1));
    }, 1000);
    return () => clearInterval(t);
  }, [resendCooldown]);

  const resendDisabled = resending || resendCooldown > 0;

  const handleResendVerification = async () => {
    if (resendDisabled) return;
    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      toast.error("Enter your email above first, then tap Resend.");
      return;
    }
    setResending(true);
    try {
      const { error } = await supabase.auth.resend({
        type: "signup",
        email,
        options: { emailRedirectTo: `${window.location.origin}/` },
      });
      if (error) throw error;
      toast.success(`Verification email re-sent to ${email}`);
      setNotice(
        `We re-sent the verification email to ${email}. Check your inbox (and spam folder), click the link, then sign in.`,
      );
      setResendCooldown(60);
    } catch (err) {
      const errAny = err as { message?: string; status?: number; code?: string; name?: string } | null;
      const raw =
        (err instanceof Error && err.message) ||
        (typeof errAny?.message === "string" && errAny.message) ||
        "Couldn't resend verification email";
      const status = (err as { status?: number })?.status;
      const code = (err as { code?: string })?.code;
      const name = (err as { name?: string })?.name;
      const lower = raw.toLowerCase();
      let friendly = raw;
      let category: "rate_limit" | "already_verified" | "network" | "invalid_email" | "unknown" = "unknown";

      const secMatch = raw.match(/after\s+(\d+)\s*seconds?/i);
      if (secMatch) {
        const secs = Math.max(1, parseInt(secMatch[1], 10));
        friendly = `Please wait ${secs}s before requesting another verification email.`;
        setResendCooldown(secs);
        category = "rate_limit";
      } else if (status === 429 || /rate.?limit|too many|for security/i.test(lower)) {
        friendly = "You're sending verification emails too quickly. Please wait a minute and try again.";
        setResendCooldown(60);
        category = "rate_limit";
      } else if (/already.*confirmed|already.*verified/i.test(lower)) {
        friendly = "This email is already verified — try signing in.";
        setShowResend(false);
        category = "already_verified";
      } else if (/network|fetch|load failed|timeout|connection/i.test(lower)) {
        friendly = "We couldn't reach the server. Check your connection and try again.";
        category = "network";
      } else if (/invalid.*email|not.*found|user.*not/i.test(lower)) {
        friendly = "We couldn't find an account for that email. Double-check the address.";
        category = "invalid_email";
      } else {
        // Unknown failure — give a safe, actionable fallback and keep UI consistent.
        friendly =
          "We couldn't resend the verification email right now. Please try again in a minute, or contact support if it keeps failing.";
        // Soft cooldown so users don't spam the button on unknown errors.
        setResendCooldown((prev) => (prev > 0 ? prev : 15));
        category = "unknown";
      }

      // Structured log for debugging without leaking the address into prod logs.
      // eslint-disable-next-line no-console
      console.error("[auth] resend_verification_failed", {
        category,
        status: status ?? null,
        code: code ?? null,
        name: name ?? null,
        rawMessage: raw,
        emailDomain: email.includes("@") ? email.split("@")[1] : null,
        cooldownAfter: resendCooldown,
        timestamp: new Date().toISOString(),
      });

      toast.error(friendly);
      // Keep the inline message in sync with the toast without duplicating
      // the resend button. If a notice (verification prompt) is already
      // visible, update it; otherwise surface a fresh submitError.
      if (notice) {
        setNotice(friendly);
      } else {
        setSubmitError(friendly);
      }
      setShowResend(category !== "already_verified");
    } finally {
      setResending(false);
    }
  };



  const validate = (): boolean => {
    const next: typeof errors = {};
    if (!email) next.email = "Email is required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) next.email = "Enter a valid email";
    if (!password) next.password = "Password is required";
    else if (password.length < 8) next.password = "At least 8 characters";

    if (mode === "signup") {
      if (!confirm) next.confirm = "Please confirm your password";
      else if (confirm !== password) next.confirm = "Passwords do not match";
    }
    setErrors(next);
    return Object.keys(next).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitError(null);
    setSubmitErrorHint(null);
    setNotice(null);
    setShowResend(false);
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
          // Either email confirmation is required, or auto sign-in failed.
          const { error: signInErr } = await supabase.auth.signInWithPassword({
            email,
            password,
          });
          if (signInErr) {
            // Most likely cause: project requires email verification.
            setNotice(
              `We sent a verification email to ${email}. Open it and click the link to activate your account, then sign in. Didn't get it? Check spam, or tap "Resend verification email" below.`,
            );
            setShowResend(true);
            toast.success("Check your inbox to verify your email");
            setMode("signin");
            return;
          }
        }
        toast.success("Welcome! 500 free credits added to your account 🎉");
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        toast.success("Welcome back!");
      }
    } catch (err) {
      const raw = err instanceof Error ? err.message : "Authentication failed";
      const status = (err as { status?: number })?.status;
      const code = (err as { code?: string })?.code ?? "";
      const lower = `${raw} ${code}`.toLowerCase();

      let friendly = raw;
      let hint: string | null = null;

      if (/invalid.*credentials|invalid login|wrong password|bad password/i.test(lower)) {
        friendly = "That email and password don't match.";
        hint = "Double-check your password, or use 'Forgot password?' to reset it.";
      } else if (/email.*not.*confirmed|not.*verified|confirm.*email/i.test(lower)) {
        friendly = "Your email address isn't verified yet.";
        hint = `Check ${email || "your inbox"} for the verification link, or tap "Resend verification email" below.`;
        setShowResend(true);
      } else if (/already.*registered|already.*exists|user.*exists|duplicate/i.test(lower)) {
        friendly = "An account with this email already exists.";
        hint = "Try signing in instead, or use 'Forgot password?' if you don't remember it.";
      } else if (/pwned|leaked|compromis|weak.*password|too.*weak/i.test(lower)) {
        friendly = "That password has appeared in a known data breach.";
        hint = "Choose a stronger, unique password (mix letters, numbers, and symbols).";
      } else if (/password.*should be at least|at least 8|minimum.*length/i.test(lower)) {
        friendly = "Your password is too short.";
        hint = "Use at least 8 characters.";
      } else if (/rate.?limit|too many|temporarily/i.test(lower) || status === 429) {
        friendly = "Too many attempts — please wait a minute and try again.";
      } else if (/network|fetch.*failed|load failed|timeout/i.test(lower)) {
        friendly = "We couldn't reach the server.";
        hint = "Check your internet connection and try again.";
      } else if (/invalid.*email/i.test(lower)) {
        friendly = "That email address looks invalid.";
        hint = "Double-check for typos.";
      } else if (/signup.*disabled|signups.*not.*allowed/i.test(lower)) {
        friendly = "New signups are temporarily disabled.";
      }

      setSubmitError(friendly);
      setSubmitErrorHint(hint);
      toast.error(friendly);
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
          <h1 className="text-2xl font-bold tracking-tight">TirthoAI — Your Multi-Model AI Workspace</h1>
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
                  placeholder="Password (min 8 characters)"
                  required
                  minLength={8}

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
              {mode === "signup" && !errors.password && (
                <ul className="mt-1.5 space-y-0.5 text-[11px] text-muted-foreground">
                  <li>• At least 8 characters</li>
                  <li>• Letters, numbers, and symbols (!@#$%^&* …) allowed</li>
                  <li>• Avoid common or previously breached passwords</li>
                </ul>
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
                    minLength={8}
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

            {notice && (
              <div
                role="status"
                className="rounded-lg border border-primary/30 bg-primary/5 px-3 py-2 text-[12px] text-foreground"
              >
                <div className="mb-0.5 font-semibold text-primary">Verify your email</div>
                <div className="leading-snug">{notice}</div>
                {showResend && (
                  <button
                    type="button"
                    onClick={handleResendVerification}
                    disabled={resendDisabled}
                    aria-disabled={resendDisabled}
                    data-testid="resend-verification-btn"
                    className="mt-2 inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1 text-[11px] font-semibold text-primary hover:bg-primary/15 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {resending && <Loader2 className="h-3 w-3 animate-spin" />}
                    {resending
                      ? "Resending…"
                      : resendCooldown > 0
                        ? `Resend in ${resendCooldown}s`
                        : "Resend verification email"}
                  </button>
                )}
              </div>
            )}

            {submitError && (
              <div
                role="alert"
                className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-[12px] text-destructive"
              >
                <div className="font-semibold mb-0.5">
                  {mode === "signin" ? "Couldn't sign you in" : "Couldn't create your account"}
                </div>
                <div className="leading-snug">{submitError}</div>
                {submitErrorHint && (
                  <div className="mt-1 text-[11px] text-destructive/80">{submitErrorHint}</div>
                )}
                {showResend && (
                  <button
                    type="button"
                    onClick={handleResendVerification}
                    disabled={resendDisabled}
                    aria-disabled={resendDisabled}
                    data-testid="resend-verification-btn"
                    className="mt-2 inline-flex items-center gap-1.5 rounded-md border border-destructive/40 bg-destructive/10 px-2.5 py-1 text-[11px] font-semibold text-destructive hover:bg-destructive/15 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {resending && <Loader2 className="h-3 w-3 animate-spin" />}
                    {resending
                      ? "Resending…"
                      : resendCooldown > 0
                        ? `Resend in ${resendCooldown}s`
                        : "Resend verification email"}
                  </button>
                )}
              </div>
            )}


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
        {onContinueAsGuest && (
          <div className="mt-4 flex items-center gap-3">
            <div className="h-px flex-1 bg-border" />
            <span className="text-[11px] uppercase tracking-wider text-muted-foreground">or</span>
            <div className="h-px flex-1 bg-border" />
          </div>
        )}
        {onContinueAsGuest && (
          <button
            type="button"
            onClick={onContinueAsGuest}
            className="mt-3 w-full rounded-lg border border-border bg-background px-4 py-2.5 text-sm font-semibold text-foreground transition hover:bg-accent"
          >
            Try free without an account · 50 messages
          </button>
        )}
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
