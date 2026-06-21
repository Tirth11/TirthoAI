import { Sparkles, Gift, X } from "lucide-react";

interface Props {
  open: boolean;
  onClose?: () => void;
  onSignUp: () => void;
  onSignIn: () => void;
  variant?: "exhausted" | "soft";
}

export function SignupPrompt({ open, onClose, onSignUp, onSignIn, variant = "exhausted" }: Props) {
  if (!open) return null;
  const exhausted = variant === "exhausted";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 px-4 backdrop-blur-sm">
      <div className="relative w-full max-w-md rounded-2xl border border-border bg-card p-6 shadow-2xl">
        {onClose && !exhausted && (
          <button
            onClick={onClose}
            className="absolute right-3 top-3 rounded-lg p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        )}
        <div
          className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl text-white shadow-xl"
          style={{ background: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
        >
          <Sparkles className="h-7 w-7" />
        </div>
        <h2 className="text-center text-xl font-bold tracking-tight">
          {exhausted ? "You've used your 50 free messages" : "Keep the conversation going"}
        </h2>
        <p className="mx-auto mt-2 max-w-sm text-center text-sm text-muted-foreground">
          {exhausted
            ? "Create a free account to unlock 500 more credits, save your chat history, and use every model."
            : "Sign up free to save this chat and get 500 credits."}
        </p>

        <div className="mt-5 flex items-center justify-center gap-2 rounded-xl border border-primary/20 bg-primary/5 px-3 py-2 text-xs font-medium text-primary">
          <Gift className="h-3.5 w-3.5" />
          500 free credits on signup · No credit card
        </div>

        <div className="mt-5 space-y-2">
          <button
            type="button"
            onClick={onSignUp}
            className="flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold text-primary-foreground shadow-md transition hover:opacity-95"
            style={{ background: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
          >
            Sign up free
          </button>
          <button
            type="button"
            onClick={onSignIn}
            className="flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-background px-4 py-2.5 text-sm font-semibold text-foreground transition hover:bg-accent"
          >
            I already have an account
          </button>
        </div>
      </div>
    </div>
  );
}
