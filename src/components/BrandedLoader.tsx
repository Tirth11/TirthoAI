import { Sparkles } from "lucide-react";

interface Props {
  label?: string;
  fullScreen?: boolean;
}

/**
 * Branded loading state used at the auth gate and during conversation
 * hydration. Visually continues from the pre-hydration #boot-splash so the
 * handoff between the static HTML splash and the React app is seamless.
 */
export function BrandedLoader({ label = "Loading your workspace…", fullScreen = true }: Props) {
  return (
    <div
      className={
        fullScreen
          ? "flex h-screen w-full items-center justify-center bg-background"
          : "flex h-full w-full items-center justify-center"
      }
      role="status"
      aria-live="polite"
    >
      <div className="flex flex-col items-center gap-4">
        <div
          className="flex h-14 w-14 items-center justify-center rounded-2xl text-white shadow-xl animate-pulse"
          style={{ background: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
        >
          <Sparkles className="h-7 w-7" />
        </div>
        <div className="text-center">
          <div className="text-base font-semibold tracking-tight text-foreground">TirthoAI</div>
          <div className="mt-1 text-xs text-muted-foreground">{label}</div>
        </div>
        <div
          className="relative h-1 w-40 overflow-hidden rounded-full bg-muted"
          aria-hidden="true"
        >
          <div
            className="absolute inset-y-0 left-0 w-1/3 rounded-full animate-[shimmer_1.4s_ease-in-out_infinite]"
            style={{ background: "var(--gradient-primary)" }}
          />
        </div>
      </div>
    </div>
  );
}
