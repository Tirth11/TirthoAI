// Generic client-side error reporting (replaces the Lovable cloud reporter).
// Logs to the console by default. To forward errors to a real service (Sentry,
// PostHog, etc.), wire it up inside reportClientError.

export function reportClientError(error: unknown, context: Record<string, unknown> = {}) {
  if (typeof window === "undefined") return;
  // eslint-disable-next-line no-console
  console.error("[client-error]", {
    route: window.location.pathname,
    ...context,
    error,
  });
}
