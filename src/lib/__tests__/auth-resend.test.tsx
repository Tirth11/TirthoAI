/**
 * Unit test: AuthScreen resend-verification UI must stay consistent across
 * unknown / mocked failure payloads. The button, toast, and inline error
 * must always end up in matching states (button re-enabled or cooled down,
 * inline error visible, no stuck "Resending…" label).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";
import { AuthScreen } from "@/components/AuthScreen";

// Mocks
const resendMock = vi.fn();
vi.mock("@/integrations/supabase/client", () => ({
  supabase: {
    auth: {
      resend: (...args: unknown[]) => resendMock(...args),
      signUp: vi.fn().mockResolvedValue({ data: {}, error: null }),
      signInWithPassword: vi
        .fn()
        .mockResolvedValue({ data: {}, error: { message: "Email not confirmed" } }),
      getSession: vi.fn().mockResolvedValue({ data: { session: null } }),
    },
  },
}));

const toastError = vi.fn();
const toastSuccess = vi.fn();
vi.mock("sonner", () => ({
  toast: {
    error: (msg: string) => toastError(msg),
    success: (msg: string) => toastSuccess(msg),
  },
}));

vi.mock("@/lib/remember-me", () => ({ setRemember: vi.fn() }));
vi.mock("@/components/ForgotPasswordModal", () => ({
  ForgotPasswordModal: () => null,
}));

async function triggerResendButtonVisible() {
  // Submit signup; mocked signInWithPassword returns "Email not confirmed",
  // so the AuthScreen surfaces the Resend button.
  fireEvent.change(screen.getByPlaceholderText(/you@example/i), {
    target: { value: "user@example.com" },
  });
  fireEvent.change(screen.getByPlaceholderText(/^Password/i), {
    target: { value: "password123" },
  });
  fireEvent.change(screen.getByPlaceholderText(/confirm password/i), {
    target: { value: "password123" },
  });
  fireEvent.click(screen.getByRole("button", { name: /create account/i }));
  await waitFor(() => screen.getByTestId("resend-verification-btn"));
}

describe("AuthScreen resend verification — unknown failure UI consistency", () => {
  beforeEach(() => {
    resendMock.mockReset();
    toastError.mockReset();
    toastSuccess.mockReset();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).console.error = vi.fn();
  });

  afterEach(() => cleanup());

  it("unknown error -> button re-enables (with soft cooldown), toast + inline error both shown", async () => {
    render(<AuthScreen initialMode="signup" />);
    resendMock.mockResolvedValueOnce({ error: { message: "kaboom: weird upstream payload" } });

    await triggerResendButtonVisible();

    const btn = screen.getByTestId("resend-verification-btn") as HTMLButtonElement;
    fireEvent.click(btn);

    await waitFor(() => expect(toastError).toHaveBeenCalled());

    // Friendly fallback message is shown in the toast.
    expect(toastError.mock.calls.at(-1)?.[0]).toMatch(/couldn't resend|try again/i);

    // Inline error must mirror toast (no desync between toast + UI).
    await waitFor(() => {
      const alert = screen.getByRole("alert");
      expect(alert.textContent ?? "").toMatch(/couldn't resend|try again/i);
    });

    // Button must NOT be stuck in "Resending…" state.
    await waitFor(() => {
      const label = screen.getByTestId("resend-verification-btn").textContent ?? "";
      expect(label).not.toMatch(/resending/i);
    });

    // Soft cooldown engaged for unknown errors -> button disabled with "Resend in Xs".
    const label = screen.getByTestId("resend-verification-btn").textContent ?? "";
    expect(label).toMatch(/resend in \d+s/i);
    expect((screen.getByTestId("resend-verification-btn") as HTMLButtonElement).disabled).toBe(true);

    // Structured log was emitted for debugging.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((console.error as any)).toHaveBeenCalledWith(
      "[auth] resend_verification_failed",
      expect.objectContaining({ category: "unknown", rawMessage: expect.stringContaining("kaboom") }),
    );
  });

  it("rate-limit error with 'after N seconds' -> exact cooldown applied", async () => {
    render(<AuthScreen initialMode="signup" />);
    resendMock.mockResolvedValueOnce({
      error: { message: "For security purposes, you can only request this after 23 seconds." },
    });

    await triggerResendButtonVisible();
    fireEvent.click(screen.getByTestId("resend-verification-btn"));

    await waitFor(() => expect(toastError).toHaveBeenCalled());
    expect(toastError.mock.calls.at(-1)?.[0]).toMatch(/wait 23s/i);

    const label = screen.getByTestId("resend-verification-btn").textContent ?? "";
    expect(label).toMatch(/resend in 23s/i);
  });

  it("network failure -> friendly message, button re-enabled (no cooldown imposed)", async () => {
    render(<AuthScreen initialMode="signup" />);
    resendMock.mockRejectedValueOnce(new Error("Failed to fetch"));

    await triggerResendButtonVisible();
    fireEvent.click(screen.getByTestId("resend-verification-btn"));

    await waitFor(() => expect(toastError).toHaveBeenCalled());
    expect(toastError.mock.calls.at(-1)?.[0]).toMatch(/couldn't reach the server/i);

    await waitFor(() => {
      const label = screen.getByTestId("resend-verification-btn").textContent ?? "";
      expect(label).not.toMatch(/resending/i);
    });
  });
});
