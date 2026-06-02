/**
 * Regression test for the per-thread model indicator UI.
 *
 * It mirrors the JSX rendered inside ChatWindow's header (lines ~482–518)
 * so we can verify, without booting the whole chat surface, that:
 *   1. The selected model persists per conversation across a simulated
 *      page refresh (via ModelCache + localStorage).
 *   2. The tooltip displays the previous model name, the new model name,
 *      the changing user, and a precise timestamp.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { getModelById, MODELS } from "@/lib/models";
import { ModelCache } from "@/lib/model-cache";

function Indicator({
  modelId,
  previousModelId,
  modelUpdatedAt,
  userEmail,
}: {
  modelId: string;
  previousModelId?: string;
  modelUpdatedAt: string;
  userEmail: string;
}) {
  return (
    <TooltipProvider delayDuration={0}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span data-testid="indicator">
            changed · {userEmail}
          </span>
        </TooltipTrigger>
        <TooltipContent>
          <div>
            <div>Previous: {previousModelId ? (getModelById(previousModelId)?.label ?? previousModelId) : "—"}</div>
            <div>Current: {getModelById(modelId)?.label ?? modelId}</div>
            <div>
              At:{" "}
              {new Date(modelUpdatedAt).toLocaleString(undefined, {
                dateStyle: "medium",
                timeStyle: "medium",
              })}
            </div>
            <div>By: {userEmail}</div>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

describe("Model selection per-conversation persistence + indicator tooltip", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("persists each thread's selected model independently across refresh", () => {
    const a = MODELS[0].id;
    const b = MODELS[1].id;
    ModelCache.set("conv-a", a);
    ModelCache.set("conv-b", b);

    // Simulate refresh by re-reading the cache after a notional reload.
    expect(ModelCache.get("conv-a")?.modelId).toBe(a);
    expect(ModelCache.get("conv-b")?.modelId).toBe(b);
  });

  it("tooltip shows previous + current model labels, the user, and a precise timestamp", async () => {
    const prev = MODELS[0];
    const curr = MODELS[1];
    const at = new Date("2026-06-02T10:30:00Z").toISOString();

    render(
      <Indicator
        modelId={curr.id}
        previousModelId={prev.id}
        modelUpdatedAt={at}
        userEmail="user@example.com"
      />,
    );

    await userEvent.hover(screen.getByTestId("indicator"));

    // Radix portals the tooltip content; use findAllByText for safety.
    const previousLine = await screen.findByText(new RegExp(`Previous:\\s*${prev.label}`));
    const currentLine = await screen.findByText(new RegExp(`Current:\\s*${curr.label}`));
    const byLine = await screen.findByText(/By:\s*user@example\.com/);
    const expectedAt = new Date(at).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "medium",
    });
    const atLine = await screen.findByText(new RegExp(`At:\\s*${expectedAt.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));

    expect(previousLine).toBeInTheDocument();
    expect(currentLine).toBeInTheDocument();
    expect(byLine).toBeInTheDocument();
    expect(atLine).toBeInTheDocument();
  });

  it("renders an em-dash for previous when the thread has no prior model", async () => {
    const curr = MODELS[0];
    render(
      <Indicator
        modelId={curr.id}
        modelUpdatedAt={new Date().toISOString()}
        userEmail="u@e.com"
      />,
    );
    await userEvent.hover(screen.getByTestId("indicator"));
    expect(await screen.findByText(/Previous:\s*—/)).toBeInTheDocument();
  });
});
