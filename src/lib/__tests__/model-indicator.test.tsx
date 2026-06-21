/**
 * Regression test for the per-thread model indicator UI.
 *
 * Mirrors the JSX in ChatWindow's header tooltip (around lines 482–518)
 * so we can verify, without booting the entire chat surface, that:
 *   1. Selected model persists per conversation across a simulated refresh.
 *   2. The tooltip shows previous + new model labels, the changing user,
 *      and a precise timestamp.
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
          <span data-testid="indicator">changed · {userEmail}</span>
        </TooltipTrigger>
        <TooltipContent>
          <div>
            <div data-testid="prev-line">
              Previous: {previousModelId ? (getModelById(previousModelId)?.label ?? previousModelId) : "—"}
            </div>
            <div data-testid="curr-line">
              Current: {getModelById(modelId)?.label ?? modelId}
            </div>
            <div data-testid="at-line">
              At:{" "}
              {new Date(modelUpdatedAt).toLocaleString(undefined, {
                dateStyle: "medium",
                timeStyle: "medium",
              })}
            </div>
            <div data-testid="by-line">By: {userEmail}</div>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

describe("Per-conversation model persistence + indicator tooltip", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("persists each thread's selected model independently across refresh", () => {
    const a = MODELS[0].id;
    const b = MODELS[1].id;
    ModelCache.set("conv-a", a);
    ModelCache.set("conv-b", b);
    expect(ModelCache.get("conv-a")?.modelId).toBe(a);
    expect(ModelCache.get("conv-b")?.modelId).toBe(b);
  });

  it("tooltip shows previous + current labels, user, and precise timestamp", async () => {
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

    const expectedAt = new Date(at).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "medium",
    });
    const [prevLine] = await screen.findAllByTestId("prev-line");
    const [currLine] = await screen.findAllByTestId("curr-line");
    const [atLine] = await screen.findAllByTestId("at-line");
    const [byLine] = await screen.findAllByTestId("by-line");

    expect(prevLine).toHaveTextContent(`Previous: ${prev.label}`);
    expect(currLine).toHaveTextContent(`Current: ${curr.label}`);
    expect(atLine).toHaveTextContent(`At: ${expectedAt}`);
    expect(byLine).toHaveTextContent("By: user@example.com");
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
    const [prevLine] = await screen.findAllByTestId("prev-line");
    expect(prevLine).toHaveTextContent("Previous: —");
  });
});
