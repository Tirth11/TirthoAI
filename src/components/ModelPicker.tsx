import { useState, useRef, useEffect } from "react";
import { Check, ChevronDown, Zap, Plus } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { MODELS, CATEGORY_META, type ModelCategory, getModelById } from "@/lib/models";
import { USER_MODEL_PREFIX, presetFor, type UserModelDTO } from "@/lib/user-models-shared";
import { useUserModels } from "@/hooks/use-user-models";
import { cn } from "@/lib/utils";

interface Props {
  modelId: string;
  onChange: (id: string) => void;
  autoMode: boolean;
  onAutoToggle: (v: boolean) => void;
  /** When true, hide custom user models (e.g. guest mode). */
  hideUserModels?: boolean;
}

export function ModelPicker({ modelId, onChange, autoMode, onAutoToggle, hideUserModels }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { models: userModels } = useUserModels();
  const enabledUserModels = hideUserModels ? [] : userModels.filter((m) => m.enabled);

  const userMatch = modelId.startsWith(USER_MODEL_PREFIX)
    ? userModels.find((m) => `${USER_MODEL_PREFIX}${m.id}` === modelId)
    : undefined;
  const selectedBuiltIn = getModelById(modelId);
  const selectedLabel = userMatch?.label ?? selectedBuiltIn?.label ?? "Select model";
  const selectedBadge = userMatch ? presetFor(userMatch.provider)?.badge ?? "🧩" : selectedBuiltIn?.badge;

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const categories = Object.keys(CATEGORY_META) as ModelCategory[];

  return (
    <div className="flex min-w-0 items-center gap-1.5 sm:gap-2" ref={ref}>
      <button
        onClick={() => onAutoToggle(!autoMode)}
        className={cn(
          "flex shrink-0 items-center gap-1.5 rounded-lg border px-2 py-1.5 text-xs font-semibold transition sm:px-2.5",
          autoMode
            ? "border-primary bg-primary/15 text-primary"
            : "border-border bg-card text-muted-foreground hover:border-primary/40"
        )}
        title="Auto-pick the best model for each prompt"
      >
        <Zap className="h-3 w-3" />
        Auto
      </button>

      <div className="relative">
        <button
          onClick={() => setOpen((v) => !v)}
          disabled={autoMode}
          data-testid="model-picker-trigger"
          data-model-id={modelId}
          className={cn(
            "flex max-w-[44vw] items-center gap-1.5 rounded-lg border border-border bg-card px-2 py-1.5 text-xs font-medium transition hover:border-primary/40 sm:max-w-none sm:gap-2 sm:px-3",
            autoMode && "opacity-60 cursor-not-allowed"
          )}
        >
          <span>{selectedBadge}</span>
          <span className="max-w-[84px] truncate sm:max-w-[160px]">{selectedLabel}</span>
          <ChevronDown className="h-3 w-3 text-muted-foreground" />
        </button>


        {open && (
          <div className="fixed inset-x-3 top-16 z-50 max-h-[70dvh] overflow-y-auto rounded-xl border border-border bg-popover p-2 shadow-2xl sm:absolute sm:inset-auto sm:right-0 sm:top-auto sm:mt-2 sm:w-80 sm:max-h-[60vh]">
            {enabledUserModels.length > 0 && (
              <div className="mb-3">
                <div className="flex items-center justify-between px-2 py-1.5">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                    🧩 Your models
                  </span>
                  <Link
                    to="/settings"
                    onClick={() => setOpen(false)}
                    className="text-[10px] font-semibold text-primary hover:underline"
                  >
                    Manage
                  </Link>
                </div>
                {enabledUserModels.map((m) => (
                  <UserModelRow
                    key={m.id}
                    model={m}
                    active={modelId === `${USER_MODEL_PREFIX}${m.id}`}
                    onPick={() => {
                      onChange(`${USER_MODEL_PREFIX}${m.id}`);
                      setOpen(false);
                    }}
                  />
                ))}
              </div>
            )}

            {categories.map((cat) => {
              const meta = CATEGORY_META[cat];
              const items = MODELS.filter((m) => m.category === cat);
              return (
                <div key={cat} className="mb-2">
                  <div className="px-2 py-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                    {meta.icon} {meta.label}
                  </div>
                  {items.map((m) => {
                    const active = m.id === modelId && m.label === selectedBuiltIn?.label;
                    return (
                      <button
                        key={m.label}
                        data-testid="model-option"
                        data-model-id={m.id}
                        onClick={() => {
                          onChange(m.id);
                          setOpen(false);
                        }}

                        className={cn(
                          "flex w-full items-start gap-2 rounded-lg px-2 py-2 text-left text-xs transition",
                          active ? "bg-accent" : "hover:bg-accent/60"
                        )}
                      >
                        <span className="text-base leading-none">{m.badge}</span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="font-semibold text-foreground truncate">{m.label}</span>
                            {m.supportsVision && (
                              <span className="rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[9px] font-bold uppercase text-emerald-500">
                                Vision
                              </span>
                            )}
                          </div>
                          <p className="text-muted-foreground line-clamp-1">{m.description}</p>
                        </div>
                        {active && <Check className="h-3.5 w-3.5 text-primary mt-0.5" />}
                      </button>
                    );
                  })}
                </div>
              );
            })}

            {!hideUserModels && (
              <Link
                to="/settings"
                onClick={() => setOpen(false)}
                className="mt-1 flex items-center justify-center gap-1.5 rounded-lg border border-dashed border-border px-2 py-2 text-[11px] font-medium text-muted-foreground transition hover:border-primary/50 hover:text-primary"
              >
                <Plus className="h-3 w-3" />
                Add your own model
              </Link>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function UserModelRow({
  model,
  active,
  onPick,
}: {
  model: UserModelDTO;
  active: boolean;
  onPick: () => void;
}) {
  const preset = presetFor(model.provider);
  return (
    <button
      onClick={onPick}
      className={cn(
        "flex w-full items-start gap-2 rounded-lg px-2 py-2 text-left text-xs transition",
        active ? "bg-accent" : "hover:bg-accent/60",
      )}
    >
      <span className="text-base leading-none">{preset?.badge ?? "🧩"}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="font-semibold text-foreground truncate">{model.label}</span>
          <span className="rounded-full border border-border bg-muted px-1.5 py-0.5 text-[9px] font-bold uppercase text-muted-foreground">
            {preset?.label ?? model.provider}
          </span>
        </div>
        <p className="text-muted-foreground line-clamp-1 font-mono text-[10px]">{model.model_id}</p>
      </div>
      {active && <Check className="h-3.5 w-3.5 text-primary mt-0.5" />}
    </button>
  );
}
