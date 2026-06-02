import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { ArrowLeft, Loader2, Plus, Trash2, KeyRound, User, Cpu, ExternalLink } from "lucide-react";
import { toast } from "sonner";

import { useAuthSession } from "@/hooks/use-auth";
import { useProfile } from "@/hooks/use-profile";
import { useUserModels, type AddModelInput } from "@/hooks/use-user-models";
import { supabase } from "@/integrations/supabase/client";
import { BrandedLoader } from "@/components/BrandedLoader";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  PROVIDER_PRESETS,
  presetFor,
  type UserModelCategory,
  type UserModelProvider,
} from "@/lib/user-models-shared";
import { CATEGORY_META } from "@/lib/models";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/settings")({
  component: SettingsPage,
  ssr: false,
});

function SettingsPage() {
  const { session, loading } = useAuthSession();
  const navigate = useNavigate();

  useEffect(() => {
    if (!loading && !session) navigate({ to: "/" });
  }, [loading, session, navigate]);

  if (loading || !session) return <BrandedLoader label="Loading settings…" />;

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <header
        className="sticky top-0 z-10 flex items-center gap-3 border-b border-border bg-background/80 px-3 py-3 backdrop-blur sm:px-6"
        style={{
          paddingTop: "max(0.75rem, env(safe-area-inset-top))",
          paddingLeft: "max(0.75rem, env(safe-area-inset-left))",
          paddingRight: "max(0.75rem, env(safe-area-inset-right))",
        }}
      >
        <Link
          to="/"
          className="rounded-lg p-1.5 text-muted-foreground transition hover:bg-accent hover:text-foreground"
          aria-label="Back"
        >
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <h1 className="text-base font-semibold sm:text-lg">Settings</h1>
      </header>

      <main className="mx-auto w-full max-w-3xl px-3 py-5 sm:px-6 sm:py-8">
        <Tabs defaultValue="profile" className="w-full">
          <TabsList className="mb-6 grid w-full grid-cols-2 sm:w-auto sm:inline-grid">
            <TabsTrigger value="profile" className="gap-1.5">
              <User className="h-3.5 w-3.5" /> Profile
            </TabsTrigger>
            <TabsTrigger value="models" className="gap-1.5">
              <Cpu className="h-3.5 w-3.5" /> Models
            </TabsTrigger>
          </TabsList>

          <TabsContent value="profile">
            <ProfileTab email={session.user.email ?? ""} />
          </TabsContent>
          <TabsContent value="models">
            <ModelsTab />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

/* ---------------------- Profile tab ---------------------- */

function ProfileTab({ email }: { email: string }) {
  const { profile, loading, save } = useProfile();
  const [displayName, setDisplayName] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const [avatarBusy, setAvatarBusy] = useState(false);

  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [savingPassword, setSavingPassword] = useState(false);

  useEffect(() => {
    if (profile) setDisplayName(profile.display_name ?? "");
  }, [profile]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-muted-foreground">
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
    );
  }

  const initials = (profile?.display_name || email || "U").slice(0, 1).toUpperCase();

  const handleAvatarPick = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      toast.error("Please pick an image file");
      return;
    }
    setAvatarBusy(true);
    try {
      const dataUrl = await downscaleImage(file, 256, 0.85);
      if (dataUrl.length > 400_000) {
        toast.error("Image still too large after compression — try a smaller photo");
        return;
      }
      await save({ avatar_url: dataUrl });
      toast.success("Avatar updated");
    } catch (e) {
      console.error(e);
      toast.error("Couldn't update avatar");
    } finally {
      setAvatarBusy(false);
    }
  };

  const handleSaveName = async (e: React.FormEvent) => {
    e.preventDefault();
    const name = displayName.trim();
    if (!name) {
      toast.error("Display name can't be empty");
      return;
    }
    setSavingProfile(true);
    try {
      await save({ display_name: name });
      toast.success("Profile saved");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSavingProfile(false);
    }
  };

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword.length < 6) {
      toast.error("Password must be at least 6 characters");
      return;
    }
    if (newPassword !== confirmPassword) {
      toast.error("Passwords don't match");
      return;
    }
    setSavingPassword(true);
    try {
      const { error } = await supabase.auth.updateUser({ password: newPassword });
      if (error) throw error;
      toast.success("Password updated");
      setNewPassword("");
      setConfirmPassword("");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Couldn't update password");
    } finally {
      setSavingPassword(false);
    }
  };

  return (
    <div className="space-y-8">
      <SectionCard title="Profile" description="How you appear in TirthoAI.">
        <div className="flex flex-col items-start gap-4 sm:flex-row sm:items-center">
          <Avatar className="h-20 w-20 ring-2 ring-border">
            {profile?.avatar_url && <AvatarImage src={profile.avatar_url} alt="Avatar" />}
            <AvatarFallback
              className="text-xl font-bold text-white"
              style={{ background: "var(--gradient-primary)" }}
            >
              {initials}
            </AvatarFallback>
          </Avatar>
          <div className="flex flex-col gap-2">
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void handleAvatarPick(f);
                if (fileRef.current) fileRef.current.value = "";
              }}
            />
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={avatarBusy}
                onClick={() => fileRef.current?.click()}
              >
                {avatarBusy && <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />}
                {profile?.avatar_url ? "Change photo" : "Upload photo"}
              </Button>
              {profile?.avatar_url && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  disabled={avatarBusy}
                  onClick={async () => {
                    setAvatarBusy(true);
                    try {
                      await save({ avatar_url: null });
                      toast.success("Avatar removed");
                    } finally {
                      setAvatarBusy(false);
                    }
                  }}
                >
                  Remove
                </Button>
              )}
            </div>
            <p className="text-xs text-muted-foreground">PNG or JPG · auto-resized to 256px.</p>
          </div>
        </div>

        <form onSubmit={handleSaveName} className="mt-6 space-y-4">
          <div>
            <Label htmlFor="email">Email</Label>
            <Input id="email" value={email} disabled className="mt-1.5 bg-muted" />
          </div>
          <div>
            <Label htmlFor="displayName">Display name</Label>
            <Input
              id="displayName"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              maxLength={60}
              className="mt-1.5"
              placeholder="Your name"
            />
          </div>
          <div>
            <Button type="submit" disabled={savingProfile}>
              {savingProfile && <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />}
              Save changes
            </Button>
          </div>
        </form>
      </SectionCard>

      <SectionCard
        title="Password"
        description="Update the password used to sign in to TirthoAI."
        icon={<KeyRound className="h-4 w-4" />}
      >
        <form onSubmit={handleChangePassword} className="space-y-4">
          <div>
            <Label htmlFor="newPassword">New password</Label>
            <Input
              id="newPassword"
              type="password"
              autoComplete="new-password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="Min 6 characters"
              className="mt-1.5"
            />
          </div>
          <div>
            <Label htmlFor="confirmPassword">Confirm new password</Label>
            <Input
              id="confirmPassword"
              type="password"
              autoComplete="new-password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="mt-1.5"
            />
          </div>
          <Button type="submit" disabled={savingPassword || !newPassword}>
            {savingPassword && <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />}
            Update password
          </Button>
        </form>
      </SectionCard>
    </div>
  );
}

/* ---------------------- Models tab ---------------------- */

function ModelsTab() {
  const {
    models,
    loading,
    addModel,
    updateModel,
    deleteModelOptimistic,
  } = useUserModels();
  const [showForm, setShowForm] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<{ id: string; label: string } | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [pendingOps, setPendingOps] = useState<Set<string>>(new Set());

  // Draft toggles: id -> desired enabled value (only present when it differs from server state).
  const [draftToggles, setDraftToggles] = useState<Record<string, boolean>>({});
  const [showSaveConfirm, setShowSaveConfirm] = useState(false);
  const [savingToggles, setSavingToggles] = useState(false);

  // Drop draft entries that already match the server (e.g. after a refresh).
  useEffect(() => {
    setDraftToggles((prev) => {
      const next: Record<string, boolean> = {};
      let changed = false;
      for (const [id, val] of Object.entries(prev)) {
        const m = models.find((x) => x.id === id);
        if (m && m.enabled !== val) {
          next[id] = val;
        } else {
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [models]);

  const addPending = (id: string) =>
    setPendingOps((prev) => new Set(prev).add(id));
  const removePending = (id: string) =>
    setPendingOps((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });

  const effectiveEnabled = (m: { id: string; enabled: boolean }) =>
    draftToggles[m.id] ?? m.enabled;

  const draftEntries = Object.entries(draftToggles);
  const pendingChangeCount = draftEntries.length;

  const handleToggleDraft = (id: string, serverValue: boolean, checked: boolean) => {
    setDraftToggles((prev) => {
      const next = { ...prev };
      if (checked === serverValue) {
        delete next[id];
      } else {
        next[id] = checked;
      }
      return next;
    });
  };

  const performDelete = async (id: string, label: string) => {
    addPending(id);
    try {
      await deleteModelOptimistic(id);
      toast.success(`Deleted "${label}"`);
    } catch (err) {
      toast.error(`Couldn't delete "${label}"`, {
        description: "Check your connection and try again.",
        action: {
          label: "Retry",
          onClick: () => {
            void performDelete(id, label);
          },
        },
      });
    } finally {
      removePending(id);
    }
  };

  const handleConfirmDelete = async () => {
    if (!pendingDelete) return;
    const { id, label } = pendingDelete;
    setDeleting(true);
    setPendingDelete(null);
    try {
      await performDelete(id, label);
    } finally {
      setDeleting(false);
    }
  };

  const handleSaveToggles = async () => {
    const entries = Object.entries(draftToggles);
    if (entries.length === 0) {
      setShowSaveConfirm(false);
      return;
    }
    setSavingToggles(true);
    try {
      const results = await Promise.allSettled(
        entries.map(([id, enabled]) => updateModel({ id, enabled })),
      );
      const failed: [string, boolean][] = [];
      results.forEach((r, i) => {
        if (r.status === "rejected") failed.push(entries[i]);
      });
      if (failed.length === 0) {
        toast.success(
          `Saved ${entries.length} model change${entries.length === 1 ? "" : "s"}`,
        );
        setDraftToggles({});
        setShowSaveConfirm(false);
      } else {
        const failedIds = new Set(failed.map(([id]) => id));
        setDraftToggles((prev) => {
          const next: Record<string, boolean> = {};
          for (const [id, val] of Object.entries(prev)) {
            if (failedIds.has(id)) next[id] = val;
          }
          return next;
        });
        setShowSaveConfirm(false);
        const failedLabels = failed
          .map(([id]) => models.find((m) => m.id === id)?.label ?? id)
          .join(", ");
        toast.error(
          `Couldn't save ${failed.length} of ${entries.length} change${entries.length === 1 ? "" : "s"}`,
          {
            description: `Still pending: ${failedLabels}`,
            action: {
              label: "Retry",
              onClick: () => {
                void handleSaveToggles();
              },
            },
          },
        );
      }
    } finally {
      setSavingToggles(false);
    }
  };

  return (
    <div className="space-y-6">
      <SectionCard
        title="Your models"
        description="Bring your own OpenAI-compatible LLMs. Keys are encrypted per-user and only used when you select the model."
        rightSlot={
          <Button size="sm" onClick={() => setShowForm((v) => !v)}>
            <Plus className="mr-1 h-3.5 w-3.5" />
            {showForm ? "Close" : "Add model"}
          </Button>
        }
      >
        {showForm && (
          <div className="mb-6">
            <AddModelForm
              onCancel={() => setShowForm(false)}
              onSubmit={async (input) => {
                try {
                  await addModel(input);
                  toast.success(`Added ${input.label}`);
                  setShowForm(false);
                } catch (err) {
                  toast.error(err instanceof Error ? err.message : "Couldn't add model");
                }
              }}
            />
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        ) : models.length === 0 ? (
          <p className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-sm text-muted-foreground">
            No custom models yet. Add one to make it appear in the chat model picker.
          </p>
        ) : (
          <>
            <ul className="space-y-2">
              {models.map((m) => {
                const preset = presetFor(m.provider);
                const checked = effectiveEnabled(m);
                const isDirty = draftToggles[m.id] !== undefined;
                return (
                  <li
                    key={m.id}
                    className={cn(
                      "flex flex-col gap-3 rounded-lg border bg-card p-3 sm:flex-row sm:items-center",
                      isDirty ? "border-primary/50 ring-1 ring-primary/30" : "border-border",
                    )}
                  >
                    <div className="flex min-w-0 flex-1 items-center gap-3">
                      <span className="text-xl" aria-hidden>
                        {preset?.badge ?? "🧩"}
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="truncate text-sm font-semibold">{m.label}</span>
                          <span className="rounded-full border border-border bg-muted px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                            {preset?.label ?? m.provider}
                          </span>
                          <span
                            className={cn(
                              "rounded-full bg-gradient-to-r px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white",
                              CATEGORY_META[m.category].color,
                            )}
                          >
                            {CATEGORY_META[m.category].label}
                          </span>
                          {isDirty && (
                            <span className="rounded-full bg-primary/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-primary">
                              Unsaved
                            </span>
                          )}
                        </div>
                        <div className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
                          {m.model_id} · key {m.key_hint ?? "—"}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 self-end sm:self-auto">
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={checked}
                          disabled={pendingOps.has(m.id) || savingToggles}
                          onCheckedChange={(v) => handleToggleDraft(m.id, m.enabled, v)}
                          aria-label={checked ? "Disable model" : "Enable model"}
                        />
                        <span className="text-xs text-muted-foreground">
                          {pendingOps.has(m.id) ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : checked ? (
                            "On"
                          ) : (
                            "Off"
                          )}
                        </span>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                        disabled={pendingOps.has(m.id) || savingToggles}
                        onClick={() => setPendingDelete({ id: m.id, label: m.label })}
                        aria-label="Delete model"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </li>
                );
              })}
            </ul>

            {pendingChangeCount > 0 && (
              <div className="sticky bottom-3 mt-4 flex flex-col gap-2 rounded-lg border border-primary/40 bg-card/95 p-3 shadow-lg backdrop-blur sm:flex-row sm:items-center sm:justify-between">
                <p className="text-sm text-foreground">
                  <span className="font-semibold">{pendingChangeCount}</span> unsaved{" "}
                  {pendingChangeCount === 1 ? "change" : "changes"}
                </p>
                <div className="flex gap-2 self-end sm:self-auto">
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={savingToggles}
                    onClick={() => setDraftToggles({})}
                  >
                    Discard
                  </Button>
                  <Button
                    size="sm"
                    disabled={savingToggles}
                    onClick={() => setShowSaveConfirm(true)}
                  >
                    {savingToggles && <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />}
                    Save changes
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </SectionCard>

      <AlertDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open && !deleting) setPendingDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this model?</AlertDialogTitle>
            <AlertDialogDescription>
              {pendingDelete
                ? `"${pendingDelete.label}" will be removed from your model picker. Its API key will be permanently deleted. This can't be undone.`
                : ""}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                e.preventDefault();
                void handleConfirmDelete();
              }}
              disabled={deleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleting && <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog
        open={showSaveConfirm}
        onOpenChange={(open) => {
          if (!open && !savingToggles) setShowSaveConfirm(false);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Save {pendingChangeCount} change{pendingChangeCount === 1 ? "" : "s"}?
            </AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div>
                <p className="mb-2">These models will be updated in your chat picker:</p>
                <ul className="space-y-1 text-sm">
                  {draftEntries.map(([id, enabled]) => {
                    const m = models.find((x) => x.id === id);
                    if (!m) return null;
                    return (
                      <li key={id} className="flex items-center justify-between gap-2">
                        <span className="truncate font-medium text-foreground">{m.label}</span>
                        <span
                          className={cn(
                            "rounded-full px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider",
                            enabled
                              ? "bg-primary/15 text-primary"
                              : "bg-muted text-muted-foreground",
                          )}
                        >
                          {enabled ? "Enable" : "Disable"}
                        </span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={savingToggles}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                e.preventDefault();
                void handleSaveToggles();
              }}
              disabled={savingToggles}
            >
              {savingToggles && <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />}
              Save changes
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

const CATEGORY_OPTS: UserModelCategory[] = ["general", "reasoning", "coding", "creative", "vision"];

function AddModelForm({
  onSubmit,
  onCancel,
}: {
  onSubmit: (input: AddModelInput) => Promise<void>;
  onCancel: () => void;
}) {
  const [provider, setProvider] = useState<UserModelProvider>("openai");
  const preset = presetFor(provider);
  const [label, setLabel] = useState("");
  const [baseUrl, setBaseUrl] = useState(preset?.baseUrl ?? "");
  const [modelId, setModelId] = useState(preset?.sampleModelId ?? "");
  const [apiKey, setApiKey] = useState("");
  const [category, setCategory] = useState<UserModelCategory>("general");
  const [busy, setBusy] = useState(false);

  // Sync baseUrl/modelId when preset changes
  useEffect(() => {
    const p = presetFor(provider);
    if (!p) return;
    if (provider !== "custom") {
      setBaseUrl(p.baseUrl);
      setModelId((cur) => cur || p.sampleModelId);
      setLabel((cur) => cur || `${p.label} — ${p.sampleModelId}`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!label.trim() || !baseUrl.trim() || !modelId.trim() || !apiKey.trim()) {
      toast.error("Label, base URL, model ID, and API key are required");
      return;
    }
    setBusy(true);
    try {
      await onSubmit({
        label: label.trim(),
        provider,
        base_url: baseUrl.trim(),
        model_id: modelId.trim(),
        api_key: apiKey,
        category,
      });
      setApiKey("");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-lg border border-border bg-muted/30 p-4 space-y-4"
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <Label htmlFor="provider">Provider</Label>
          <Select value={provider} onValueChange={(v) => setProvider(v as UserModelProvider)}>
            <SelectTrigger id="provider" className="mt-1.5">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PROVIDER_PRESETS.map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.badge} {p.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {preset?.docsUrl && (
            <a
              href={preset.docsUrl}
              target="_blank"
              rel="noreferrer"
              className="mt-1 inline-flex items-center gap-1 text-[11px] text-primary hover:underline"
            >
              Get an API key <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </div>
        <div>
          <Label htmlFor="category">Category</Label>
          <Select value={category} onValueChange={(v) => setCategory(v as UserModelCategory)}>
            <SelectTrigger id="category" className="mt-1.5">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {CATEGORY_OPTS.map((c) => (
                <SelectItem key={c} value={c}>
                  {CATEGORY_META[c].icon} {CATEGORY_META[c].label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div>
        <Label htmlFor="label">Display label</Label>
        <Input
          id="label"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="e.g. GPT-4o Mini (work)"
          className="mt-1.5"
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <Label htmlFor="baseUrl">Base URL</Label>
          <Input
            id="baseUrl"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="https://api.openai.com/v1"
            className="mt-1.5 font-mono text-xs"
          />
        </div>
        <div>
          <Label htmlFor="modelId">Model ID</Label>
          <Input
            id="modelId"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            placeholder="gpt-4o-mini"
            className="mt-1.5 font-mono text-xs"
          />
        </div>
      </div>

      <div>
        <Label htmlFor="apiKey">API key</Label>
        <Input
          id="apiKey"
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="sk-…"
          autoComplete="new-password"
          className="mt-1.5 font-mono text-xs"
        />
        <p className="mt-1 text-[11px] text-muted-foreground">
          Encrypted server-side. Never sent back to the browser.
        </p>
      </div>

      <div className="flex justify-end gap-2">
        <Button type="button" variant="ghost" onClick={onCancel} disabled={busy}>
          Cancel
        </Button>
        <Button type="submit" disabled={busy}>
          {busy && <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />}
          Add model
        </Button>
      </div>
    </form>
  );
}

/* ---------------------- Shared bits ---------------------- */

function SectionCard({
  title,
  description,
  children,
  rightSlot,
  icon,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
  rightSlot?: React.ReactNode;
  icon?: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-border bg-card p-5 shadow-sm sm:p-6">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="flex items-center gap-2 text-base font-semibold sm:text-lg">
            {icon}
            {title}
          </h2>
          {description && (
            <p className="mt-1 text-sm text-muted-foreground">{description}</p>
          )}
        </div>
        {rightSlot}
      </div>
      {children}
    </section>
  );
}

async function downscaleImage(file: File, maxSize: number, quality: number): Promise<string> {
  const bitmap = await createImageBitmap(file);
  const ratio = Math.min(1, maxSize / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * ratio));
  const h = Math.max(1, Math.round(bitmap.height * ratio));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas not supported");
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close?.();
  return canvas.toDataURL("image/jpeg", quality);
}
