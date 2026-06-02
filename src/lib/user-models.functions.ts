// Server functions for managing user-added LLM models.
// Keys are AES-GCM encrypted at rest and never returned to the client.
import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { z } from "zod";
import {
  PROVIDER_PRESETS,
  type UserModelDTO,
} from "./user-models-shared";

const VALID_PROVIDERS = PROVIDER_PRESETS.map((p) => p.id) as [string, ...string[]];
const VALID_CATEGORIES = ["reasoning", "coding", "creative", "vision", "general"] as const;

const AddInput = z.object({
  label: z.string().trim().min(1).max(80),
  provider: z.enum(VALID_PROVIDERS),
  base_url: z.string().trim().url().max(500),
  model_id: z.string().trim().min(1).max(200),
  api_key: z.string().trim().min(4).max(500),
  category: z.enum(VALID_CATEGORIES).default("general"),
});

const UpdateInput = z.object({
  id: z.string().uuid(),
  label: z.string().trim().min(1).max(80).optional(),
  base_url: z.string().trim().url().max(500).optional(),
  model_id: z.string().trim().min(1).max(200).optional(),
  api_key: z.string().trim().min(4).max(500).optional(),
  category: z.enum(VALID_CATEGORIES).optional(),
  enabled: z.boolean().optional(),
});

type Row = {
  id: string;
  label: string;
  provider: string;
  base_url: string;
  model_id: string;
  api_key_ciphertext: string | null;
  category: string;
  enabled: boolean;
  created_at: string;
};

async function rowToDTO(row: Row): Promise<UserModelDTO> {
  let hint: string | null = null;
  if (row.api_key_ciphertext) {
    try {
      const { decryptSecret, maskHint } = await import("./secret-crypto.server");
      hint = maskHint(await decryptSecret(row.api_key_ciphertext));
    } catch {
      hint = "••••";
    }
  }
  return {
    id: row.id,
    label: row.label,
    provider: row.provider as UserModelDTO["provider"],
    base_url: row.base_url,
    model_id: row.model_id,
    category: row.category as UserModelDTO["category"],
    enabled: row.enabled,
    has_key: !!row.api_key_ciphertext,
    key_hint: hint,
    created_at: row.created_at,
  };
}

export const listUserModels = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }): Promise<{ models: UserModelDTO[] }> => {
    const { supabase } = context;
    const { data, error } = await supabase
      .from("user_models")
      .select("id,label,provider,base_url,model_id,api_key_ciphertext,category,enabled,created_at")
      .order("created_at", { ascending: false });
    if (error) throw new Error(error.message);
    const models = await Promise.all(((data ?? []) as Row[]).map(rowToDTO));
    return { models };
  });

export const addUserModel = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: unknown) => AddInput.parse(input))
  .handler(async ({ data, context }): Promise<{ model: UserModelDTO }> => {
    const { supabase, userId } = context;
    const { encryptSecret } = await import("./secret-crypto.server");
    const ciphertext = await encryptSecret(data.api_key);
    const { data: row, error } = await supabase
      .from("user_models")
      .insert({
        user_id: userId,
        label: data.label,
        provider: data.provider,
        base_url: data.base_url,
        model_id: data.model_id,
        api_key_ciphertext: ciphertext,
        category: data.category,
        enabled: true,
      })
      .select("id,label,provider,base_url,model_id,api_key_ciphertext,category,enabled,created_at")
      .single();
    if (error) throw new Error(error.message);
    return { model: await rowToDTO(row as Row) };
  });

export const updateUserModel = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: unknown) => UpdateInput.parse(input))
  .handler(async ({ data, context }): Promise<{ model: UserModelDTO }> => {
    const { supabase } = context;
    const patch: {
      label?: string;
      base_url?: string;
      model_id?: string;
      category?: string;
      enabled?: boolean;
      api_key_ciphertext?: string;
    } = {};
    if (data.label !== undefined) patch.label = data.label;
    if (data.base_url !== undefined) patch.base_url = data.base_url;
    if (data.model_id !== undefined) patch.model_id = data.model_id;
    if (data.category !== undefined) patch.category = data.category;
    if (data.enabled !== undefined) patch.enabled = data.enabled;
    if (data.api_key !== undefined) {
      const { encryptSecret } = await import("./secret-crypto.server");
      patch.api_key_ciphertext = await encryptSecret(data.api_key);
    }
    const { data: row, error } = await supabase
      .from("user_models")
      .update(patch)
      .eq("id", data.id)
      .select("id,label,provider,base_url,model_id,api_key_ciphertext,category,enabled,created_at")
      .single();
    if (error) throw new Error(error.message);
    return { model: await rowToDTO(row as Row) };
  });

export const deleteUserModel = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: unknown) => z.object({ id: z.string().uuid() }).parse(input))
  .handler(async ({ data, context }): Promise<{ ok: true }> => {
    const { supabase } = context;
    const { error } = await supabase.from("user_models").delete().eq("id", data.id);
    if (error) throw new Error(error.message);
    return { ok: true };
  });
