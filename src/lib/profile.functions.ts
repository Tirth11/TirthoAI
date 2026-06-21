import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { z } from "zod";

export interface ProfileDTO {
  user_id: string;
  display_name: string | null;
  avatar_url: string | null;
}

const UpdateInput = z.object({
  display_name: z.string().trim().min(1).max(60).optional(),
  // Either a data: URL (base64, capped at ~400KB) or null to clear.
  avatar_url: z
    .string()
    .max(550_000)
    .refine((v) => v === "" || v.startsWith("data:image/"), "Avatar must be a data: URL")
    .nullable()
    .optional(),
});

export const getProfile = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }): Promise<{ profile: ProfileDTO }> => {
    const { supabase, userId } = context;
    const { data, error } = await supabase
      .from("profiles")
      .select("user_id,display_name,avatar_url")
      .eq("user_id", userId)
      .maybeSingle();
    if (error) throw new Error(error.message);
    if (!data) {
      // Defensive: backfill if trigger missed (shouldn't happen)
      const { data: inserted, error: insertErr } = await supabase
        .from("profiles")
        .insert({ user_id: userId })
        .select("user_id,display_name,avatar_url")
        .single();
      if (insertErr) throw new Error(insertErr.message);
      return { profile: inserted as ProfileDTO };
    }
    return { profile: data as ProfileDTO };
  });

export const updateProfile = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: unknown) => UpdateInput.parse(input))
  .handler(async ({ data, context }): Promise<{ profile: ProfileDTO }> => {
    const { supabase, userId } = context;
    const patch: { display_name?: string; avatar_url?: string | null } = {};
    if (data.display_name !== undefined) patch.display_name = data.display_name;
    if (data.avatar_url !== undefined) patch.avatar_url = data.avatar_url;
    const { data: row, error } = await supabase
      .from("profiles")
      .update(patch)
      .eq("user_id", userId)
      .select("user_id,display_name,avatar_url")
      .single();
    if (error) throw new Error(error.message);
    return { profile: row as ProfileDTO };
  });
