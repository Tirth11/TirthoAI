import { useCallback, useEffect, useState } from "react";
import { useServerFn } from "@tanstack/react-start";
import {
  listUserModels,
  addUserModel,
  updateUserModel,
  deleteUserModel,
} from "@/lib/user-models.functions";
import type { UserModelDTO, UserModelCategory, UserModelProvider } from "@/lib/user-models-shared";

export interface AddModelInput {
  label: string;
  provider: UserModelProvider;
  base_url: string;
  model_id: string;
  api_key: string;
  category?: UserModelCategory;
}

export interface UpdateModelInput {
  id: string;
  label?: string;
  base_url?: string;
  model_id?: string;
  api_key?: string;
  category?: UserModelCategory;
  enabled?: boolean;
}

export function useUserModels() {
  const list = useServerFn(listUserModels);
  const add = useServerFn(addUserModel);
  const update = useServerFn(updateUserModel);
  const del = useServerFn(deleteUserModel);

  const [models, setModels] = useState<UserModelDTO[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await list();
      setModels(res.models);
    } catch (e) {
      console.error("listUserModels failed", e);
    } finally {
      setLoading(false);
    }
  }, [list]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    models,
    loading,
    refresh,
    addModel: async (input: AddModelInput) => {
      const res = await add({ data: input });
      await refresh();
      return res.model;
    },
    updateModel: async (input: UpdateModelInput) => {
      const res = await update({ data: input });
      await refresh();
      return res.model;
    },
    /**
     * Optimistically flip `enabled` for a model. Updates local state
     * immediately, then commits to the server. On failure, rolls back
     * and rethrows so the caller can surface a toast.
     */
    toggleEnabledOptimistic: async (id: string, enabled: boolean) => {
      let snapshot: UserModelDTO[] = [];
      setModels((prev) => {
        snapshot = prev;
        return prev.map((m) => (m.id === id ? { ...m, enabled } : m));
      });
      try {
        await update({ data: { id, enabled } });
      } catch (err) {
        setModels(snapshot);
        throw err;
      }
    },
    /**
     * Optimistically remove a model. On failure, restores it and rethrows.
     */
    deleteModelOptimistic: async (id: string) => {
      let snapshot: UserModelDTO[] = [];
      setModels((prev) => {
        snapshot = prev;
        return prev.filter((m) => m.id !== id);
      });
      try {
        await del({ data: { id } });
      } catch (err) {
        setModels(snapshot);
        throw err;
      }
    },
    deleteModel: async (id: string) => {
      await del({ data: { id } });
      await refresh();
    },
  };
}
