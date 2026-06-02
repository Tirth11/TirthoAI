import { useCallback, useEffect, useState } from "react";
import { useServerFn } from "@tanstack/react-start";
import {
  listUserModels,
  addUserModel,
  updateUserModel,
  deleteUserModel,
} from "@/lib/user-models.functions";
import type { UserModelDTO } from "@/lib/user-models-shared";

export function useUserModels() {
  const list = useServerFn(listUserModels);
  const add = useServerFn(addUserModel);
  const update = useServerFn(updateUserModel);
  const del = useServerFn(deleteUserModel);

  const [models, setModels] = useState<UserModelDTO[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await list({});
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
    addModel: async (input: Parameters<typeof add>[0]["data"]) => {
      const res = await add({ data: input });
      await refresh();
      return res.model;
    },
    updateModel: async (input: Parameters<typeof update>[0]["data"]) => {
      const res = await update({ data: input });
      await refresh();
      return res.model;
    },
    deleteModel: async (id: string) => {
      await del({ data: { id } });
      await refresh();
    },
  };
}
