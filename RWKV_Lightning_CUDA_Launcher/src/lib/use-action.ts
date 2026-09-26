import { useCallback } from "react";
import { tNow } from "@/lib/i18n";
import { toast } from "@/stores/ui";

/**
 * Run a node command and report the outcome. Every Start / Stop / Apply
 * button in the console had its own copy of the same try / catch / toast
 * block, which is how two of them ended up phrasing failures differently.
 *
 * Returns the command's value, or `undefined` when it threw — callers that
 * only care about success can ignore it.
 */
export function useAction() {
  return useCallback(
    async <T>(
      action: () => Promise<T>,
      success?: string,
    ): Promise<T | undefined> => {
      try {
        const result = await action();
        if (success) toast.success(success);
        return result;
      } catch (error) {
        toast.error(
          tNow("toast.failed", {
            error: error instanceof Error ? error.message : String(error),
          }),
        );
        return undefined;
      }
    },
    [],
  );
}
