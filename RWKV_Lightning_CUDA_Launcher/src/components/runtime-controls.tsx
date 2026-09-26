import { Play, RotateCcw, Square } from "lucide-react";
import { Button, type ButtonProps } from "@/components/ui/button";
import { applyDevice } from "@/lib/api/launcher";
import { useAction } from "@/lib/use-action";
import { useI18n } from "@/lib/i18n";
import { useCurrent } from "@/app/use-current";
import { useRuntimeFormEntry } from "@/stores/forms";
import { useNodes } from "@/stores/nodes";

/**
 * Which of Start / Stop / Restart are usable, from two independent facts the
 * Agent reports — neither of which answers the question on its own:
 *
 *   managed — this launcher owns the process (p.cmd != nil). True from the
 *             moment it spawns, so it covers the whole cold-load window.
 *   running — something is serving the port. True both for a runtime we own
 *             and for one started outside this launcher, and false while ours
 *             is still loading a model.
 *
 * Stop and Restart therefore follow `managed`: is there a process of ours to
 * act on. Gating them on `running` left a multi-GB cold load uninterruptible,
 * because a loading runtime reports running=false for minutes.
 *
 * Start follows both: refuse when a process of ours exists (it would answer
 * "backend is already running") and when something else already holds the
 * port.
 *
 * Exported because it is the rule worth pinning in tests — the component
 * itself cannot be rendered with seeded store state, since zustand serves
 * `getInitialState` to `renderToStaticMarkup`.
 */
export function runtimeButtonState(input: {
  reachableAgent: boolean;
  busy: boolean;
  managed: boolean;
  running: boolean;
  canStart: boolean;
}) {
  const { reachableAgent, busy, managed, running, canStart } = input;
  const blocked = !reachableAgent || busy;
  return {
    busy,
    managed,
    running,
    disabled: !reachableAgent,
    canStart,
    startDisabled: blocked || managed || running || !canStart,
    stopDisabled: blocked || !managed,
    restartDisabled: blocked || !managed,
  };
}

function useActions() {
  const { t } = useI18n();
  const { backendId, hasAgent, runtime, backend, busy } = useCurrent();
  const { config: form, devices } = useRuntimeFormEntry(backendId);
  const start = useNodes((s) => s.startRuntime);
  const stop = useNodes((s) => s.stopRuntime);
  const restart = useNodes((s) => s.restartRuntime);
  const run = useAction();

  return {
    t,
    ...runtimeButtonState({
      // An inference-only or unreachable node cannot be controlled at all.
      reachableAgent:
        Boolean(backendId) && hasAgent && Boolean(backend?.reachable),
      busy,
      managed: Boolean(runtime?.managed),
      running: Boolean(runtime?.running),
      canStart: Boolean(form.model_path.trim()),
    }),
    start: () =>
      run(
        () => start(backendId, applyDevice({ ...form }, devices)),
        t("runtime.started"),
      ),
    stop: () => run(() => stop(backendId), t("runtime.stopped")),
    restart: () => run(() => restart(backendId), t("runtime.restarted")),
  };
}

export function RuntimeControls({
  size = "sm",
  className,
}: {
  size?: ButtonProps["size"];
  className?: string;
}) {
  const actions = useActions();
  return (
    <div className={className} data-testid="runtime-controls">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="default"
          size={size}
          disabled={actions.startDisabled}
          onClick={() => void actions.start()}
        >
          <Play className="size-3.5" />
          {actions.t("common.start")}
        </Button>
        <Button
          size={size}
          disabled={actions.stopDisabled}
          onClick={() => void actions.stop()}
        >
          <Square className="size-3.5" />
          {actions.t("common.stop")}
        </Button>
        <Button
          size={size}
          disabled={actions.restartDisabled}
          onClick={() => void actions.restart()}
        >
          <RotateCcw className="size-3.5" />
          {actions.t("common.restart")}
        </Button>
      </div>
    </div>
  );
}
