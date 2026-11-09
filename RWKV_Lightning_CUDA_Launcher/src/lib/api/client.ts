import { tNow } from "@/lib/i18n";

/** Pure request-body helpers for the native inference API. */

/**
 * A model may emit an adapter ID we should reuse for a checkpoint filename.
 * `.pth` is stripped; an empty result falls back to `adapter`.
 */
export function adapterIDFromFilename(filename: string) {
  const name = filename
    .trim()
    .replace(/\.pth$/i, "")
    .trim();
  return name || "adapter";
}

/**
 * Adapter fields are only sent when an adapter is actually selected; an
 * explicit `0` scale must survive, an unset one must be omitted entirely.
 */
export function adapterFields(v: {
  adapter_id?: string;
  adapter_version?: string;
  adapter_scale?: string;
}) {
  if (!v.adapter_id?.trim()) return {};
  const scale = v.adapter_scale?.trim();
  if (scale && !Number.isFinite(Number(scale)))
    throw new Error(tNow("error.adapterScale"));
  return {
    adapter_id: v.adapter_id.trim(),
    ...(v.adapter_version?.trim()
      ? { adapter_version: v.adapter_version.trim() }
      : {}),
    ...(scale ? { adapter_scale: Number(scale) } : {}),
  };
}

/** Flatten the UI-shaped adapter triple into wire fields. */
export function generationBody<
  T extends {
    adapter_id?: string;
    adapter_version?: string;
    adapter_scale?: string;
  },
>(v: T) {
  const { adapter_id, adapter_version, adapter_scale, ...rest } = v;
  return {
    ...rest,
    ...adapterFields({ adapter_id, adapter_version, adapter_scale }),
  };
}
