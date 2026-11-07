import {
  normalizeGeneration,
  type GenerationSettings,
} from "@/stores/settings";

/** Serialize only supported generation fields, including the selected State. */
export function generationBody(v: GenerationSettings) {
  return normalizeGeneration(v);
}
