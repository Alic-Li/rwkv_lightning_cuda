export function chunkText(input: string): string[] {
  const text = input.replace(/\r\n?/g, "\n").trim();
  if (!text) return [];
  // Each non-empty input line is an independent translation unit. This keeps
  // user-authored paragraph boundaries deterministic and easy to inspect.
  return text
    .split(/\n+/u)
    .map((line) => line.trim())
    .filter(Boolean);
}
export function translationPrompt(source: string, from: string, to: string) {
  return `${from.trim()}: ${source.trim()}\n\n${to.trim()}:`;
}
