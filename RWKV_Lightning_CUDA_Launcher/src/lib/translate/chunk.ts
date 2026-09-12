export function chunkText(input: string, target = 800): string[] {
  if (!Number.isInteger(target) || target < 32)
    throw new Error("Chunk target must be an integer ≥ 32");
  const text = input.replace(/\r\n?/g, "\n").trim();
  if (!text) return [];
  const units = text.match(
    /[^.!?。！？\n]+[.!?。！？]*[ \t]*\n*|[.!?。！？]+\s*|\n+/gu,
  ) || [text];
  const chunks: string[] = [];
  let current = "";
  for (const unit of units) {
    // Prefer complete sentences. Long unpunctuated spans fall back to whitespace, then Unicode code points.
    const pieces =
      Array.from(unit).length > target * 2
        ? unit.match(/\S+\s*|\s+/gu) || [unit]
        : [unit];
    for (const piece of pieces) {
      if (current && current.length + piece.length > target) {
        chunks.push(current);
        current = "";
      }
      if (Array.from(piece).length > target * 2) {
        const points = Array.from(piece);
        while (points.length > target) {
          chunks.push(points.splice(0, target).join(""));
        }
        current += points.join("");
      } else current += piece;
    }
  }
  if (current) chunks.push(current);
  return chunks;
}
export function translationPrompt(source: string, from: string, to: string) {
  return `${from.trim()}: ${source.trim()}\n\n${to.trim()}:`;
}
