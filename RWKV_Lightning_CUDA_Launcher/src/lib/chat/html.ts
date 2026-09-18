const htmlFence =
  /(?:^|\n)[\t ]*```(?:html|htm)[^\S\r\n]*\r?\n([\s\S]*?)(?:\r?\n)?[\t ]*```(?=\r?\n|$)/gi;

type LooseHTMLResponse = {
  html: string;
  remainder: string;
};

function parseLooseHTMLResponse(markdown: string): LooseHTMLResponse | null {
  const source = markdown.replace(/^\uFEFF/, "");
  const start = source.match(
    /^[\t \r\n]*(?:>[\t ]*)?(?=(?:<!doctype\s+html\b|<html(?:\s|>)))/i,
  );
  if (!start) return null;

  const documentAndRemainder = source.slice(start[0].length);
  const closingTags = Array.from(
    documentAndRemainder.matchAll(/<\/html\s*>/gi),
  );
  const closingTag = closingTags.at(-1);
  if (!closingTag || closingTag.index === undefined) {
    return { html: documentAndRemainder.trim(), remainder: "" };
  }

  const documentEnd = closingTag.index + closingTag[0].length;
  const html = documentAndRemainder.slice(0, documentEnd).trim();
  const remainder = documentAndRemainder
    .slice(documentEnd)
    // Some models omit the opening fence but still emit its closing fence.
    .replace(/^[\t ]*(?:\r?\n[\t ]*)?```[^\S\r\n]*(?:\r?\n|$)/, "")
    .trim();
  return { html, remainder };
}

export function extractStandaloneHTMLDocument(markdown: string) {
  return parseLooseHTMLResponse(markdown)?.html ?? null;
}

export function extractHTMLDocuments(markdown: string) {
  const documents: string[] = [];
  for (const match of markdown.matchAll(htmlFence)) {
    const html = match[1].trim();
    if (html) documents.push(html);
  }
  if (documents.length === 0) {
    const raw = extractStandaloneHTMLDocument(markdown);
    if (raw) documents.push(raw);
  }
  return documents;
}

/**
 * A model may return a complete HTML document without a Markdown code fence.
 * Feeding that directly to a Markdown parser makes the doctype and tags render
 * as unrelated HTML blocks. Add a fence for display only; the original HTML is
 * still used by the preview and copy actions.
 */
export function formatHTMLForMarkdown(markdown: string) {
  const response = parseLooseHTMLResponse(markdown);
  if (!response) return markdown;

  let longestBacktickRun = 0;
  for (const match of response.html.matchAll(/`+/g))
    longestBacktickRun = Math.max(longestBacktickRun, match[0].length);
  const fence = "`".repeat(Math.max(3, longestBacktickRun + 1));
  const htmlBlock = `${fence}html\n${response.html}\n${fence}`;
  return response.remainder
    ? `${htmlBlock}\n\n${response.remainder}`
    : htmlBlock;
}

const escapeAttribute = (value: string) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");

export function buildHTMLPreviewDocument(html: string) {
  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="referrer" content="no-referrer">
  <title>RWKV HTML Preview</title>
  <style>
    html, body, iframe { width: 100%; height: 100%; margin: 0; border: 0; }
    body { overflow: hidden; background: #fff; }
  </style>
</head>
<body>
  <iframe title="Generated HTML preview" sandbox="allow-scripts allow-forms allow-modals" referrerpolicy="no-referrer" srcdoc="${escapeAttribute(html)}"></iframe>
</body>
</html>`;
}

export function openHTMLPreview(html: string) {
  const url = URL.createObjectURL(
    new Blob([buildHTMLPreviewDocument(html)], {
      type: "text/html;charset=utf-8",
    }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
