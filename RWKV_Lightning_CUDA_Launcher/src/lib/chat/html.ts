const htmlFence =
  /(?:^|\n)[\t ]*```(?:html|htm)[^\S\r\n]*\r?\n([\s\S]*?)(?:\r?\n)?[\t ]*```(?=\r?\n|$)/gi;

export function extractHTMLDocuments(markdown: string) {
  const documents: string[] = [];
  for (const match of markdown.matchAll(htmlFence)) {
    const html = match[1].trim();
    if (html) documents.push(html);
  }
  if (documents.length === 0) {
    const raw = markdown.trim();
    if (/^(?:<!doctype\s+html\b|<html(?:\s|>))/i.test(raw)) documents.push(raw);
  }
  return documents;
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
