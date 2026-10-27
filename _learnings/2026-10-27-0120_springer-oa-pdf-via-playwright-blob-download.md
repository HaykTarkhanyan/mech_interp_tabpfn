# Springer open-access PDFs: download through Playwright's download event, not curl or page.evaluate

**Symptom** (2026-10-24, *Through the Looking Glass*, Int. J. Data Sci. Anal., doi 10.1007/s41060-023-00445-1):
the article is open access, but `curl` on `https://link.springer.com/content/pdf/10.1007/s41060-023-00445-1.pdf`
returned `text/html; charset=utf-8` (a page, not a PDF).

**What failed inside Playwright MCP's `browser_run_code_unsafe`:**
- `page.request.get(...)` then `await import('node:fs')` to write the bytes: `ERR_VM_DYNAMIC_IMPORT_CALLBACK_MISSING`
  (the sandbox cannot import Node modules);
- returning the bytes as base64 from `page.evaluate`: a 1 MB PDF is far over the tool's 30,000-character output
  limit, so it comes back truncated.

**What worked:** open the article page first (so cookies are set), fetch the PDF inside the page, turn it into a
blob URL, click a temporary `<a download>`, and save Playwright's `download` event:

```js
async (page) => {
  const info = await page.evaluate(async () => {
    const r = await fetch('/content/pdf/10.1007/s41060-023-00445-1.pdf', { credentials: 'include' });
    const blob = await r.blob();
    const head = await blob.slice(0, 5).text();          // must be "%PDF-"
    window.__pdfBlobUrl = URL.createObjectURL(blob);
    return { status: r.status, type: r.headers.get('content-type'), size: blob.size, head };
  });
  if (info.head !== '%PDF-') return { ...info, saved: false };
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.evaluate(() => { const a = document.createElement('a'); a.href = window.__pdfBlobUrl;
                          a.download = 'x.pdf'; document.body.appendChild(a); a.click(); }),
  ]);
  await download.saveAs('C:/full/path/paper.pdf');
  return { ...info, saved: true };
}
```

**Consequences:** check the first five bytes for `%PDF-` before trusting any "downloaded" paper, because publishers
answer with HTML under a `.pdf` URL. The same pattern should work for other cookie-gated open-access PDFs.
