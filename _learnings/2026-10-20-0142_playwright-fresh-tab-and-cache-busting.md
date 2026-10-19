# Playwright checks of local HTML: navigate in a fresh tab, and bust the cache after rebuilding

**Symptom (checking the five paper explainers, 2026-10-19).**
- `browser_navigate` from an already open page repeatedly timed out after 30 s, and element screenshots then
  timed out waiting for locators; `browser_close` followed by a fresh `browser_navigate` worked every time.
- After rebuilding a page, the browser still ran the old JavaScript: `python -m http.server` lets it cache, so a
  fixed demo appeared unfixed.
- Two element screenshots requested in parallel both timed out; one at a time they worked.

**Fix.** Close the tab before moving to the next page, add a query string (`page.html?v=2`) after every rebuild,
and take screenshots sequentially. Screenshots land in the workspace's `.playwright-mcp/` (gitignored); delete
only the files a check created (`find .playwright-mcp -newermt "<start>"` lists them).

**Consequences.** A width check at 390 px is easiest inside one page with iframes per file; it caught long
`<code>` paths overflowing, a bug in the paper-explainer skill's template (`code` needs `overflow-wrap: anywhere`).
