# BookCast landing page

Static, dependency-light site for GitHub Pages. No build step: `index.html`,
`assets/styles.css`, and two small scripts.

## Preview locally

From the repository root:

```bash
python -m http.server 8765 -d site
```

Open <http://127.0.0.1:8765/>.

## Before deploying

1. Set the public repository URL in `assets/config.js` so the **Source** links
   appear. Leave it empty to hide them:
   ```js
   window.BOOKCAST_CONFIG = { repoUrl: "https://github.com/your-name/bookcast" };
   ```
2. Replace the Open Graph image at `assets/og-card.png` if the copy changes.
   The editable source is `assets/og-card.svg`.
3. Check `index.html` metadata and the footer wording for the final domain.

## Deploy to GitHub Pages

This repository includes `.github/workflows/pages.yml`, which publishes the
`site/` folder.

1. Create the public GitHub repository and push `main`.
2. In **Settings → Pages**, set **Source** to **GitHub Actions**.
3. Push a change under `site/` (or run the workflow manually). The site will be
   published at `https://<owner>.github.io/<repo>/`.

If you prefer a `docs/` folder instead, move the contents of `site/` to `docs/`
and update the workflow path.

## Files

- `index.html` — the landing page
- `404.html` — branded not-found page
- `assets/styles.css` — layout, type, color, responsive rules
- `assets/app.js` — mobile nav, cast-line highlight, copy buttons, active nav
- `assets/config.js` — deployment settings
- `assets/favicon.svg` — mark
- `assets/og-card.svg` / `.png` — social preview
- `robots.txt` — allow all crawlers
