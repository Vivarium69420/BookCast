"""BookCast studio visual theme.

The studio is intentionally a light, paper-like workspace: warm neutrals,
deep green actions, and a serif display face. The CSS only styles the shell
and existing Gradio components; all workflow behaviour lives in app.py.
"""
from __future__ import annotations

import gradio as gr

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%231f6f5c'/%3E%3Cpath d='M18 18h12c5 0 9 4 9 9v22H27c-5 0-9-4-9-9V18Z' fill='%23fffdf8'/%3E%3Cpath d='M46 18H34v31h3c5 0 9-4 9-9V18Z' fill='%23e8c58f'/%3E%3C/svg%3E">
"""

HEADER_HTML = """
<div class="bc-hero">
  <div class="bc-hero-top">
    <div class="bc-brand">
      <div class="bc-mark" aria-hidden="true">B</div>
      <div>
        <span class="bc-kicker">BookCast Studio</span>
        <h1>Turn a book into a full-cast audiobook.</h1>
        <p>Open a book, shape the cast, and render chapter by chapter. Every book
        keeps its own settings while one shared Activity panel keeps the queue
        moving.</p>
      </div>
    </div>
    <div class="bc-pills" aria-label="Studio features">
      <span class="bc-pill bc-pill-green">Full cast</span>
      <span class="bc-pill">Voice-aware</span>
      <span class="bc-pill">One shared queue</span>
    </div>
  </div>
  <div class="bc-steps" aria-label="Workflow">
    <div class="bc-step"><b>1</b><span><strong>Open</strong><small>EPUB or PDF</small></span></div>
    <div class="bc-step"><b>2</b><span><strong>Analyze</strong><small>Find the cast</small></span></div>
    <div class="bc-step"><b>3</b><span><strong>Shape</strong><small>Voices &amp; chapters</small></span></div>
    <div class="bc-step"><b>4</b><span><strong>Render</strong><small>MP3 or chapters</small></span></div>
  </div>
</div>
"""

ACTIVITY_HTML = """
<div class="bc-panel-heading">
  <div>
    <span class="bc-kicker">Shared across books</span>
    <h2>Activity</h2>
    <p>Queue, progress, logs, and finished audio.</p>
  </div>
  <span class="bc-live">Live</span>
</div>
"""

RENDER_NOTE_HTML = """
<div class="bc-callout">
  <div class="bc-callout-icon" aria-hidden="true">→</div>
  <div><strong>Ready when the cast looks right.</strong>
  Rendering runs in Activity, so you can keep working in another book while
  this one finishes.</div>
</div>
"""

FOOTER_HTML = """
<div class="bc-footer">
  <strong>BookCast Studio</strong>
  <span>Local-first casting · one queue for every open book</span>
</div>
"""


THEME = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="amber",
    neutral_hue="stone",
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="transparent",
    body_background_fill_dark="transparent",
    body_text_color="#211d18",
    body_text_color_dark="#211d18",
    body_text_color_subdued="#756a5c",
    body_text_color_subdued_dark="#756a5c",
    block_background_fill="#fffdf8",
    block_background_fill_dark="#fffdf8",
    block_border_color="#e3d8c7",
    block_border_color_dark="#e3d8c7",
    block_border_width="1px",
    block_label_background_fill="#fffdf8",
    block_label_background_fill_dark="#fffdf8",
    block_label_text_color="#6f6355",
    block_label_text_color_dark="#6f6355",
    block_title_text_color="#211d18",
    block_title_text_color_dark="#211d18",
    input_background_fill="#fffefb",
    input_background_fill_dark="#fffefb",
    input_border_color="#dcd0bd",
    input_border_color_dark="#dcd0bd",
    input_border_color_focus="#1f6f5c",
    input_border_color_focus_dark="#1f6f5c",
    button_primary_background_fill="#1f6f5c",
    button_primary_background_fill_dark="#1f6f5c",
    button_primary_background_fill_hover="#175948",
    button_primary_background_fill_hover_dark="#175948",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#f1e8da",
    button_secondary_background_fill_dark="#f1e8da",
    button_secondary_background_fill_hover="#e9ddca",
    button_secondary_background_fill_hover_dark="#e9ddca",
    button_secondary_text_color="#3d342a",
    button_secondary_text_color_dark="#3d342a",
    slider_color="#1f6f5c",
    slider_color_dark="#1f6f5c",
    table_even_background_fill="#fffdf8",
    table_even_background_fill_dark="#fffdf8",
    table_odd_background_fill="#faf5ec",
    table_odd_background_fill_dark="#faf5ec",
    table_border_color="#e8dfd1",
    table_border_color_dark="#e8dfd1",
    table_text_color="#4a4035",
    table_text_color_dark="#4a4035",
)

CSS = r"""
:root {
  --bc-bg: #f3eee5;
  --bc-panel: #fffdf8;
  --bc-panel-2: #faf5ec;
  --bc-panel-3: #f5eddf;
  --bc-ink: #211d18;
  --bc-muted: #756a5c;
  --bc-faint: #a4998b;
  --bc-line: #e3d8c7;
  --bc-line-strong: #d3c4ae;
  --bc-green: #1f6f5c;
  --bc-green-dark: #175948;
  --bc-green-soft: #e5f1eb;
  --bc-amber: #bd7422;
  --bc-amber-soft: #f8ecd9;
  --bc-rose: #a84b4b;
  --bc-shadow: 0 18px 50px rgba(73, 55, 30, 0.08);
  --bc-radius: 20px;
}

html,
body {
  background: var(--bc-bg) !important;
  color: var(--bc-ink) !important;
}

body {
  overflow-x: hidden;
  background-color: var(--bc-bg) !important;
  background-image:
    radial-gradient(circle at 0% 0%, rgba(189, 116, 34, 0.09), transparent 29%),
    radial-gradient(circle at 100% 8%, rgba(31, 111, 92, 0.08), transparent 31%),
    linear-gradient(180deg, #f8f3ea 0%, #f1ebe1 100%) !important;
  background-attachment: fixed !important;
  font-family: "Inter", ui-sans-serif, system-ui, sans-serif !important;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

::selection {
  background: var(--bc-amber-soft);
  color: var(--bc-ink);
}

footer,
.footer {
  display: none !important;
}

.gradio-container {
  max-width: 1480px !important;
  margin: 0 auto !important;
  padding: 28px 30px 64px !important;
}

.bc-plain {
  width: 100% !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* --------------------------------------------------------------- header */
.bc-hero {
  position: relative;
  width: 100%;
  overflow: hidden;
  padding: 30px 32px 26px;
  border: 1px solid var(--bc-line);
  border-radius: 28px;
  background:
    linear-gradient(135deg, rgba(255, 253, 248, 0.96), rgba(250, 244, 234, 0.90)),
    var(--bc-panel);
  box-shadow: var(--bc-shadow);
}

.bc-hero::after {
  content: "";
  position: absolute;
  right: -90px;
  top: -130px;
  width: 330px;
  height: 330px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(189, 116, 34, 0.15), transparent 68%);
  pointer-events: none;
}

.bc-hero-top {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 28px;
}

.bc-brand {
  display: flex;
  align-items: flex-start;
  gap: 18px;
  min-width: 0;
}

.bc-mark {
  flex: 0 0 auto;
  display: grid;
  place-items: center;
  width: 54px;
  height: 54px;
  border-radius: 17px;
  background: var(--bc-green);
  color: #fffdf8;
  font-family: "Fraunces", Georgia, serif;
  font-size: 25px;
  font-weight: 700;
  box-shadow: 0 10px 22px rgba(31, 111, 92, 0.25);
}

.bc-kicker {
  display: block;
  color: var(--bc-green);
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.19em;
  line-height: 1.3;
  text-transform: uppercase;
}

.bc-hero h1 {
  margin: 5px 0 8px;
  color: var(--bc-ink);
  font-family: "Fraunces", Georgia, serif;
  font-size: clamp(29px, 3vw, 43px);
  font-weight: 700;
  letter-spacing: -0.025em;
  line-height: 1.04;
}

.bc-hero p {
  max-width: 780px;
  margin: 0;
  color: var(--bc-muted);
  font-size: 14.5px;
  line-height: 1.6;
}

.bc-pills {
  display: flex;
  flex: 0 0 auto;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  max-width: 380px;
  padding-top: 4px;
}

.bc-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 11px;
  border: 1px solid var(--bc-line);
  border-radius: 999px;
  background: rgba(255, 253, 248, 0.72);
  color: var(--bc-muted);
  font-size: 11.5px;
  font-weight: 600;
  line-height: 1;
  white-space: nowrap;
}

.bc-pill-green {
  border-color: rgba(31, 111, 92, 0.22);
  background: var(--bc-green-soft);
  color: var(--bc-green-dark);
}

.bc-pill::before {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
  opacity: 0.55;
}

.bc-steps {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-top: 24px;
  padding-top: 18px;
  border-top: 1px solid var(--bc-line);
}

.bc-step {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  padding: 10px 12px;
  border: 1px solid rgba(227, 216, 199, 0.9);
  border-radius: 14px;
  background: rgba(255, 253, 248, 0.6);
}

.bc-step b {
  display: grid;
  flex: 0 0 auto;
  place-items: center;
  width: 25px;
  height: 25px;
  border-radius: 9px;
  background: var(--bc-amber-soft);
  color: var(--bc-amber);
  font-size: 11px;
  font-weight: 700;
}

.bc-step strong,
.bc-step small {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bc-step strong {
  color: var(--bc-ink);
  font-size: 12.5px;
  font-weight: 650;
}

.bc-step small {
  margin-top: 1px;
  color: var(--bc-faint);
  font-size: 10.5px;
}

/* --------------------------------------------------------------- layout */
.bc-shell {
  align-items: flex-start !important;
  gap: 22px !important;
  margin-top: 22px !important;
}

.bc-workspace,
.bc-activity {
  width: 100% !important;
  gap: 16px !important;
}

.bc-activity {
  position: sticky;
  top: 0;
  align-self: flex-start;
}

/* --------------------------------------------------------------- panels */
.bc-card {
  overflow: hidden !important;
  border: 1px solid var(--bc-line) !important;
  border-radius: var(--bc-radius) !important;
  background: rgba(255, 253, 248, 0.90) !important;
  box-shadow: 0 10px 30px rgba(73, 55, 30, 0.045) !important;
}

.bc-card > .label-wrap,
.bc-card > button.label-wrap {
  padding: 17px 20px !important;
  border: 0 !important;
  background: transparent !important;
}

.bc-card > .label-wrap:hover,
.bc-card > button.label-wrap:hover {
  background: rgba(241, 232, 218, 0.55) !important;
}

.bc-card > .label-wrap span,
.bc-card > button.label-wrap span {
  color: var(--bc-ink) !important;
  font-family: "Fraunces", Georgia, serif;
  font-size: 18px !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em;
}

.bc-card .block {
  border-color: var(--bc-line) !important;
}

.bc-card .bc-subcard {
  border: 1px solid var(--bc-line) !important;
  border-radius: 15px !important;
  background: var(--bc-panel-2) !important;
  box-shadow: none !important;
}

.bc-subcard > .label-wrap,
.bc-subcard > button.label-wrap {
  padding: 13px 15px !important;
}

.bc-subcard > .label-wrap span,
.bc-subcard > button.label-wrap span {
  color: var(--bc-muted) !important;
  font-family: "Inter", ui-sans-serif, system-ui, sans-serif;
  font-size: 13px !important;
  font-weight: 650 !important;
}

/* --------------------------------------------------------------- library */
.bc-library {
  width: 100% !important;
  align-items: flex-end !important;
  gap: 10px !important;
  padding: 12px 14px !important;
  border: 1px solid var(--bc-line) !important;
  border-radius: 18px !important;
  background: rgba(255, 253, 248, 0.76) !important;
  box-shadow: 0 8px 22px rgba(73, 55, 30, 0.035) !important;
}

.bc-library .book-rail {
  min-height: 0 !important;
  margin: 0 !important;
}

.bc-library .book-rail label {
  margin: 0 0 5px !important;
  color: var(--bc-muted) !important;
  font-size: 11px !important;
  font-weight: 650 !important;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.bc-library button {
  min-height: 42px;
}

/* --------------------------------------------------------------- content */
.bc-book-card.block {
  padding: 16px 18px !important;
  border: 1px solid var(--bc-line) !important;
  border-radius: 16px !important;
  background:
    linear-gradient(135deg, rgba(229, 241, 235, 0.62), rgba(255, 253, 248, 0.88)) !important;
  box-shadow: none !important;
}

.bc-book-card .prose {
  color: var(--bc-ink);
  font-size: 14px;
  line-height: 1.55;
}

.bc-book-card .prose p:last-child,
.bc-book-card .prose ul:last-child {
  margin-bottom: 0;
}

.prose h1,
.prose h2,
.prose h3,
.prose h4 {
  color: var(--bc-ink);
  font-family: "Fraunces", Georgia, serif;
  letter-spacing: -0.015em;
}

.prose p,
.prose li {
  color: var(--bc-muted);
}

.prose code {
  padding: 2px 5px;
  border-radius: 6px;
  background: var(--bc-panel-3);
  color: var(--bc-green-dark);
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 0.88em;
}

.prose a {
  color: var(--bc-green) !important;
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
}

.bc-callout {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 14px 16px;
  border: 1px solid rgba(189, 116, 34, 0.18);
  border-radius: 15px;
  background: var(--bc-amber-soft);
  color: #6f4a1d;
  font-size: 13px;
  line-height: 1.5;
}

.bc-callout strong {
  display: block;
  margin-bottom: 2px;
  color: #5d3c15;
  font-size: 13px;
}

.bc-callout-icon {
  display: grid;
  flex: 0 0 auto;
  place-items: center;
  width: 30px;
  height: 30px;
  border-radius: 10px;
  background: rgba(189, 116, 34, 0.14);
  color: var(--bc-amber);
  font-size: 15px;
}

.bc-render-btn {
  min-height: 50px !important;
  font-size: 15px !important;
  font-weight: 700 !important;
  letter-spacing: -0.005em;
  box-shadow: 0 10px 24px rgba(31, 111, 92, 0.18) !important;
}

.bc-render-btn:hover {
  transform: translateY(-1px);
}

/* --------------------------------------------------------------- inputs */
.bc-workspace .block label > span,
.bc-activity .block label > span {
  color: var(--bc-ink);
  font-size: 12.5px;
  font-weight: 600;
}

.bc-workspace input,
.bc-workspace textarea,
.bc-workspace select,
.bc-activity input,
.bc-activity textarea,
.bc-activity select {
  border-radius: 12px !important;
}

.bc-workspace input:focus,
.bc-workspace textarea:focus,
.bc-activity input:focus,
.bc-activity textarea:focus {
  box-shadow: 0 0 0 3px rgba(31, 111, 92, 0.11) !important;
}

button.primary,
button.secondary,
button.stop {
  border-radius: 12px !important;
  font-weight: 650 !important;
  transition: transform 120ms ease, background-color 120ms ease,
    box-shadow 120ms ease !important;
}

button.primary:hover,
button.secondary:hover,
button.stop:hover {
  transform: translateY(-1px);
}

button.primary {
  box-shadow: 0 7px 18px rgba(31, 111, 92, 0.14) !important;
}

/* --------------------------------------------------------------- tables */
.bc-table {
  overflow: hidden !important;
  border: 1px solid var(--bc-line) !important;
  border-radius: 14px !important;
  background: var(--bc-panel) !important;
}

.bc-table table {
  font-size: 12.5px !important;
}

.bc-table thead th {
  background: var(--bc-panel-2) !important;
  color: var(--bc-muted) !important;
  font-size: 10.5px !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.bc-table tbody tr {
  transition: background-color 100ms ease;
}

.bc-table tbody tr:hover {
  background: rgba(229, 241, 235, 0.48) !important;
}

.bc-table tbody td {
  border-color: rgba(232, 223, 209, 0.7) !important;
}

/* --------------------------------------------------------------- activity */
.bc-panel-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 4px 2px 2px;
}

.bc-panel-heading h2 {
  margin: 2px 0 0;
  color: var(--bc-ink);
  font-family: "Fraunces", Georgia, serif;
  font-size: 27px;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.05;
}

.bc-panel-heading p {
  margin: 4px 0 0;
  color: var(--bc-muted);
  font-size: 12.5px;
}

.bc-live {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  margin-top: 4px;
  padding: 6px 10px;
  border: 1px solid rgba(31, 111, 92, 0.18);
  border-radius: 999px;
  background: var(--bc-green-soft);
  color: var(--bc-green-dark);
  font-size: 11px;
  font-weight: 700;
  line-height: 1;
}

.bc-live::before {
  content: "";
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--bc-green);
  box-shadow: 0 0 0 3px rgba(31, 111, 92, 0.12);
}

.bc-log textarea {
  min-height: 180px !important;
  border-color: #2b2722 !important;
  border-radius: 14px !important;
  background: #211d18 !important;
  color: #e9e0d2 !important;
  font-family: "JetBrains Mono", ui-monospace, monospace !important;
  font-size: 11.5px !important;
  line-height: 1.55 !important;
}

.bc-log label > span {
  color: var(--bc-muted) !important;
}

.bc-footer {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: 30px;
  padding: 18px 4px 0;
  border-top: 1px solid var(--bc-line);
  color: var(--bc-faint);
  font-size: 11.5px;
}

.bc-footer strong {
  color: var(--bc-muted);
  font-family: "Fraunces", Georgia, serif;
  font-size: 14px;
  font-weight: 600;
}

/* --------------------------------------------------------------- motion */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
  }
}

/* --------------------------------------------------------------- narrow */
@media (max-width: 1180px) {
  .bc-shell {
    flex-direction: column !important;
  }

  .bc-workspace,
  .bc-activity {
    min-width: 0 !important;
  }

  .bc-activity {
    position: static;
  }
}

@media (max-width: 860px) {
  .gradio-container,
  .gradio-container .app {
    box-sizing: border-box !important;
    width: 100% !important;
    max-width: 100% !important;
  }

  .gradio-container {
    padding: 14px 10px 48px !important;
  }

  .gradio-container .app {
    padding-left: 10px !important;
    padding-right: 10px !important;
  }

  .bc-hero {
    padding: 22px 20px;
    border-radius: 22px;
  }

  .bc-hero-top {
    flex-direction: column;
    gap: 18px;
  }

  .bc-pills {
    justify-content: flex-start;
    max-width: none;
    padding-top: 0;
  }

  .bc-steps {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 540px) {
  .bc-library {
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 8px !important;
  }

  .bc-library > * {
    width: 100% !important;
  }

  .bc-brand {
    gap: 12px;
  }

  .bc-mark {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    font-size: 21px;
  }

  .bc-hero {
    padding: 18px 16px 16px;
  }

  .bc-hero h1 {
    font-size: 27px;
  }

  .bc-hero p {
    font-size: 13.5px;
  }

  .bc-pills {
    display: none;
  }

  .bc-steps {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 18px;
    padding-top: 14px;
  }

  .bc-step {
    padding: 9px 10px;
  }

  .bc-step small {
    display: none;
  }

  .bc-footer {
    display: block;
    line-height: 1.7;
  }
}
"""

# Inject globally rather than through Blocks.css, which scopes selectors to
# the inner .contain element and cannot reach the page shell.
HEAD = HEAD + "\n<style>\n" + CSS + "\n</style>\n"
