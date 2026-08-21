# VAFT documentation site (`gh-pages`)

This branch is the **source of the VAFT documentation site** published at
<https://vest-tokamak.github.io/vaft/>.

It contains only the Jekyll site. **The VAFT library itself is maintained on the
[`develop` branch](https://github.com/VEST-Tokamak/vaft/tree/develop)** — install
instructions, source, and notebooks belong there, not here. Do not duplicate
library documentation in this README; document the library *on the site*.

The site uses the remote theme [`sighingnow/jekyll-gitbook`](https://github.com/sighingnow/jekyll-gitbook)
(GitBook look: fixed sidebar + per-page table of contents).

## Layout

| Path | Purpose |
| --- | --- |
| `_data/navigation.yml` | Explicit titles, stable IDs, order, and canonical URLs for the two flat sidebar sections. |
| `_guide/` | Canonical workflow and reference source pages plus hidden legacy redirect sources. |
| `_pages/` | The `pages` collection (About, Contact). |
| `_others/`, `_posts/` | Theme demo/legacy content, kept for reference. |
| `index.markdown` | Site landing page (`layout: home`). |
| `assets/` | Images and theme assets. Site images live in `assets/images/`. |
| `_config.yml` | Jekyll config: collections, permalinks, kramdown/MathJax, remote theme. |
| `Gemfile` | Ruby dependencies for local preview. |

Sidebar order and canonical URLs come only from `_data/navigation.yml`; page dates
do not control navigation. Cross-link canonical routes with `{{ site.baseurl }}`.
Every retired `/guide/.../` or `/pages/.../` URL must have a redirect document and
an entry in `_data/page_migrations.yml`.

## Adding or editing a guide page

Create `_guide/<Basename>.md` with this front matter:

```yaml
---
title: Human readable title
author: VEST team
date: 2026-07-01 10:20
category: guide
layout: post
---
```

Add `mermaid: true` **only** if the page contains a mermaid fence.

Content is kramdown + GFM: fenced code blocks (``python``, ``bash``, ``text``),
math with `$...$` via MathJax, and images referenced as
`![alt]({{ site.baseurl }}/assets/images/...)`.

## Preview locally

Use a current Ruby rather than the macOS system Ruby. On Apple Silicon, a
one-time Homebrew setup is:

```bash
brew install ruby
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
gem install bundler -v 2.5.16
```

```bash
bundle install
bundle exec jekyll serve --livereload
```

Then open <http://localhost:4000/vaft/> (the `baseurl` is `/vaft`, so the trailing
path matters). Pushing to `gh-pages` triggers the GitHub Pages build, which
deploys the live site — there is no Actions workflow.

## Visual regression checks

The layout suite runs locally against Chromium and WebKit at desktop and mobile
viewports. Install its dependencies and browser engines once:

```bash
npm install
npx playwright install chromium webkit
```

Run the committed snapshots with `npm run test:visual`. After an intentional
layout change, review the rendered site and refresh baselines with
`npm run test:visual:update`.

Run `npm run test:docs` for a clean Jekyll build, internal-link checks, migration
coverage, resource-ID validation, and notebook artifact/provenance SHA checks.
Notebook provenance remains intentionally invalid until the separately authorized
companion-branch commit is pinned with:

```bash
ruby scripts/finalize_notebook_provenance.rb <40-character-commit-sha>
```

## Notes

- The site index is `index.markdown`; this README is not published as a page.
- `remote_theme` is resolved by the GitHub Pages build. It is not backed by a gem
  in the `Gemfile`, so a local `jekyll serve` renders from the `_layouts/` and
  `_includes/` vendored in this branch. Expect minor differences from production,
  and change those two directories only deliberately — they affect every page.
