# VAFT documentation site (`gh-pages`)

This branch is the **source of the VAFT documentation site** published at
<https://vest-tokamak.github.io/vaft/>.

It contains only the Jekyll site. **The VAFT library itself lives on the
[`main` branch](https://github.com/VEST-Tokamak/vaft/tree/main)** — install
instructions, source, and notebooks belong there, not here. Do not duplicate
library documentation in this README; document the library *on the site*.

The site uses the remote theme [`sighingnow/jekyll-gitbook`](https://github.com/sighingnow/jekyll-gitbook)
(GitBook look: fixed sidebar + per-page table of contents).

## Layout

| Path | Purpose |
| --- | --- |
| `_guide/` | The `guide` collection — the documentation chapters that make up the sidebar. One file per chapter. |
| `_pages/` | The `pages` collection (About, Contact). |
| `_others/`, `_posts/` | Theme demo/legacy content, kept for reference. |
| `index.markdown` | Site landing page (`layout: home`). |
| `assets/` | Images and theme assets. Site images live in `assets/images/`. |
| `_config.yml` | Jekyll config: collections, permalinks, kramdown/MathJax, remote theme. |
| `Gemfile` | Ruby dependencies for local preview. |

**Sidebar order is `date` ascending.** Guide pages currently run from `09:00`
(Installation) to `11:20` (API reference) on the same day — to insert a chapter,
give it a time that slots between its neighbours. To reorder chapters, change
their `date`.

**URLs** come from `permalink: /:collection/:title/`, where `:title` is the file
basename, case preserved: `_guide/Installation.md` → `/vaft/guide/Installation/`.
The one exception is `_guide/Examples.md`, which pins `permalink: /guide/examples/`
in its front matter (`guide/Examples.md` is a redirect stub kept for the old
capitalised URL). Cross-link internally with `{{ site.baseurl }}/guide/<Basename>/`
— never hardcode the domain.

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

```bash
bundle install
bundle exec jekyll serve
```

Then open <http://localhost:4000/vaft/> (the `baseurl` is `/vaft`, so the trailing
path matters). Pushing to `gh-pages` triggers the GitHub Pages build, which
deploys the live site — there is no Actions workflow.

## Notes

- The site index is `index.markdown`; this README is not published as a page.
- `remote_theme` is resolved by the GitHub Pages build. It is not backed by a gem
  in the `Gemfile`, so a local `jekyll serve` renders from the `_layouts/` and
  `_includes/` vendored in this branch. Expect minor differences from production,
  and change those two directories only deliberately — they affect every page.
