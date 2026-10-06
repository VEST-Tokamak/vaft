# VAFT documentation source

This directory is the source of the documentation site published at
<https://vest-tokamak.github.io/vaft/>. It is a Jekyll site using the GitBook
look of [`sighingnow/jekyll-gitbook`](https://github.com/sighingnow/jekyll-gitbook),
whose layouts and includes are vendored here and locally modified.

## Two tracks, one published branch

The site has two tracks, and each is built from the branch it documents:

| URL | Built from | Purpose |
| --- | --- | --- |
| `/vaft/` | `main/docs/` | stable documentation |
| `/vaft/develop/` | `develop/docs/` | development documentation, with a banner and `noindex` |

Both are composed into one tree and published as a single commit on the
**`gh-pages`** branch. That branch is generated output only: every file on it is
rewritten on each publish, so an edit made there survives until the next build
and reaches nobody's review. Change documentation here, on the branch it belongs
to.

Building each track from its own branch is the point rather than a detail. The
generated reference pages are built by introspecting the library, so they are
only correct for the exact tree they were generated from. Which ones a track has
is whatever its own `generators.yml` declares:

| Generated page | Generator | `main` | `develop` |
| --- | --- | --- | --- |
| `/reference/vest-diagnostics/` | `vaft.machine_mapping.registry` | yes | yes |
| `/reference/formula/` | `vaft.formula.catalog` | yes | yes |
| `/reference/process/` | `vaft.process.catalog` | yes | yes |
| `/reference/plot/` | `vaft.plot.docs_catalog` | no | yes |
| `/reference/diagram/` | `vaft.diagram.docs_catalog` | no | yes |
| `/reference/api/<page>/` | `vaft._api_catalog` | no | yes |
| `/reference/dependency-graph/` | `vaft._dependency_graph` | no | yes |
| `/reference/pipeline-graph/` | `vaft._pipeline_graph` | no | yes |
| `/reference/software-dependencies/`, `/reference/external-codes/` | `vaft._ecosystem_catalog` | no | yes |

`vaft._dependency_graph` (#1646) needs Grimp, the optional `architecture`
extra (`pip install -e ".[architecture]"`; also in `[dev]`). Without it the
generator stops with that install hint instead of falling back to another
analyser, and `import vaft` never needs it. `--check` compares an existing
snapshot with what the tree derives. The page renders it with the shared graph
viewer in `assets/graph/` and Cytoscape.js, vendored under `assets/lib/`:
the version is pinned in `package.json`, and `assets/lib/vendor.yml` records
the file's checksum, which `test/test_dependency_graph.py` verifies. To upgrade,
bump the pin, copy `dist/cytoscape.min.js` from `npm pack cytoscape@<version>`
into a new versioned directory, and update the manifest and
`_includes/graph/viewer.html`.

`vaft._pipeline_graph` (#1647) dry-runs the production Snakefiles with a
documentation configuration (`PIPELINES` in the module) in a temporary
directory, so it needs Snakemake (a core dependency) and nothing else: no
database, HSDS server, solver or credential. Non-scheduling references between
pipelines are declared in `workflow/automatic_pipeline_1_routine_data_processing/paths.py`
(`SCIENTIFIC_REFERENCES`); the pipeline explorer reuses the shared viewer.

`main` gains the last two when a release carries the generators and its
`generators.yml` declares them; nothing about the stable track changes before then.

On a track that ships `scripts/catalog_coverage.py`, `build.py` runs it right
after the generators, against the same tree. It fails the build when a public
formula, process, plot or diagram is in no catalog, or a catalog entry no longer
exists; its docstring says what "public" means for each layer.
`validate_docs.rb` then checks the other half: every catalog entry is rendered
exactly once on its reference page (`data-catalog` elements in the built HTML),
and nothing is rendered that the catalog no longer holds.

The API pages under `/reference/api/` list every object a public module (no `_`
in its dotted name) names in its `__all__`, with the signature, summary,
deprecation status and source read from the code. `api_inventory.yml` says which
page each module belongs to and lists the public modules that declare no `__all__`
yet. A new module without `__all__` fails the build until it declares one or is
added to that list; so does a listed module that has since declared one.
Functions with a scientific detail page (formula, process, plot, diagram) appear
there only as a link. The API pages are left out of the site search index, which
would otherwise double in size.

Every generated entry (formula, process, plot, diagram, API object and class
member) links its source as
`github.com/VEST-Tokamak/vaft/blob/<commit>/<path>#L<first>-L<last>`. `<commit>`
is the catalog's `provenance.commit`, and the range is the whole definition,
decorators included. There is never a branch link, and a catalog without a
provenance commit renders no links. `build.py` archives the commit it pins to.
`npm run data` pins to `HEAD` but reads the working tree, so uncommitted edits
shift the local links.

On the formula, process, plot and diagram pages, functions of at most 80 lines
(`vaft._docstring.INLINE_SOURCE_LINES`) also show that code under a collapsed
"Show source". The API pages link only: inlining their ~3000 functions would
double the site.

`catalog_coverage.py` resolves each row's object and requires the span to be
its definition in the tree: its file, its first line, and the end of its block.
It also requires the inline code to be exactly those lines. `validate_docs.rb`
requires every rendered link and inline view to match its catalog.

The pictures on `/reference/plot/` are committed, not drawn by the build.
`python -m vaft.plot.docs_thumbnails` renders each registered plot from the first
packaged sample that can draw it into `assets/plots/<name>.png`, and records in
`assets/plots/manifest.json` the sample, renderer and view-model hashes it was
drawn from (or why a plot has no picture). An orphaned or hand-edited thumbnail,
or a recorded one whose PNG is missing, fails the build. A newly registered plot
with no manifest entry yet only warns: its page says it has no thumbnail until
the next render. A stale thumbnail (its renderer or sample changed since) also
only warns and is labelled on the page. Re-render after changing a renderer with
the same command (a few minutes; `--only NAME` for one plot), and run
`--check` to also compare the view models, which needs the samples.

## Building

Use a current Ruby rather than the macOS system Ruby. On Apple Silicon, once:

```bash
brew install ruby
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
gem install bundler -v 2.5.16
```

Then:

```bash
cd docs
bundle install
python build.py                      # dry run: both tracks, composed and validated
python build.py --output /tmp/site   # ...and keep the result to look at
```

`build.py` extracts `main` and `develop` with `git archive` into a temporary
directory, runs each branch's declared generators against its own tree, builds
both with Jekyll, composes them, validates the result, and only then publishes.
It never reads or writes your checkout, and the branch you are standing on makes
no difference to the output. If any step fails, `gh-pages` is left exactly as it
was.

Publishing normally happens in CI, on a push to `main` or `develop`:

```bash
python build.py --publish
```

## Working on a page

```bash
cd docs
npm run data          # regenerate _data from this checkout
npm run docs:serve    # http://localhost:4000/vaft/ with live reload
npm run test:docs     # build and validate the stable track
npm run test:docs:develop
```

`_data/vest_diagnostics.yml`, `_data/formula_catalog.yml`,
`_data/process_catalog.yml`, `_data/plot_catalog.yml`,
`_data/diagram_catalog.yml`, `_data/api_catalog.yml`, `_data/dependency_graph.yml`, `_data/pipeline_graph.yml`, `_data/ecosystem.yml` and `_data/provenance.yml` are generated and are
not committed. `generators.yml`
declares which generators this branch has, which is why that file differs
between `main` and `develop`.

Visual regression tests are run by hand, not by the publish workflow:

```bash
npm install && npx playwright install chromium webkit
npm run test:visual          # npm run test:visual:update after an intended change
```

The committed screenshots were captured on arm64 macOS and will not match a
Linux renderer.

## Layout

| Path | Purpose |
| --- | --- |
| `_config.yml` | site config; `_config.develop.yml` overlays the development track |
| `_data/navigation.yml` | sidebar titles, ids, order and canonical URLs |
| `_data/resources.yml` | notebook, API and data-source ids referenced by page front matter |
| `_data/page_migrations.yml` | retired URLs and where they now point |
| `_data/notebook_outputs.yml` | published figure cards and their provenance |
| `_guide/` | the canonical pages, plus hidden redirect stubs |
| `_pages/` | the `pages` collection (About, Contact) |
| `_includes/`, `_layouts/` | vendored theme partials, locally modified |
| `assets/` | images and theme assets |
| `assets/graph/`, `_includes/graph/` | the shared interactive graph viewer and its per-graph adapters |
| `assets/lib/` | pinned third-party browser libraries (`vendor.yml` records version and checksum) |
| `scripts/validate_docs.rb` | the validator that `build.py` and `npm run test:docs` run |
| `scripts/catalog_coverage.py` | the public-surface check `build.py` runs after the generators |
| `build.py`, `generators.yml` | the build and publish pipeline |

Sidebar order and canonical URLs come only from `_data/navigation.yml`; page
dates do not control navigation. Cross-link canonical routes with
`{{ site.baseurl }}`. Every retired URL needs a redirect document and an entry in
`_data/page_migrations.yml`.

New guide page:

```yaml
---
title: Human readable title
author: VEST team
date: 2026-07-01 10:20
category: guide
layout: post
---
```

Add `mermaid: true` only if the page contains a mermaid fence. Content is
kramdown with GFM: fenced code blocks, `$...$` math via MathJax, and images
referenced as `![alt]({{ site.baseurl }}/assets/images/...)`.

The vendored `_layouts/` and `_includes/` are **not** upstream copies. They carry
the per-page table of contents, the navigation built from `navigation.yml`, the
notebook-output cards and the related-resources block, all of which the test
suite asserts on. Change them deliberately; they affect every page.

## Rolling back a publish

Each publish is one ordinary commit on `gh-pages`, so the previous site is the
previous commit:

```bash
git push origin <previous-tip>:refs/heads/gh-pages --force-with-lease
```

That is the only place a force push is legitimate here.

## If the workflow cannot push

The publish job needs a write-scoped `GITHUB_TOKEN`. The repository's default
workflow permission is read, which the workflow overrides explicitly, but an
organisation policy can still forbid it; the job checks for this and says so
rather than failing at the push. The fallback is a deploy key with write access,
stored as a repository secret, with the remote switched to SSH before
`build.py --publish` runs.
