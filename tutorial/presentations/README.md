# VAFT presentations (pilot)

One Quarto source per lecture, rendered to two backends:

```text
.qmd (canonical source)
   ├── HTML → Reveal.js   live teaching, presenter view, speaker notes
   └── PDF  → Beamer      archival and printable
```

This is the **pilot slice** of [issue #322](https://github.com/VEST-Tokamak/vaft/issues/322),
not yet a repository-wide convention. Session 01's slides are authored here;
sessions 02–06 remain hand-written Beamer in the parent directory until the
pilot has been reviewed.

## Which output do I use?

| You are… | Use | Why |
| --- | --- | --- |
| teaching live | `01_….html` | presenter view (press **s**), responsive, links and video work |
| distributing or archiving | `01_….pdf` | deterministic, printable, no browser needed |
| preparing to teach | `01_….-presenter.pdf` | the same slides with the speaker notes as facing pages |

## Building

```bash
make -C tutorial presentations            # all three outputs
make -C tutorial presentations-html       # Reveal.js only
make -C tutorial presentations-pdf        # archival Beamer PDF only
make -C tutorial presentations-presenter  # Beamer PDF with notes shown
```

Requires [Quarto](https://quarto.org) and a LaTeX installation for the PDF
targets. `make -C tutorial slides` still builds the five Beamer decks and is
unaffected.

> **Target order matters.** Quarto's `--output` *renames* the render rather than
> adding a second file, so the presenter build must run before the archival one.
> The `presentations` target already sequences them correctly; if you invoke the
> targets by hand, run `presentations-pdf` last.

## Nothing here is committed

The rendered `.html`, `.pdf`, `_files/` and Quarto's `.quarto/`/`_freeze/` cache
are all git-ignored. This differs from the Beamer decks in the parent directory,
which do commit their PDFs.

The reason is the Reveal.js output: it is a directory of JS, CSS and fonts
regenerated on every render, and committing it would put exactly the kind of
generated weight into git that the tutorial tree's rules exist to prevent. CI
renders both backends on every pull request and uploads them as the
`presentation-pilot` artifact, which is also how a reviewer compares the two
forms side by side.

If the pilot is adopted, whether to commit the PDF is worth revisiting — the
paired-source-plus-verified-rebuild machinery the Beamer decks use would apply
unchanged.

## Speaker notes reach both backends

Notes are authored **once**, in `::: {.notes}` blocks:

```markdown
::: {.notes}
[3 min]

NARRATE:
Explain why a common fusion data model is needed.

WARN:
Do not describe IMAS as only a file format.
:::
```

What happens to them, verified rather than assumed:

- **Reveal.js** renders them into `<aside class="notes">`, which the presenter
  view shows. Press **s**.
- **Beamer** receives them too. Pandoc turns every `::: {.notes}` block into
  `\note{…}`, and Beamer's default is to *hide* notes — which is what the
  archival PDF wants. The presenter target passes
  `\setbeameroption{show notes}` through `--include-in-header`, and the same
  notes come out as facing pages.

So the issue's optional third target works, and no note is ever written twice.

## Layout

```text
presentations/
├── README.md
├── _quarto.yml
├── _extensions/vaft/vaftslides/     the shared theme, both backends
│   ├── _extension.yml
│   ├── vaft-beamer.tex              Beamer preamble
│   ├── vaft.scss                    Reveal.js theme
│   └── show-notes.tex               presenter build only
├── assets/bibliography/vaft.bib
└── 01_getting_started_with_vaft.qmd
```

### Typography

Both backends set the same text the same way, which is what makes them read as
one course rather than two decks that happen to share a colour:

| | Beamer | Reveal.js |
| --- | --- | --- |
| typeface | `helvet` (Nimbus Sans) | `Helvetica Neue, Helvetica, Arial, Nimbus Sans` |
| title | 36 pt bold | 100 px bold |
| frame title | 23 pt bold | 64 px bold |
| body | 15 pt | 40 px |

A 16:9 Beamer canvas is 90 mm tall and a Reveal slide is 700 logical px, so one
px is `90/700/0.35278 = 0.3645 pt`. The Beamer sizes above are Quarto's Reveal
defaults converted through that factor — which is why **`vaft.scss` deliberately
does not override `$presentation-font-size-root`**. Changing it there shrinks one
backend relative to the other while both still look individually reasonable.

Reveal ships Source Sans Pro, and matching it exactly on the Beamer side would
mean `sourcesanspro` from `texlive-fonts-extra` — roughly 1.2 GB on a CI job that
exists to be fast. Both sides meet on a Helvetica-metric face instead:
`helvet` is in `texlive-fonts-recommended`, which CI already installs, and
Nimbus Sans and Liberation Sans are the usual metric-identical Linux
substitutes. If the team would rather have Source Sans on both, the only change
needed is that package plus the apt line.

> **Do not add horizontal padding or margins to `.column` in `vaft.scss`.**
> Quarto lays columns out as `inline-block` boxes at the `width=` you give them,
> with `content-box` sizing, so two 50% columns already fill the line exactly.
> Any extra padding pushes the pair past 100%, the second column wraps onto the
> next line, and a two-column slide silently becomes a stacked one that
> overflows the bottom of the slide.

`_extensions/vaft/vaftslides/` is the architectural point. The six hand-written
decks each carry a duplicated 13-line preamble, and `verify_tutorial.py` forbids
factoring it out with `\input{}`. Here one definition of VAFT's slide identity —
`vaftblue` `RGB(25,76,127)`, type sizes, frame numbering, section slides —
serves every deck and both outputs. Keep `vaft-beamer.tex` and `vaft.scss` in
step: they are the same design expressed twice, and drift between them shows up
as two decks that no longer look like the same course.

Figures live in the shared `../figures/` tree the Beamer decks already use, so a
figure can serve either kind of deck.
