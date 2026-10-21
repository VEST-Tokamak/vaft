"""Vendored Lucide icon sources (#2101): verbatim SVG child elements, not code.

Copied from the npm package ``lucide-static`` at the pinned version below; each entry is the
icon's child elements exactly as published (whitespace collapsed), with its sha256 so an
accidental edit fails ``test_diagram_pictograms``. ``vaft.diagram.pictograms`` converts them to
scene polylines; nothing is fetched at run time. Update by re-extracting from a newer pinned
release and regenerating the affected diagrams (``python -m vaft.diagram.build``).
"""

LUCIDE_VERSION = "1.52.0"
LUCIDE_SOURCE = "https://registry.npmjs.org/lucide-static/-/lucide-static-1.52.0.tgz"
#: sha1 of the tarball, as npm publishes it (``dist.shasum``)
LUCIDE_TARBALL_SHA1 = "ed1a4f3e64f06e30a50d7898fa1f7a4c33545218"

#: the package's LICENSE, reproduced as its ISC and MIT terms require
LUCIDE_LICENSE = """\
ISC License

Copyright (c) 2026 Lucide Icons and Contributors

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

---

The following Lucide icons are derived from the Feather project:

airplay, alert-circle, alert-octagon, alert-triangle, aperture, arrow-down-circle, arrow-down-left, arrow-down-right, arrow-down, arrow-left-circle, arrow-left, arrow-right-circle, arrow-right, arrow-up-circle, arrow-up-left, arrow-up-right, arrow-up, at-sign, calendar, cast, check, chevron-down, chevron-left, chevron-right, chevron-up, chevrons-down, chevrons-left, chevrons-right, chevrons-up, circle, clipboard, clock, code, columns, command, compass, corner-down-left, corner-down-right, corner-left-down, corner-left-up, corner-right-down, corner-right-up, corner-up-left, corner-up-right, crosshair, database, divide-circle, divide-square, dollar-sign, download, external-link, feather, frown, hash, headphones, help-circle, info, italic, key, layout, life-buoy, link-2, link, loader, lock, log-in, log-out, maximize, meh, minimize, minimize-2, minus-circle, minus-square, minus, monitor, moon, more-horizontal, more-vertical, move, music, navigation-2, navigation, octagon, pause-circle, percent, plus-circle, plus-square, plus, power, radio, rss, search, server, share, shopping-bag, sidebar, smartphone, smile, square, table-2, tablet, target, terminal, trash-2, trash, triangle, tv, type, upload, x-circle, x-octagon, x-square, x, zoom-in, zoom-out

The MIT License (MIT) (for the icons listed above)

Copyright (c) 2013-present Cole Bemis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#: lucide name -> (SVG child elements, sha256 of that markup)
ICONS = {
    "archive": (
        '<rect width="20" height="5" x="2" y="3" rx="1" /><path d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8" /><path d="M10 12h4" />',
        "323a0c4f52f184ecd145c7bf66a8ebfa18cbb2020f24852f67594af7346f04da",
    ),
    "axis-3d": (
        '<path d="M13.5 10.5 15 9" /><path d="M4 4v15a1 1 0 0 0 1 1h15" /><path d="M4.293 19.707 6 18" /><path d="m9 15 1.5-1.5" />',
        "42657d2d4a0593922bed077bbe30e576e5bd1149973217adee4188fddfe904fd",
    ),
    "book-open": (
        '<path d="M12 5v16" /><path d="M20.001 19A2 2 0 0022 17V5a2 2 0 00-1.999-2L16 3.002A5 5 0 0012 5a5 5 0 00-4-2H4a2 2 0 00-2 2v12a2 2 0 001.999 2H8a5 5 0 014 2 5 5 0 014-2z" />',
        "f51ab33c9783adb6ca5e57d79503d1444b7c3e70e74ad723c54fce777bc4a95c",
    ),
    "bot": (
        '<path d="M12 8V4H8" /><rect width="16" height="12" x="4" y="8" rx="2" /><path d="M2 14h2" /><path d="M20 14h2" /><path d="M15 13v2" /><path d="M9 13v2" />',
        "26217e7a14de2c6072902e36dd79243916f74df49b00f71f5512a58ef836fcba",
    ),
    "cable": (
        '<path d="M17 19a1 1 0 0 1-1-1v-2a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2a1 1 0 0 1-1 1z" /><path d="M17 21v-2" /><path d="M19 14V6.5a1 1 0 0 0-7 0v11a1 1 0 0 1-7 0V10" /><path d="M21 21v-2" /><path d="M3 5V3" /><path d="M4 10a2 2 0 0 1-2-2V6a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2a2 2 0 0 1-2 2z" /><path d="M7 5V3" />',
        "e54b97c799e51d95a5a3505860cf98e35f7a58004d1474b641498d8837703452",
    ),
    "calculator": (
        '<rect width="16" height="20" x="4" y="2" rx="2" /><line x1="8" x2="16" y1="6" y2="6" /><line x1="16" x2="16" y1="14" y2="18" /><path d="M16 10h.01" /><path d="M12 10h.01" /><path d="M8 10h.01" /><path d="M12 14h.01" /><path d="M8 14h.01" /><path d="M12 18h.01" /><path d="M8 18h.01" />',
        "190a2e04211a5a6c9ea68d1dcaa521ffd43cb69993e8dc0295082eba7f6ed2de",
    ),
    "chart-line": (
        '<path d="M3 3v16a2 2 0 0 0 2 2h16" /><path d="m19 9-5 5-4-4-3 3" />',
        "57fe1d273db81ea5d8e725124a0b24dcbd613c46cc68f639018bc04070ff97af",
    ),
    "code": (
        '<path d="m16 18 6-6-6-6" /><path d="m8 6-6 6 6 6" />',
        "9169a3a3d60c47733ad5ad29dd0bfedf9ed9c39bb00c116ef1703d7260a33f1d",
    ),
    "cpu": (
        '<path d="M12 20v2" /><path d="M12 2v2" /><path d="M17 20v2" /><path d="M17 2v2" /><path d="M2 12h2" /><path d="M2 17h2" /><path d="M2 7h2" /><path d="M20 12h2" /><path d="M20 17h2" /><path d="M20 7h2" /><path d="M7 20v2" /><path d="M7 2v2" /><rect x="4" y="4" width="16" height="16" rx="2" /><rect x="8" y="8" width="8" height="8" rx="1" />',
        "6fe82f01481963fcc379b799e39b7080b0a208a1009d433bb3f0c377d360d04f",
    ),
    "database": (
        '<ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M3 5V19A9 3 0 0 0 21 19V5" /><path d="M3 12A9 3 0 0 0 21 12" />',
        "45b01a8abe49a1ebe2f1e9325b05c946c23eb91ee2a9fa7f50980dc9dcdaaa63",
    ),
    "file-code": (
        '<path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z" /><path d="M14 2v5a1 1 0 0 0 1 1h5" /><path d="M10 12.5 8 15l2 2.5" /><path d="m14 12.5 2 2.5-2 2.5" />',
        "2713d2292ea56c0da3d68ada52ed916853a9ded66ebd5dce64bd711ec2090daa",
    ),
    "file-input": (
        '<path d="M4 11V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.706.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-1" /><path d="M14 2v5a1 1 0 0 0 1 1h5" /><path d="M2 15h10" /><path d="m9 18 3-3-3-3" />',
        "4a8eb135dd283e59df24ac648ae4dff387d3d4680a725da12af4f5a155444807",
    ),
    "file-output": (
        '<path d="M4.226 20.925A2 2 0 0 0 6 22h12a2 2 0 0 0 2-2V8a2.4 2.4 0 0 0-.706-1.706l-3.588-3.588A2.4 2.4 0 0 0 14 2H6a2 2 0 0 0-2 2v3.127" /><path d="M14 2v5a1 1 0 0 0 1 1h5" /><path d="m5 11-3 3" /><path d="m5 17-3-3h10" />',
        "9444b9b304cf52e7ed63258dc83f24aa2bbb7420783d073a1de3e9cac56532c9",
    ),
    "file-text": (
        '<path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z" /><path d="M14 2v5a1 1 0 0 0 1 1h5" /><path d="M10 9H8" /><path d="M16 13H8" /><path d="M16 17H8" />',
        "3c02af223311f1c05d0b940def1f851c2faeda508b161a8e7916e2ecde863f00",
    ),
    "flask-conical": (
        '<path d="M14 2v6a2 2 0 0 0 .245.96l5.51 10.08A2 2 0 0 1 18 22H6a2 2 0 0 1-1.755-2.96l5.51-10.08A2 2 0 0 0 10 8V2" /><path d="M6.453 15h11.094" /><path d="M8.5 2h7" />',
        "7a70889d3657ec9ec9b78fdd83ac35d6f2330fb40cc00b8d8947d06ff2cf089e",
    ),
    "funnel": (
        '<path d="M10 20a1 1 0 0 0 .553.895l2 1A1 1 0 0 0 14 21v-7a2 2 0 0 1 .517-1.341L21.74 4.67A1 1 0 0 0 21 3H3a1 1 0 0 0-.742 1.67l7.225 7.989A2 2 0 0 1 10 14z" />',
        "f7cb434c2dd9355e9ea9bc7be5ac4fbf5a4aef5de2b5362bebd2aefe0329fc52",
    ),
    "history": (
        '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" /><path d="M12 7v5l4 2" />',
        "99714994d49061eccc8c9bf7de70b0feaa0be18951e0a913a28da71dcc4644d7",
    ),
    "laptop": (
        '<path d="M18 5a2 2 0 0 1 2 2v8.526a2 2 0 0 0 .212.897l1.068 2.127a1 1 0 0 1-.9 1.45H3.62a1 1 0 0 1-.9-1.45l1.068-2.127A2 2 0 0 0 4 15.526V7a2 2 0 0 1 2-2z" /><path d="M20.054 15.987H3.946" />',
        "9d85483605e204158c6aa0e0da08dacf4b023684692e60129ea2f8910b897c2d",
    ),
    "library": (
        '<path d="m16 6 4 14" /><path d="M12 6v14" /><path d="M8 8v12" /><path d="M4 4v16" />',
        "75e4a131c295d08327dbefb9662865a214b2c0c9a3fbd3fd048848d5a9654a67",
    ),
    "lightbulb": (
        '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5" /><path d="M9 18h6" /><path d="M10 22h4" />',
        "0ee6265cf83f112b54f1e759993ad9242106684ec7f5b1bf5c20bf9017bc4721",
    ),
    "list-tree": (
        '<path d="M8 5h13" /><path d="M13 12h8" /><path d="M13 19h8" /><path d="M3 10a2 2 0 0 0 2 2h3" /><path d="M3 5v12a2 2 0 0 0 2 2h3" />',
        "b2d78c6fecf68f5e7313b02cfc6ad58a839c5b67c6f259a44564d370858e09ac",
    ),
    "monitor": (
        '<rect width="20" height="14" x="2" y="3" rx="2" /><line x1="8" x2="16" y1="21" y2="21" /><line x1="12" x2="12" y1="17" y2="21" />',
        "a15b2ced27d0e0ebf16d983a4cff1916a53ab58af714b9fbf0d0a68f249548fc",
    ),
    "network": (
        '<rect x="16" y="16" width="6" height="6" rx="1" /><rect x="2" y="16" width="6" height="6" rx="1" /><rect x="9" y="2" width="6" height="6" rx="1" /><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3" /><path d="M12 12V8" />',
        "c636189c7c5ab419b5084fb13a2b617e129586b745029b3bc6bab205c4b4a6fd",
    ),
    "package": (
        '<path d="M11 21.73a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73z" /><path d="M12 22V12" /><polyline points="3.29 7 12 12 20.71 7" /><path d="m7.5 4.27 9 5.15" />',
        "70e914aedb6fcf76db853eb8b6e95da2a169e8287f2ca7a02a7275908e4f6107",
    ),
    "plug": (
        '<path d="M12 22v-5" /><path d="M15 8V2" /><path d="M17 8a1 1 0 0 1 1 1v4a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V9a1 1 0 0 1 1-1z" /><path d="M9 8V2" />',
        "f653496fb29629136f9a1c085475fac4a6c9b0e0eb94004539490cb58a8fd119",
    ),
    "scale": (
        '<path d="M12 3v18" /><path d="m19 8 3 8a5 5 0 0 1-6 0zV7" /><path d="M3 7h1a17 17 0 0 0 8-2 17 17 0 0 0 8 2h1" /><path d="m5 8 3 8a5 5 0 0 1-6 0zV7" /><path d="M7 21h10" />',
        "bbc142bad938b8e935970369e3915ad16448bf6f6a213285477a36740f80b395",
    ),
    "server": (
        '<rect width="20" height="8" x="2" y="2" rx="2" ry="2" /><rect width="20" height="8" x="2" y="14" rx="2" ry="2" /><line x1="6" x2="6.01" y1="6" y2="6" /><line x1="6" x2="6.01" y1="18" y2="18" />',
        "4d7b94eee27023f09fd6c18e8de5e1e40cb7af30b138ecd7078761d0bdeeed81",
    ),
    "shield-check": (
        '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z" /><path d="m9 12 2 2 4-4" />',
        "2ac23312d7b1669c5a1a79107fe320417b629e76782cf2000c05bad1369fda33",
    ),
    "sigma": (
        '<path d="M18 7V5a1 1 0 0 0-1-1H6.5a.5.5 0 0 0-.4.8l4.5 6a2 2 0 0 1 0 2.4l-4.5 6a.5.5 0 0 0 .4.8H17a1 1 0 0 0 1-1v-2" />',
        "2cfdf87791b44a2e7f583bd92a64a0d558a4d2183e0a62b9f9fd70c4c0adcc0c",
    ),
    "sliders-horizontal": (
        '<path d="M10 5H3" /><path d="M12 19H3" /><path d="M14 3v4" /><path d="M16 17v4" /><path d="M21 12h-9" /><path d="M21 19h-5" /><path d="M21 5h-7" /><path d="M8 10v4" /><path d="M8 12H3" />',
        "c87e804b0726c52b7d67f308cd4ec8cd2ec92e334fc4c9043a5dbbd2d74dbc25",
    ),
    "terminal": (
        '<path d="M12 19h8" /><path d="m4 17 6-6-6-6" />',
        "4a4f78e2f0cb34efd15364c8fe99b3edcc2759c30d15546b2a519af2249769e2",
    ),
    "user": (
        '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />',
        "6d45d70f0d26d4c9296dee1b54a5269c15ceaa4d186bada31016565d3a76fda8",
    ),
    "users": (
        '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" /><path d="M16 3.128a4 4 0 0 1 0 7.744" /><path d="M22 21v-2a4 4 0 0 0-3-3.87" /><circle cx="9" cy="7" r="4" />',
        "a03e35f20c9ec215f3a3419893e5f2318868bd2961221e6f1c4dc88ff1673792",
    ),
    "workflow": (
        '<rect width="8" height="8" x="3" y="3" rx="2" /><path d="M7 11v4a2 2 0 0 0 2 2h4" /><rect width="8" height="8" x="13" y="13" rx="2" />',
        "f0797799f334dcae53825feff988c07ed9fed04237dd2cce9d2ab4fb8d2056d3",
    ),
}
