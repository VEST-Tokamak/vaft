---
title: Contacts
author: VEST team
date: 2026-07-01 12:10
category: pages
layout: post
permalink: /reference/contacts/
guide:
  architecture: Human support boundary for data access, contribution, and issue reporting.
  prerequisites: A reproducible example and environment details when reporting a defect.
  expected: The correct public-access path, issue tracker, or maintainer contact route.
related:
  notebooks: [database-initialization]
  api: [database]
  data_sources: [hsds-public]
  outputs: [hsds-39915]
---

VAFT is developed and maintained by the VEST team at Seoul National University. This is the
canonical route for human support: public data access, privileged writes, library maintenance, and
VEST technical-system enquiries.

## Reading the VEST database

**Read access is public.** You do not need to negotiate credentials with anyone — the shot database
ships with a read-only account. Install the HSDS client, then run `hsconfigure` (a console script
provided by `h5pyd`, which writes `~/.hscfg`):

```bash
python -m pip install --no-deps h5pyd==0.20.0
hsconfigure
```

Answer the prompts with the public reader account:

```text
Server endpoint []: http://147.46.36.244:5101
Username []: reader
Password []: test
API Key [None]:
Testing connection...
connection ok
```

Confirm from Python:

```python
import vaft

vaft.database.is_connect()      # -> True
ods = vaft.database.load(39915)
```

The full procedure, including environment setup, is in the
[Quick start guide]({{ site.baseurl }}/guide/Quick_start_guide/) and
[Installation]({{ site.baseurl }}/guide/Installation/) guides.

## Database write access and data policy

Saving to the shared database — uploading shots, writing into a directory of your own — is
restricted to authorized users. To request write credentials, contact:

**[peppertonic18@snu.ac.kr](mailto:peppertonic18@snu.ac.kr)**

Please say who you are, which institution you are with, and what you intend to write. Until write
access is granted you can still run the entire analysis chain locally: load a shot with the reader
account (or use the sample data packaged in `vaft.data`), process it, and save the result to a
local file.

## VAFT maintenance, bugs, and features

For a library-maintenance question, contact
**[satelite2517@snu.ac.kr](mailto:satelite2517@snu.ac.kr)**. File bugs, feature requests, and patches
on the issue tracker — that way the discussion stays with the code:

**[github.com/VEST-Tokamak/vaft/issues](https://github.com/VEST-Tokamak/vaft/issues)**

A useful report states the VAFT commit or version, your Python version, the shot number (if the
problem is data-dependent), the call you made, and the full traceback. Pull requests are welcome on
the same repository.

## Diagnostic and heating-system enquiries

The [VEST systems inventory]({{ site.baseurl }}/reference/vest-systems/) names the manager recorded
for each diagnostic, control, or heating/current-drive system. Individual technical email addresses
are not published there. Email the VAFT maintainer for routing, naming the system and shot number
where applicable.

## Contact routes

| Purpose | Contact |
| --- | --- |
| Library maintainer, general questions | [satelite2517@snu.ac.kr](mailto:satelite2517@snu.ac.kr) |
| Database write access, VEST data policy | [peppertonic18@snu.ac.kr](mailto:peppertonic18@snu.ac.kr) |
| Diagnostic or heating-system technical enquiry | [VEST systems inventory]({{ site.baseurl }}/reference/vest-systems/) → [satelite2517@snu.ac.kr](mailto:satelite2517@snu.ac.kr) |
| Bugs, feature requests, patches | [GitHub issues](https://github.com/VEST-Tokamak/vaft/issues) |

## The lab

VAFT comes out of the **Nuplex** laboratory at Seoul National University, which operates the VEST
spherical torus. For the device, the research programme, and the people behind it:

**[nuplex.snu.ac.kr](http://nuplex.snu.ac.kr)**

Background on the framework itself, including how to cite it, is on the
[About]({{ site.baseurl }}/pages/about/) page.
