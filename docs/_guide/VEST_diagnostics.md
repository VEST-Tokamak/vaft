---
title: VEST diagnostics and data availability
author: VEST team
category: guide
layout: post
permalink: /reference/vest-diagnostics/
guide:
  architecture: Generated reference view of the VEST diagnostic registry in vaft.machine_mapping.vest.yaml.
  prerequisites: None.
  expected: A system's data source, operational classification, lifecycle, and VAFT support state.
related:
  notebooks: [database-initialization]
  api: [mapping]
  data_sources: [raw-daq, hsds-public]
---

This page is generated from VAFT's canonical diagnostic registry in
[`vest.yaml`](https://github.com/VEST-Tokamak/vaft/blob/develop/vaft/machine_mapping/vest.yaml).
It separates operational data availability from hardware lifecycle and VAFT mapping support.
`If-requested` is a conservative status pending device-operation confirmation; it does not mean
that the system cannot have historical data.

For public data access and general administration, see [Database and data sources]({{ site.baseurl }}/reference/database-data-sources/)
and [Contacts]({{ site.baseurl }}/reference/contacts/). Manager names are retained from the legacy
inventory; no personal email is shown unless it is explicitly recorded in the registry.

## Overview

<table>
  <thead><tr><th>System</th><th>IDS</th><th>Responsible</th><th>Source</th><th>Availability</th><th>Lifecycle</th><th>VAFT mapping</th></tr></thead>
  <tbody>
  {% for item in site.data.vest_diagnostics.diagnostics %}
    {% assign managers = item.responsible | map: "name" | join: ", " %}
    <tr>
      <td>{{ item.name }}</td><td><code>{{ item.ids_path }}</code></td><td>{{ managers | default: "Not listed" }}</td>
      <td>{% if item.source.type == "raw_daq" %}Raw DAQ (MySQL / archived raw dump){% elsif item.source.type == "file" %}File ({{ item.source.formats | join: ", " }}){% elsif item.source.type == "reference_ods" %}Reference ODS{% else %}Unknown{% endif %}</td>
      <td>{{ item.availability }}</td><td>{{ item.lifecycle }}</td><td>{{ item.mapping_status }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

## By physical family

{% assign families = "magnetic_diagnostics,kinetic_diagnostics,spectroscopy_radiation,imaging,edge_diagnostics,heating_current_drive,machine_systems,machine_models,information" | split: "," %}
{% for family in families %}
  {% assign members = site.data.vest_diagnostics.diagnostics | where: "family", family %}
  {% if members.size > 0 %}
### {{ family | replace: "_", " " | capitalize }}

{% for item in members %}- **{{ item.name }}** — `{{ item.ids_path }}`; {{ item.mapping_status }} via {% if item.source.type == "raw_daq" %}raw DAQ{% elsif item.source.type == "file" %}file-backed {{ item.source.formats | join: ", " }}{% else %}{{ item.source.type }}{% endif %}.
{% endfor %}
  {% endif %}
{% endfor %}

## Refreshing this snapshot

From a checkout of the `develop` branch, run:

```bash
python -m vaft.machine_mapping.registry --output /path/to/vaft-gh/_data/vest_diagnostics.yml
```

The snapshot records the SHA-256 of its source `vest.yaml`; documentation validation compares it
when `VAFT_REGISTRY_SOURCE` points to the corresponding source checkout.
