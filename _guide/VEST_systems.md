---
title: VEST systems inventory
author: VEST team
category: guide
layout: post
permalink: /reference/vest-systems/
guide:
  architecture: Maintained inventory linking VEST systems to their IMAS IDS and VAFT coverage.
  prerequisites: None.
  expected: The current mapping status and the appropriate route for technical enquiries.
related:
  notebooks: [database-initialization]
  api: [mapping]
  data_sources: [raw-daq, hsds-public]
---

This inventory is the maintained site view of the VEST experimental-data list. It records the
system owner named in that source, the intended IMAS IDS, and whether VAFT currently maps it.
**Implemented** means that the named VAFT module exists; it does not guarantee that every shot
contains that diagnostic.

Individual technical email addresses are not published in the source inventory. Use the
[Contacts]({{ site.baseurl }}/reference/contacts/) page to route a system-specific request to the
listed manager. For implementation details, see
[Data access and IMAS]({{ site.baseurl }}/workflows/data-access-imas/).

{% assign categories = "Information and control,Heating and current drive,Diagnostics,Device and model" | split: "," %}
{% for category in categories %}
## {{ category }}

<table>
  <thead>
    <tr>
      <th>System</th>
      <th>IMAS IDS</th>
      <th>VAFT module</th>
      <th>Status</th>
      <th>Technical manager</th>
    </tr>
  </thead>
  <tbody>
  {% for item in site.data.vest_systems.systems %}{% if item.category == category %}
    <tr>
      <td>{{ item.system }}</td>
      <td><code>{{ item.ids }}</code></td>
      <td>{% if item.module == "—" %}&mdash;{% else %}<code>{{ item.module }}</code>{% endif %}</td>
      <td>{{ item.status }}</td>
      <td>{{ item.manager }}</td>
    </tr>
  {% endif %}{% endfor %}
  </tbody>
</table>
{% endfor %}

## Keeping this list current

Update `_data/vest_systems.yml` when a system changes owner, gains a VAFT mapping, or is retired.
The data file is derived from
[`vest_experimental_data_list.ipynb`](https://github.com/VEST-Tokamak/vaft/blob/develop/notebooks/vest_experimental_data_list.ipynb);
do not add personal email addresses unless their publication has been explicitly authorized.
