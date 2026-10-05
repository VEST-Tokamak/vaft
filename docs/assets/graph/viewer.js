/*
 * Shared interactive graph viewer for the generated reference graphs
 * (#1646 dependency explorer, #1647 pipeline lineage).
 *
 * The viewer owns rendering, search, checkbox filters, the neighbourhood
 * controls, the detail panel, URL hash state and source links.  It knows
 * nothing about what a node *means*: each page passes an adapter that turns
 * its generated snapshot into Cytoscape elements and renders node details.
 *
 * Adapter contract (all functions receive the viewer `v`):
 *   load(data, v)            -> called once with the fetched snapshot
 *   controls(v)              -> [{id, label, type:'radio'|'checks', options:[{value,label,checked}]}]
 *   elements(state, v)       -> {nodes:[{data}], edges:[{data}], layout, message}
 *   details(id, state, v)    -> HTML string for the detail panel
 *   searchItems(v)           -> [{id, label, hint}]
 *   resolveSearch(id, v)     -> node id to focus (an API object resolves to its module)
 *   style(v)                 -> extra Cytoscape style rules
 *
 * Graph state lives in `state`: one key per control plus `focus`, `direction`
 * and `depth`.  Edges always carry `data.kind`; `direction` follows edge
 * orientation (out = what the focus points at, in = what points at it).
 */
(function () {
  'use strict';

  var REPO = 'https://github.com/VEST-Tokamak/vaft';

  function escapeHtml(text) {
    return String(text == null ? '' : text)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function sourceUrl(commit, source) {
    if (!commit || !source || !source.path) return '';
    var url = REPO + '/blob/' + commit + '/' + source.path;
    if (source.line > 0) {
      url += '#L' + source.line + (source.end_line > source.line ? '-L' + source.end_line : '');
    }
    return url;
  }

  function readHash() {
    var out = {};
    (window.location.hash || '').replace(/^#/, '').split('&').forEach(function (pair) {
      if (!pair) return;
      var bits = pair.split('=');
      out[decodeURIComponent(bits[0])] = decodeURIComponent(bits.slice(1).join('=') || '');
    });
    return out;
  }

  function writeHash(state, keys, defaults) {
    var parts = [];
    keys.forEach(function (key) {
      var value = state[key];
      if (key === 'focus' && value) { parts.push('focus=' + encodeURIComponent(value)); return; }
      if (value === undefined || value === null || value === '') return;
      if (Array.isArray(value)) value = value.join(',');
      var fallback = defaults[key];
      if (Array.isArray(fallback)) fallback = fallback.join(',');
      if (value === fallback) return;
      parts.push(encodeURIComponent(key) + '=' + encodeURIComponent(value));
    });
    var hash = parts.length ? '#' + parts.join('&') : '';
    if (window.history && window.history.replaceState) {
      window.history.replaceState(null, '', window.location.pathname + window.location.search + hash);
    }
  }

  function Viewer(root, adapter) {
    this.root = root;
    this.adapter = adapter;
    this.baseurl = root.getAttribute('data-baseurl') || '';
    this.state = { focus: '', direction: 'both', depth: '1' };
    this.data = null;
    this.cy = null;
  }

  Viewer.prototype.link = function (url, label, cls) {
    if (!url) return '';
    var href = /^https?:/.test(url) ? url : this.baseurl + url;
    return '<a class="' + (cls || 'vg-link') + '" href="' + escapeHtml(href) + '">' + escapeHtml(label) + '</a>';
  };

  Viewer.prototype.sourceLink = function (source, label) {
    var commit = this.data && this.data.provenance ? this.data.provenance.commit : '';
    var url = sourceUrl(commit, source);
    return url ? '<a class="vg-link vg-source" href="' + escapeHtml(url) + '">' + escapeHtml(label || 'Source') + '</a>' : '';
  };

  Viewer.prototype.nodeButton = function (id, label) {
    return '<button type="button" class="vg-node-link" data-focus="' + escapeHtml(id) + '">' +
      escapeHtml(label || id) + '</button>';
  };

  Viewer.prototype.escape = escapeHtml;

  Viewer.prototype.start = function () {
    var self = this;
    var url = this.root.getAttribute('data-src');
    this.setStatus('Loading graph…');
    fetch(url, { credentials: 'same-origin' })
      .then(function (response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.json();
      })
      .then(function (data) {
        if (!data || !data.nodes) throw new Error('the generated snapshot is empty');
        self.data = data;
        self.adapter.load(data, self);
        self.buildControls();
        self.restore();
        self.mountGraph();
        self.render();
        // a pasted link that differs only in its hash does not reload the page
        window.addEventListener('hashchange', function () {
          self.restore();
          self.render();
        });
      })
      .catch(function (error) {
        self.setStatus('The graph could not be loaded (' + error.message + ').', true);
      });
  };

  Viewer.prototype.setStatus = function (text, isError) {
    var status = this.root.querySelector('[data-vg-status]');
    if (!status) return;
    status.textContent = text || '';
    status.hidden = !text;
    status.classList.toggle('vg-error', !!isError);
  };

  Viewer.prototype.hashKeys = function () {
    return this.controls.map(function (control) { return control.id; }).concat(['focus', 'direction', 'depth']);
  };

  Viewer.prototype.buildControls = function () {
    var self = this;
    var rail = this.root.querySelector('[data-vg-controls]');
    this.controls = this.adapter.controls(this);
    var html = '';
    this.controls.forEach(function (control) {
      html += '<fieldset class="vg-group" data-control="' + control.id + '"><legend>' + escapeHtml(control.label) + '</legend>';
      if (control.type === 'checks') {
        self.state[control.id] = control.options.filter(function (o) { return o.checked; }).map(function (o) { return o.value; });
      } else {
        var chosen = control.options.filter(function (o) { return o.checked; })[0] || control.options[0];
        self.state[control.id] = chosen.value;
      }
      control.options.forEach(function (option) {
        var type = control.type === 'checks' ? 'checkbox' : 'radio';
        html += '<label class="vg-option">' +
          (option.swatch ? '<span class="vg-swatch" style="background:' + option.swatch + '"></span>' : '') +
          '<input type="' + type + '" name="vg-' + control.id + '" value="' + escapeHtml(option.value) + '"' +
          (option.checked ? ' checked' : '') + '> ' + escapeHtml(option.label) + '</label>';
      });
      if (control.type === 'checks') {
        html += '<div class="vg-bulk"><button type="button" data-bulk="all">All</button><button type="button" data-bulk="none">None</button></div>';
      }
      html += '</fieldset>';
    });
    html += '<fieldset class="vg-group" data-control="neighbourhood"><legend>Neighbourhood of the selection</legend>' +
      '<label class="vg-option"><input type="radio" name="vg-direction" value="both" checked> Dependencies and dependents</label>' +
      '<label class="vg-option"><input type="radio" name="vg-direction" value="out"> Dependencies only (downstream)</label>' +
      '<label class="vg-option"><input type="radio" name="vg-direction" value="in"> Dependents only (upstream)</label>' +
      '<label class="vg-option vg-inline">Depth <select name="vg-depth">' +
      '<option value="1">1 hop</option><option value="2">2 hops</option><option value="all">all</option></select></label>' +
      '<button type="button" class="vg-clear" data-vg-clear>Show whole view</button></fieldset>';
    rail.innerHTML = html;
    this.defaults = JSON.parse(JSON.stringify(this.state));

    rail.addEventListener('change', function (event) {
      var input = event.target;
      if (input.name === 'vg-direction') self.state.direction = input.value;
      else if (input.name === 'vg-depth') self.state.depth = input.value;
      else self.readControl(input.closest('[data-control]').getAttribute('data-control'));
      self.render();
    });
    rail.addEventListener('click', function (event) {
      var bulk = event.target.getAttribute('data-bulk');
      if (bulk) {
        var group = event.target.closest('[data-control]');
        group.querySelectorAll('input').forEach(function (box) { box.checked = bulk === 'all'; });
        self.readControl(group.getAttribute('data-control'));
        self.render();
      }
      if (event.target.hasAttribute('data-vg-clear')) {
        self.state.focus = '';
        self.render();
      }
    });

    var search = this.root.querySelector('[data-vg-search]');
    var list = this.root.querySelector('datalist');
    var items = this.adapter.searchItems(this);
    list.innerHTML = items.map(function (item) {
      return '<option value="' + escapeHtml(item.id) + '">' + escapeHtml(item.hint || '') + '</option>';
    }).join('');
    var known = {};
    items.forEach(function (item) { known[item.id] = true; });
    function submit() {
      var query = search.value.trim();
      if (!query) return;
      var target = known[query] ? query : null;
      if (!target) {
        var lower = query.toLowerCase();
        var match = items.filter(function (item) { return item.id.toLowerCase().indexOf(lower) !== -1; })[0];
        target = match ? match.id : null;
      }
      var message = self.root.querySelector('[data-vg-search-result]');
      if (!target) {
        message.textContent = 'Nothing matches “' + query + '”.';
        return;
      }
      message.textContent = '';
      self.focus(target);
    }
    search.addEventListener('change', submit);
    search.addEventListener('keydown', function (event) {
      if (event.key === 'Enter') { event.preventDefault(); submit(); }
    });

    this.root.querySelector('[data-vg-details]').addEventListener('click', function (event) {
      var target = event.target.closest('[data-focus]');
      if (target) self.focus(target.getAttribute('data-focus'));
    });
  };

  Viewer.prototype.readControl = function (id) {
    var control = this.controls.filter(function (c) { return c.id === id; })[0];
    if (!control) return;
    var inputs = this.root.querySelectorAll('[data-control="' + id + '"] input');
    if (control.type === 'checks') {
      this.state[id] = Array.prototype.filter.call(inputs, function (i) { return i.checked; }).map(function (i) { return i.value; });
    } else {
      var checked = Array.prototype.filter.call(inputs, function (i) { return i.checked; })[0];
      if (checked) this.state[id] = checked.value;
    }
  };

  Viewer.prototype.syncControls = function () {
    var self = this;
    this.controls.forEach(function (control) {
      var value = self.state[control.id];
      self.root.querySelectorAll('[data-control="' + control.id + '"] input').forEach(function (input) {
        input.checked = Array.isArray(value) ? value.indexOf(input.value) !== -1 : input.value === value;
      });
    });
    this.root.querySelectorAll('input[name="vg-direction"]').forEach(function (input) {
      input.checked = input.value === self.state.direction;
    });
    var depth = this.root.querySelector('select[name="vg-depth"]');
    if (depth) depth.value = this.state.depth;
  };

  /* State is the URL hash over the defaults: a key the hash omits is its default. */
  Viewer.prototype.restore = function () {
    var self = this;
    var hash = readHash();
    var defaults = this.defaults;
    this.controls.forEach(function (control) {
      var allowed = control.options.map(function (o) { return o.value; });
      var value = defaults[control.id];
      if (control.id in hash) {
        if (control.type === 'checks') {
          value = hash[control.id].split(',').filter(function (v) { return allowed.indexOf(v) !== -1; });
        } else if (allowed.indexOf(hash[control.id]) !== -1) {
          value = hash[control.id];
        }
      }
      self.state[control.id] = Array.isArray(value) ? value.slice() : value;
    });
    this.state.direction = /^(both|in|out)$/.test(hash.direction || '') ? hash.direction : defaults.direction;
    this.state.depth = /^(1|2|all)$/.test(hash.depth || '') ? hash.depth : defaults.depth;
    this.state.focus = hash.focus || '';
    this.syncControls();
  };

  Viewer.prototype.mountGraph = function () {
    var self = this;
    var base = [
      { selector: 'node', style: {
        'label': 'data(label)', 'font-size': 11, 'text-valign': 'bottom', 'text-margin-y': 3,
        'background-color': 'data(color)', 'width': 'data(size)', 'height': 'data(size)',
        'color': '#24292f', 'text-outline-color': '#ffffff', 'text-outline-width': 2,
        'border-width': 1, 'border-color': '#57606a' } },
      { selector: 'edge', style: {
        'width': 'data(width)', 'line-color': 'data(color)', 'target-arrow-color': 'data(color)',
        'target-arrow-shape': 'triangle', 'arrow-scale': 0.8, 'curve-style': 'bezier', 'opacity': 0.75 } },
      { selector: 'node.vg-focus', style: { 'border-width': 4, 'border-color': '#cf222e' } },
      { selector: 'node:selected', style: { 'border-width': 3, 'border-color': '#0969da' } },
      { selector: '.vg-faded', style: { 'opacity': 0.15 } }
    ];
    this.cy = window.cytoscape({
      container: this.root.querySelector('[data-vg-canvas]'),
      elements: [],
      style: base.concat(this.adapter.style ? this.adapter.style(this) : []),
      wheelSensitivity: 0.3,
      minZoom: 0.05,
      maxZoom: 4
    });
    this.cy.on('tap', 'node', function (event) {
      self.select(event.target.id());
    });
    this.cy.on('mouseover', 'node', function (event) {
      var node = event.target;
      self.cy.elements().addClass('vg-faded');
      node.closedNeighborhood().removeClass('vg-faded');
    });
    this.cy.on('mouseout', 'node', function () {
      self.cy.elements().removeClass('vg-faded');
    });
  };

  /* Restrict a full element set to the neighbourhood of `focus`. */
  Viewer.prototype.neighbourhood = function (nodes, edges, focus) {
    var depth = this.state.depth === 'all' ? Infinity : parseInt(this.state.depth, 10);
    var direction = this.state.direction;
    var out = {}, inn = {};
    edges.forEach(function (edge) {
      var s = edge.data.source, t = edge.data.target;
      (out[s] = out[s] || []).push(t);
      (inn[t] = inn[t] || []).push(s);
    });
    var keep = {};
    keep[focus] = true;
    function walk(adjacency) {
      var frontier = [focus], seen = {}, level = 0;
      seen[focus] = true;
      while (frontier.length && level < depth) {
        var next = [];
        frontier.forEach(function (id) {
          (adjacency[id] || []).forEach(function (other) {
            if (!seen[other]) { seen[other] = true; keep[other] = true; next.push(other); }
          });
        });
        frontier = next;
        level += 1;
      }
    }
    if (direction !== 'in') walk(out);
    if (direction !== 'out') walk(inn);
    // pinned nodes (an adapter's leaves of the focus, e.g. API objects) survive any direction
    nodes.forEach(function (node) { if (node.data.pinned) keep[node.data.id] = true; });
    return {
      nodes: nodes.filter(function (node) { return keep[node.data.id]; }),
      edges: edges.filter(function (edge) { return keep[edge.data.source] && keep[edge.data.target]; })
    };
  };

  Viewer.prototype.render = function () {
    var view = this.adapter.elements(this.state, this);
    var nodes = view.nodes, edges = view.edges;
    var known = {};
    nodes.forEach(function (node) { known[node.data.id] = true; });
    if (this.state.focus && !known[this.state.focus]) this.state.focus = view.remap ? (view.remap[this.state.focus] || '') : '';
    if (this.state.focus && known[this.state.focus]) {
      var reduced = this.neighbourhood(nodes, edges, this.state.focus);
      nodes = reduced.nodes;
      edges = reduced.edges;
    }
    var message = view.message || '';
    var limit = view.limit || 0;
    if (limit && nodes.length > limit && !this.state.focus) {
      this.cy.elements().remove();
      this.setStatus(nodes.length + ' nodes match these filters, more than this view draws at once (' + limit +
        '). Select a node with the search box, or narrow the filters.');
      this.showDetails();
      writeHash(this.state, this.hashKeys(), this.defaults);
      this.updateCount(0, 0);
      return;
    }
    this.setStatus(message);
    this.cy.startBatch();
    this.cy.elements().remove();
    this.cy.add(nodes.map(function (n) { return { group: 'nodes', data: n.data, classes: n.classes || '' }; }));
    this.cy.add(edges.map(function (e) { return { group: 'edges', data: e.data, classes: e.classes || '' }; }));
    this.cy.endBatch();
    if (this.state.focus) this.cy.getElementById(this.state.focus).addClass('vg-focus');
    var layout = view.layout || { name: 'cose', animate: false, randomize: false, nodeRepulsion: 9000, idealEdgeLength: 90 };
    if (this.state.focus && view.focusLayout) layout = view.focusLayout(this.state.focus);
    this.cy.layout(layout).run();
    this.cy.fit(undefined, 30);
    // a handful of nodes would otherwise be blown up to fill the canvas
    if (this.cy.zoom() > 1.1) { this.cy.zoom(1.1); this.cy.center(); }
    this.updateCount(nodes.length, edges.length);
    this.showDetails(this.state.focus);
    writeHash(this.state, this.hashKeys(), this.defaults);
  };

  Viewer.prototype.updateCount = function (nodes, edges) {
    var count = this.root.querySelector('[data-vg-count]');
    if (count) count.textContent = nodes + ' nodes · ' + edges + ' edges shown';
  };

  Viewer.prototype.select = function (id) {
    this.state.focus = id;
    this.render();
  };

  Viewer.prototype.focus = function (id) {
    var target = this.adapter.resolveSearch ? this.adapter.resolveSearch(id, this) : id;
    if (target && typeof target === 'object') {
      if (target.state) Object.assign(this.state, target.state);
      this.pending = target.highlight || '';
      target = target.focus;
    }
    this.syncControls();
    this.state.focus = target || '';
    this.render();
  };

  Viewer.prototype.showDetails = function (id) {
    var panel = this.root.querySelector('[data-vg-details]');
    if (!id) {
      panel.innerHTML = '<p class="vg-hint">Select a node to see its details, or search for one above. ' +
        'Hover a node to highlight its direct neighbours.</p>';
      return;
    }
    panel.innerHTML = this.adapter.details(id, this.state, this);
    if (this.pending) {
      var row = panel.querySelector('[data-api="' + window.CSS.escape(this.pending) + '"]');
      if (row) { row.classList.add('vg-highlight'); row.scrollIntoView({ block: 'nearest' }); }
      this.pending = '';
    }
  };

  function mountAll() {
    document.querySelectorAll('[data-vaft-graph]').forEach(function (root) {
      if (root.getAttribute('data-vg-mounted')) return;
      var adapter = (window.VaftGraphAdapters || {})[root.getAttribute('data-vaft-graph')];
      if (!adapter || !window.cytoscape) return;
      root.setAttribute('data-vg-mounted', '1');
      var viewer = new Viewer(root, adapter);
      root.vaftGraph = viewer;
      viewer.start();
    });
  }

  window.VaftGraph = { mountAll: mountAll, escapeHtml: escapeHtml, sourceUrl: sourceUrl };
  window.VaftGraphAdapters = window.VaftGraphAdapters || {};

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountAll);
  else window.setTimeout(mountAll, 0);
  if (window.gitbook && window.gitbook.events) window.gitbook.events.on('page.change', mountAll);
})();
