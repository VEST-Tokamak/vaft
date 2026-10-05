/*
 * Adapter for the VAFT dependency explorer (#1646): turns
 * _data/dependency_graph.yml (served as dependency-graph.json) into the
 * shared viewer's elements.  An edge is "module A has an import statement
 * naming module B" -- a source dependency, never a scientific one.
 */
(function () {
  'use strict';

  var PALETTE = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1',
    '#ff9da7', '#9c755f', '#86bcb6', '#d37295', '#a0cbe8', '#8cd17d', '#f1ce63', '#bab0ac'];
  var EXTERNAL_COLOR = '#d0d7de';
  var OTHER_COLOR = '#8c959f';
  var MODULE_LIMIT = 260;

  var graph = null;

  function packageOf(id) {
    if (id === 'vaft') return 'vaft';
    var parts = id.split('.');
    return parts.length < 2 ? id : parts[0] + '.' + parts[1];
  }

  function summaryHtml(v, text) {
    return v.escape(text || '').replace(/`([^`]+)`/g, '<code>$1</code>');
  }

  function plural(n, word) { return n + ' ' + word + (n === 1 ? '' : 's'); }

  var adapter = {
    load: function (data) {
      graph = { data: data, nodes: {}, out: {}, inn: {}, api: {}, apiByModule: {}, color: {}, cycles: data.cycles || [] };
      data.layers.forEach(function (layer, index) {
        graph.color[layer] = layer === 'other' ? OTHER_COLOR : PALETTE[index % PALETTE.length];
      });
      graph.color.external = EXTERNAL_COLOR;
      data.nodes.forEach(function (node) { graph.nodes[node.id] = node; });
      data.edges.forEach(function (edge) {
        (graph.out[edge.source] = graph.out[edge.source] || []).push(edge);
        (graph.inn[edge.target] = graph.inn[edge.target] || []).push(edge);
      });
      (data.api || []).forEach(function (entry) {
        graph.api[entry.id] = entry;
        (graph.apiByModule[entry.module] = graph.apiByModule[entry.module] || []).push(entry);
      });
    },

    controls: function () {
      return [
        { id: 'level', label: 'Level', type: 'radio', options: [
          { value: 'packages', label: 'Packages (vaft.<layer>)', checked: true },
          { value: 'modules', label: 'Modules' }] },
        { id: 'layers', label: 'Layer', type: 'checks', options: graph.data.layers.map(function (layer) {
          return { value: layer, label: layer, checked: true, swatch: graph.color[layer] };
        }) },
        { id: 'scope', label: 'Dependencies', type: 'radio', options: [
          { value: 'internal', label: 'Internal only', checked: true },
          { value: 'external', label: 'Internal + external packages' }] },
        { id: 'api', label: 'Public API', type: 'radio', options: [
          { value: 'hide', label: 'Hide API objects', checked: true },
          { value: 'show', label: 'Show the selected module’s API objects' }] }
      ];
    },

    style: function () {
      return [
        { selector: 'node.vg-external', style: { 'shape': 'round-rectangle', 'border-style': 'dashed' } },
        { selector: 'node.vg-package', style: { 'font-size': 15, 'font-weight': 'bold' } },
        { selector: 'node.vg-api', style: { 'shape': 'diamond', 'font-size': 9 } },
        { selector: 'node.vg-cycle', style: { 'border-color': '#bf8700', 'border-width': 2 } },
        { selector: 'edge.vg-external', style: { 'line-style': 'dashed' } },
        { selector: 'edge.vg-exports', style: { 'line-style': 'dotted', 'target-arrow-shape': 'none' } }
      ];
    },

    elements: function (state) {
      var layers = {};
      state.layers.forEach(function (layer) { layers[layer] = true; });
      var external = state.scope === 'external';
      var nodes = [], edges = [];
      var remap = {};

      if (state.level === 'packages') {
        var groups = {}, links = {};
        graph.data.nodes.forEach(function (node) {
          if (node.kind === 'external') return;
          var key = packageOf(node.id);
          remap[node.id] = key;
          if (!layers[node.layer]) return;
          var group = groups[key] = groups[key] || { id: key, layer: node.layer, count: 0 };
          group.count += 1;
        });
        graph.data.edges.forEach(function (edge) {
          var source = packageOf(edge.source);
          if (!groups[source]) return;
          var target = edge.kind === 'external' ? edge.target : packageOf(edge.target);
          if (edge.kind === 'external' ? !external : !groups[target]) return;
          if (source === target) return;
          var key = source + '→' + target;
          links[key] = links[key] || { source: source, target: target, kind: edge.kind, count: 0 };
          links[key].count += 1;
        });
        Object.keys(groups).sort().forEach(function (key) {
          var group = groups[key];
          nodes.push({ classes: 'vg-package', data: { id: key, label: key.replace(/^vaft\./, ''),
            color: graph.color[group.layer], size: 18 + 4 * Math.sqrt(group.count) } });
        });
        var externals = {};
        Object.keys(links).sort().forEach(function (key) {
          var link = links[key];
          if (link.kind === 'external') externals[link.target] = true;
          edges.push({ classes: link.kind === 'external' ? 'vg-external' : '',
            data: { id: key, source: link.source, target: link.target, kind: link.kind, count: link.count,
              width: Math.min(1 + Math.log(link.count + 1), 6),
              color: link.kind === 'external' ? '#afb8c1' : '#57606a' } });
        });
        Object.keys(externals).sort().forEach(function (id) {
          nodes.push({ classes: 'vg-external', data: { id: id, label: id, color: EXTERNAL_COLOR, size: 16 } });
        });
        return { nodes: nodes, edges: edges, remap: remap, layout: {
          name: 'concentric', animate: false, minNodeSpacing: 46, startAngle: 3 * Math.PI / 2,
          // the most depended-on packages sit in the centre
          concentric: function (node) { return node.indegree(false); },
          levelWidth: function () { return 3; } } };
      }

      var shown = {};
      graph.data.nodes.forEach(function (node) {
        if (node.kind === 'external' || !layers[node.layer]) return;
        shown[node.id] = true;
        nodes.push({ classes: (node.kind === 'package' ? 'vg-package' : '') + (node.cycle ? ' vg-cycle' : ''),
          data: { id: node.id, label: node.id.replace(/^vaft\./, ''), color: graph.color[node.layer],
            size: 12 + 2 * Math.sqrt(node.imported_by) } });
      });
      var externals = {};
      graph.data.edges.forEach(function (edge, index) {
        if (!shown[edge.source]) return;
        if (edge.kind === 'external') {
          if (!external) return;
          externals[edge.target] = true;
        } else if (!shown[edge.target] || edge.source === edge.target) {
          return;
        }
        edges.push({ classes: edge.kind === 'external' ? 'vg-external' : '',
          data: { id: 'e' + index, source: edge.source, target: edge.target, kind: edge.kind, width: 1,
            color: edge.kind === 'external' ? '#afb8c1' : '#6e7781' } });
      });
      Object.keys(externals).sort().forEach(function (id) {
        nodes.push({ classes: 'vg-external', data: { id: id, label: id, color: EXTERNAL_COLOR, size: 14 } });
      });
      var view = { nodes: nodes, edges: edges, limit: MODULE_LIMIT, remap: remap };
      var focus = state.focus;
      if (state.api === 'show' && focus && graph.apiByModule[focus]) {
        // API objects hang off their module as leaves; no object-level dependency is invented.
        graph.apiByModule[focus].forEach(function (entry) {
          nodes.push({ classes: 'vg-api', data: { id: entry.id, label: entry.name, color: '#ffffff', size: 10, pinned: true } });
          edges.push({ classes: 'vg-exports', data: { id: 'x:' + entry.id, source: focus, target: entry.id,
            kind: 'exports', width: 1, color: '#8c959f' } });
        });
      }
      view.focusLayout = function (id) {
        return { name: 'concentric', animate: false, minNodeSpacing: 60,
          concentric: function (node) { return node.id() === id ? 3 : (node.data('pinned') ? 2 : 1); },
          levelWidth: function () { return 1; } };
      };
      return view;
    },

    searchItems: function () {
      var items = [];
      graph.data.nodes.forEach(function (node) {
        items.push({ id: node.id, hint: node.kind + (node.kind === 'external' ? '' : ' · ' + node.layer) });
      });
      (graph.data.api || []).forEach(function (entry) {
        items.push({ id: entry.id, hint: entry.kind + ' in ' + entry.module });
      });
      return items;
    },

    resolveSearch: function (id, v) {
      var state = {};
      var api = graph.api[id];
      var moduleId = api ? api.module : id;
      var node = graph.nodes[moduleId];
      if (!node && !api) {
        // a package node of the packages view
        return id;
      }
      if (node && node.kind !== 'external') {
        var asPackage = packageOf(moduleId) === moduleId;
        if (api || !asPackage) state.level = 'modules';
        if (v.state.layers.indexOf(node.layer) === -1) state.layers = v.state.layers.concat([node.layer]);
      } else if (node && node.kind === 'external') {
        state.scope = 'external';
      }
      if (api) state.api = 'show';
      return { focus: moduleId, state: state, highlight: api ? id : '' };
    },

    details: function (id, state, v) {
      var api = graph.api[id];
      if (api) return apiDetails(api, v, true);
      var node = graph.nodes[id];
      if (state.level === 'packages' && (!node || packageOf(id) === id) && !(node && node.kind === 'external')) {
        return packageDetails(id, v);
      }
      if (!node) return '<p>' + v.escape(id) + '</p>';
      if (node.kind === 'external') return externalDetails(node, v);
      return moduleDetails(node, v);
    }
  };

  function apiDetails(entry, v, standalone) {
    var links = [v.link(entry.api_url, 'API documentation'), v.link(entry.reference_url, 'Scientific reference'),
      v.sourceLink(entry.source, 'Source')].filter(Boolean).join(' · ');
    var html = standalone
      ? '<h3 class="vg-title"><code>' + v.escape(entry.id) + '</code></h3><p class="vg-meta">' + v.escape(entry.kind) +
        ' in ' + v.nodeButton(entry.module) + '</p>'
      : '<li data-api="' + v.escape(entry.id) + '"><code>' + v.escape(entry.name) + '</code> <span class="vg-meta">' +
        v.escape(entry.kind) + (entry.deprecated ? ', deprecated' : '') + '</span>';
    if (standalone) {
      if (entry.summary) html += '<p>' + summaryHtml(v, entry.summary) + '</p>';
      html += '<p class="vg-actions">' + links + '</p>';
      return html;
    }
    return html + (links ? '<br><span class="vg-actions">' + links + '</span>' : '') + '</li>';
  }

  function edgeList(v, edges, side, node) {
    if (!edges.length) return '<p class="vg-meta">None.</p>';
    return '<ul class="vg-list">' + edges.map(function (edge) {
      var other = edge[side];
      var lines = (edge.lines || []).map(function (line) {
        return v.sourceLink({ path: graph.nodes[edge.source].source.path, line: line, end_line: line }, 'L' + line);
      }).join(' ');
      return '<li>' + v.nodeButton(other) + (lines ? ' <span class="vg-meta">import at ' + lines + '</span>' : '') + '</li>';
    }).join('') + '</ul>';
  }

  function moduleDetails(node, v) {
    var out = (graph.out[node.id] || []).filter(function (e) { return e.target !== node.id; });
    var internalOut = out.filter(function (e) { return e.kind === 'internal'; });
    var externalOut = out.filter(function (e) { return e.kind === 'external'; });
    var inn = (graph.inn[node.id] || []).filter(function (e) { return e.source !== node.id; });
    var html = '<h3 class="vg-title"><code>' + v.escape(node.id) + '</code></h3>' +
      '<p class="vg-meta"><span class="vg-swatch" style="background:' + graph.color[node.layer] + '"></span>' +
      v.escape(node.kind) + ' · layer <strong>' + v.escape(node.layer) + '</strong>' +
      (node.public ? '' : ' · private') + '</p>';
    if (node.summary) html += '<p>' + summaryHtml(v, node.summary) + '</p>';
    html += '<p class="vg-actions">' + [v.link(node.api_url, 'API documentation'), v.sourceLink(node.source, 'Source')]
      .filter(Boolean).join(' · ') + '</p>';
    html += '<p class="vg-meta">Imports ' + plural(internalOut.length, 'VAFT module') + ' and ' +
      plural(externalOut.length, 'external package') + '; imported by ' + plural(inn.length, 'VAFT module') + '.</p>';
    if (node.cycle) {
      var members = graph.cycles[node.cycle - 1] || [];
      html += '<p class="vg-note">Part of import cycle ' + node.cycle + ' (' + members.length +
        ' modules that reach each other through imports). A cycle is a fact to inspect, not a verdict.</p>';
    }
    html += '<h4>Dependencies (imports)</h4>' + edgeList(v, internalOut, 'target', node);
    html += '<h4>Dependents (imported by)</h4>' + edgeList(v, inn, 'source', node);
    if (externalOut.length) html += '<h4>External packages</h4>' + edgeList(v, externalOut, 'target', node);
    var exported = graph.apiByModule[node.id] || [];
    if (exported.length) {
      html += '<h4>Public API (' + exported.length + ')</h4><ul class="vg-list vg-api-list">' +
        exported.map(function (entry) { return apiDetails(entry, v, false); }).join('') + '</ul>';
    }
    return html;
  }

  function packageDetails(id, v) {
    var members = graph.data.nodes.filter(function (node) { return node.kind !== 'external' && packageOf(node.id) === id; });
    var layer = members.length ? members[0].layer : 'other';
    var deps = {}, dependents = {};
    graph.data.edges.forEach(function (edge) {
      var source = packageOf(edge.source);
      var target = edge.kind === 'external' ? edge.target : packageOf(edge.target);
      if (source === target) return;
      if (source === id) deps[target] = (deps[target] || 0) + 1;
      if (target === id && edge.kind === 'internal') dependents[source] = (dependents[source] || 0) + 1;
    });
    function counted(map) {
      var keys = Object.keys(map).sort();
      if (!keys.length) return '<p class="vg-meta">None.</p>';
      return '<ul class="vg-list">' + keys.map(function (key) {
        return '<li>' + v.nodeButton(key) + ' <span class="vg-meta">' + plural(map[key], 'import') + '</span></li>';
      }).join('') + '</ul>';
    }
    var root = graph.nodes[id];
    var html = '<h3 class="vg-title"><code>' + v.escape(id) + '</code></h3><p class="vg-meta">' +
      '<span class="vg-swatch" style="background:' + graph.color[layer] + '"></span>' +
      plural(members.length, 'module') + ' · layer <strong>' + v.escape(layer) + '</strong></p>';
    if (root && root.summary) html += '<p>' + summaryHtml(v, root.summary) + '</p>';
    if (root) html += '<p class="vg-actions">' + [v.link(root.api_url, 'API documentation'), v.sourceLink(root.source, 'Source')]
      .filter(Boolean).join(' · ') + '</p>';
    html += '<p class="vg-meta">Counts are import statements between modules of the two packages.</p>';
    html += '<h4>Depends on</h4>' + counted(deps) + '<h4>Depended on by</h4>' + counted(dependents);
    html += '<h4>Modules</h4><ul class="vg-list">' + members.map(function (node) {
      return '<li>' + v.nodeButton(node.id) + '</li>';
    }).join('') + '</ul>';
    return html;
  }

  function externalDetails(node, v) {
    var inn = graph.inn[node.id] || [];
    return '<h3 class="vg-title"><code>' + v.escape(node.id) + '</code></h3>' +
      '<p class="vg-meta">external Python package (top-level import name)</p>' +
      '<p class="vg-note">Only Python imports are shown here. External scientific codes such as EFIT or CHEASE are ' +
      'executables VAFT invokes, not imports, and do not appear in this graph.</p>' +
      '<h4>Imported by (' + inn.length + ')</h4>' + edgeList(v, inn, 'source', node);
  }

  window.VaftGraphAdapters = window.VaftGraphAdapters || {};
  window.VaftGraphAdapters.dependency = adapter;
})();
