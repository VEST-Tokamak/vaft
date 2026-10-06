/*
 * Adapter for the VAFT scientific ontology explorer (#1702): turns
 * _data/ontology_graph.yml (served as ontology-graph.json) into the shared
 * viewer's elements.  Every node and edge was generated from a registry and
 * says which; relation kinds come from a fixed vocabulary.
 */
(function () {
  'use strict';

  var KIND_COLOR = {
    concept: '#4e79a7', concept_family: '#a0cbe8', diagnostic: '#f28e2b', machine: '#ffbe7d',
    code: '#59a14f', data_format: '#8cd17d', plot: '#b07aa1', api: '#9c755f', ids: '#e15759',
    dd_path: '#ff9da7', validation: '#76b7b2', convention: '#edc948'
  };
  var KIND_LABEL = {
    concept: 'Scientific concepts', concept_family: 'Families', diagnostic: 'Diagnostics', machine: 'Machine systems',
    code: 'External codes', data_format: 'Data formats', plot: 'Plots', api: 'API functions', ids: 'IMAS IDS',
    dd_path: 'Data Dictionary paths', validation: 'Validation checks', convention: 'Conventions'
  };
  var VIEW_LABEL = {
    concepts: 'Concepts (concepts, families, diagnostics, codes)',
    representations: 'Representations (IDS, Data Dictionary, mapping)',
    implementations: 'Implementations (plots, adapters)',
    assessment: 'Assessment (validation, conventions)'
  };
  var LIMIT = 220;
  // how a relation reads from its target's side
  var INVERSE = {
    member_of: 'members', measures: 'measured by', derives: 'derived by', represented_by: 'represents',
    part_of: 'contains', mapped_by: 'maps', visualized_by: 'visualizes', reads: 'read by', assessed_by: 'assesses',
    provided_by: 'provides', uses_convention: 'used by', implemented_by: 'implements', produces: 'produced by'
  };

  var g = null;

  function nodeKindsInView(view) {
    return g.data.views[view] || [];
  }

  var adapter = {
    directions: { both: 'Related in both directions', out: 'Outgoing relations', 'in': 'Incoming relations' },

    load: function (data) {
      g = { data: data, nodes: {}, out: {}, inn: {}, aliases: {} };
      data.nodes.forEach(function (n) {
        g.nodes[n.id] = n;
        (n.facets.aliases || []).forEach(function (alias) { g.aliases[alias] = n.id; });
      });
      data.edges.forEach(function (e) {
        (g.out[e.source] = g.out[e.source] || []).push(e);
        (g.inn[e.target] = g.inn[e.target] || []).push(e);
      });
    },

    controls: function () {
      var present = {};
      g.data.nodes.forEach(function (n) { present[n.kind] = true; });
      return [
        { id: 'view', label: 'View', type: 'radio', options: Object.keys(g.data.views).map(function (view, i) {
          return { value: view, label: VIEW_LABEL[view] || view, checked: i === 0 };
        }) },
        { id: 'kinds', label: 'Node kinds', type: 'checks', options: Object.keys(g.data.kinds).filter(function (k) {
          return present[k];
        }).map(function (k) { return { value: k, label: KIND_LABEL[k] || k, checked: true, swatch: KIND_COLOR[k] }; }) },
        { id: 'relations', label: 'Relation', type: 'checks', options: Object.keys(g.data.relations).map(function (r) {
          return { value: r, label: r.replace(/_/g, ' '), checked: true };
        }) }
      ];
    },

    style: function () {
      return [
        { selector: 'node', style: { 'font-size': 10 } },
        { selector: 'node.on-concept', style: { 'font-size': 12, 'font-weight': 'bold' } },
        { selector: 'node.on-concept_family', style: { 'shape': 'round-hexagon' } },
        { selector: 'node.on-diagnostic, node.on-machine', style: { 'shape': 'round-rectangle' } },
        { selector: 'node.on-code, node.on-data_format', style: { 'shape': 'round-diamond' } },
        { selector: 'node.on-ids, node.on-dd_path', style: { 'shape': 'rectangle' } },
        { selector: 'node.on-validation', style: { 'shape': 'round-triangle' } },
        { selector: 'edge.on-member_of', style: { 'line-style': 'dashed', 'target-arrow-shape': 'none' } },
        { selector: 'edge.on-part_of, edge.on-reads', style: { 'line-style': 'dotted', 'opacity': 0.5 } }
      ];
    },

    elements: function (state) {
      var kinds = {}, relations = {};
      state.kinds.forEach(function (k) { kinds[k] = true; });
      state.relations.forEach(function (r) { relations[r] = true; });
      var inView = {};
      nodeKindsInView(state.view).forEach(function (k) { inView[k] = true; });
      var nodes = [], shown = {};
      g.data.nodes.forEach(function (n) {
        if (!inView[n.kind] || !kinds[n.kind]) return;
        shown[n.id] = true;
        nodes.push({ classes: 'on-' + n.kind, data: { id: n.id, label: n.label, color: KIND_COLOR[n.kind] || '#8c959f',
          size: n.kind === 'concept' ? 22 : (n.kind === 'dd_path' ? 10 : 15), kind: n.kind } });
      });
      var edges = [];
      g.data.edges.forEach(function (e, index) {
        if (!relations[e.kind] || !shown[e.source] || !shown[e.target]) return;
        edges.push({ classes: 'on-' + e.kind, data: { id: 'oe' + index, source: e.source, target: e.target, kind: e.kind,
          width: 1.4, color: '#6e7781' } });
      });
      // a node with no relation in this view says nothing here; only concepts are always listed
      var touched = {};
      edges.forEach(function (e) { touched[e.data.source] = touched[e.data.target] = true; });
      nodes = nodes.filter(function (n) { return touched[n.data.id] || n.data.kind === 'concept'; });
      return {
        nodes: nodes, edges: edges, limit: LIMIT,
        layout: { name: 'cose', animate: false, randomize: false, nodeRepulsion: 12000, idealEdgeLength: 80 },
        focusLayout: function (id) {
          return { name: 'concentric', animate: false, minNodeSpacing: 40,
            concentric: function (node) { return node.id() === id ? 2 : 1; }, levelWidth: function () { return 1; } };
        }
      };
    },

    searchItems: function () {
      var items = g.data.nodes.map(function (n) {
        return { id: n.id, hint: (KIND_LABEL[n.kind] || n.kind) + (n.facets.aliases && n.facets.aliases.length ?
          ' · aka ' + n.facets.aliases.join(', ') : '') };
      });
      Object.keys(g.aliases).sort().forEach(function (alias) {
        items.push({ id: alias, hint: 'alias of ' + g.aliases[alias] });
      });
      return items;
    },

    resolveSearch: function (id, v) {
      // strict identity: an alias resolves to its canonical node, nothing else is folded
      var target = g.nodes[id] ? id : g.aliases[id];
      if (!target) return id;
      var n = g.nodes[target];
      var state = {};
      // a view draws a non-concept node only through a relation inside that view
      var drawnIn = function (view) {
        var kinds = g.data.views[view] || [];
        if (kinds.indexOf(n.kind) === -1) return false;
        if (n.kind === 'concept') return true;
        return (g.out[target] || []).concat(g.inn[target] || []).some(function (e) {
          var other = g.nodes[e.source === target ? e.target : e.source];
          return other && kinds.indexOf(other.kind) !== -1;
        });
      };
      if (!drawnIn(v.state.view)) {
        state.view = Object.keys(g.data.views).filter(drawnIn)[0] || v.state.view;
      }
      if (v.state.kinds.indexOf(n.kind) === -1) state.kinds = v.state.kinds.concat([n.kind]);
      return { focus: target, state: state };
    },

    details: function (id, state, v) {
      var n = g.nodes[id];
      if (!n) return '<p>' + v.escape(id) + '</p>';
      var f = n.facets;
      var html = '<h3 class="vg-title"><code>' + v.escape(n.label) + '</code></h3><p class="vg-meta">' +
        '<span class="vg-swatch" style="background:' + (KIND_COLOR[n.kind] || '#8c959f') + '"></span>' +
        v.escape(KIND_LABEL[n.kind] || n.kind) + (f.concept_kind ? ' · ' + v.escape(f.concept_kind) : '') +
        ' · <code>' + v.escape(n.id) + '</code></p>';
      if (f.aliases && f.aliases.length) html += '<p>Aliases: ' + f.aliases.map(function (a) { return '<code>' + v.escape(a) + '</code>'; }).join(', ') + '</p>';
      if (f.description || f.documentation) html += '<p>' + v.escape(f.description || f.documentation) + '</p>';
      var facts = [];
      ['units', 'coordinates', 'data_type', 'lifecycle', 'dd_version', 'view', 'domain', 'category', 'family',
        'availability', 'mapping_status', 'ids_path', 'unit', 'measure', 'tolerance', 'cocos', 'psi_unit', 'roles', 'mode',
        'maturity', 'role'].forEach(function (key) {
        if (f[key] === undefined) return;
        var value = Array.isArray(f[key]) ? f[key].join(', ') : String(f[key]);
        facts.push('<li><span class="vg-meta">' + v.escape(key.replace(/_/g, ' ')) + ':</span> ' + v.escape(value) + '</li>');
      });
      if (facts.length) html += '<ul class="vg-list">' + facts.join('') + '</ul>';
      var links = [v.link(f.plot_url, 'Plot reference'), v.link(f.api_url, 'API documentation'),
        v.link(f.code_url, 'External-code reference'), v.link(f.diagnostics_url, 'VEST diagnostics'),
        f.source && f.source.path ? v.sourceLink(f.source, 'Source') : ''];
      if (n.kind === 'api') {
        var module = f.role === 'code adapter' ? n.label : n.label.replace(/\.[^.]+$/, '');
        links.push(v.link('/reference/dependency-graph/#level=modules&focus=' + module, 'Dependency explorer'));
      }
      if (n.kind === 'code' && f.code_url) links.push(v.link('/reference/pipeline-graph/', 'Pipeline lineage'));
      html += '<p class="vg-actions">' + links.filter(Boolean).join(' · ') + '</p>';
      html += '<p class="vg-meta">From ' + n.origins.map(function (o) { return '<code>' + v.escape(o) + '</code>'; }).join(', ') + '</p>';
      var groups = {};
      (g.out[id] || []).forEach(function (e) { (groups[e.kind] = groups[e.kind] || []).push({ other: e.target, dir: 'out', e: e }); });
      (g.inn[id] || []).forEach(function (e) { (groups[e.kind] = groups[e.kind] || []).push({ other: e.source, dir: 'in', e: e }); });
      Object.keys(groups).sort().forEach(function (kind) {
        var rows = groups[kind];
        var outRows = rows.filter(function (r) { return r.dir === 'out'; });
        var inRows = rows.filter(function (r) { return r.dir === 'in'; });
        [[outRows, kind.replace(/_/g, ' ')], [inRows, INVERSE[kind] || kind]].forEach(function (pair) {
          if (!pair[0].length) return;
          html += '<h4>' + v.escape(pair[1]) + ' (' + pair[0].length + ')</h4><ul class="vg-list">' + pair[0].slice(0, 60).map(function (r) {
            var other = g.nodes[r.other];
            return '<li>' + v.nodeButton(r.other, other ? other.label : r.other) + '</li>';
          }).join('') + (pair[0].length > 60 ? '<li class="vg-meta">… ' + (pair[0].length - 60) + ' more</li>' : '') + '</ul>';
        });
      });
      return html;
    }
  };

  window.VaftGraphAdapters = window.VaftGraphAdapters || {};
  window.VaftGraphAdapters.ontology = adapter;
})();
