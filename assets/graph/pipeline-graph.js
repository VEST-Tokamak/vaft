/*
 * Adapter for the VAFT pipeline lineage explorer (#1647): turns
 * _data/pipeline_graph.yml (served as pipeline-graph.json) into the shared
 * viewer's elements.  Four views over one snapshot, never one undifferentiated
 * edge namespace: Snakemake's rule graph, its resolved job DAG for one
 * documented shot, file-level artifact lineage, and STAGE_REPLICATION's
 * stage-wise HSDS publication.
 */
(function () {
  'use strict';

  var VIEWS = [
    { value: 'rules', label: 'Rules (Snakemake rule graph)' },
    { value: 'dag', label: 'Resolved job DAG (one documented shot)' },
    { value: 'artifacts', label: 'Artifacts (files each rule produces and consumes)' },
    { value: 'publication', label: 'HSDS publication (stage → IDS → source)' }
  ];
  var EDGE_KINDS = [
    { value: 'execution', label: 'Execution (Snakemake schedules it)', color: '#57606a' },
    { value: 'scientific_reference', label: 'Scientific reference (consulted, not scheduled)', color: '#cf222e' },
    { value: 'produces', label: 'Produces', color: '#0969da' },
    { value: 'consumes', label: 'Consumes', color: '#8c959f' },
    { value: 'validates', label: 'Validation product', color: '#1a7f37' },
    { value: 'owns', label: 'Stage owns IDS', color: '#9a6700' },
    { value: 'publishes', label: 'Publishes to HSDS source', color: '#8250df' },
    { value: 'replicated_by', label: 'Replicated by rule', color: '#bf3989' },
    { value: 'records', label: 'Replication record (evidence)', color: '#bf3989' },
    { value: 'parent', label: 'Source parent', color: '#6e7781' }
  ];
  var PIPELINE_COLOR = { routine: '#4e79a7', corrective: '#f28e2b' };
  var ROLE_COLOR = { product: '#a0cbe8', record: '#d0d7de', validation: '#8cd17d', replication_record: '#d4a6c8', file: '#eaeef2' };
  var ROLE_TEXT = {
    product: 'scientific stage product',
    record: 'manifest, status or selection record of a stage',
    validation: 'validation / QA product: evidence about the stage product, not an input to it',
    replication_record: 'replication record: publication evidence, not a scientific product',
    file: 'file PipelinePaths does not name as a single product'
  };

  var g = null;

  // Layered left-to-right DAG layout when the vendored extension loaded, else breadth-first.
  function layout(spacing) {
    if (window.cytoscapeDagre) {
      return { name: 'dagre', rankDir: 'LR', nodeSep: 8 * spacing, rankSep: 60 * spacing, edgeSep: 6, animate: false };
    }
    return { name: 'breadthfirst', directed: true, spacingFactor: spacing, animate: false, avoidOverlap: true };
  }

  function edgeColor(kind) {
    var match = EDGE_KINDS.filter(function (k) { return k.value === kind; })[0];
    return match ? match.color : '#8c959f';
  }

  function pipelineTitle(id) {
    var p = g.pipelines[id];
    return p ? p.title : id;
  }

  var adapter = {
    directions: { both: 'Upstream and downstream', out: 'Downstream only (what follows from it)',
      'in': 'Upstream only (what it comes from)' },

    load: function (data) {
      g = { data: data, nodes: {}, out: {}, inn: {}, pipelines: {}, validationRule: {} };
      data.pipelines.forEach(function (p) { g.pipelines[p.id] = p; });
      data.nodes.forEach(function (n) { g.nodes[n.id] = n; });
      data.edges.forEach(function (e) {
        (g.out[e.source] = g.out[e.source] || []).push(e);
        (g.inn[e.target] = g.inn[e.target] || []).push(e);
      });
      // a rule whose every output is a validation product is a validation rule
      data.nodes.forEach(function (n) {
        if (n.kind !== 'rule') return;
        var outputs = (g.out[n.id] || []).filter(function (e) { return e.kind === 'produces' || e.kind === 'validates'; });
        g.validationRule[n.id] = outputs.length > 0 && outputs.every(function (e) { return e.kind === 'validates'; });
      });
    },

    controls: function () {
      return [
        { id: 'view', label: 'View', type: 'radio', options: VIEWS.map(function (v, i) {
          return { value: v.value, label: v.label, checked: i === 0 };
        }) },
        { id: 'pipelines', label: 'Pipeline', type: 'checks', options: g.data.pipelines.map(function (p) {
          return { value: p.id, label: p.title, checked: true, swatch: PIPELINE_COLOR[p.id] };
        }) },
        { id: 'kinds', label: 'Edge type', type: 'checks', options: EDGE_KINDS.map(function (k) {
          return { value: k.value, label: k.label, checked: true, swatch: k.color };
        }) },
        { id: 'validation', label: 'Validation plots', type: 'radio', options: [
          { value: 'show', label: 'Show validation rules and products', checked: true },
          { value: 'hide', label: 'Hide them' }] },
        { id: 'aggregate', label: 'Aggregate targets', type: 'radio', options: [
          { value: 'hide', label: 'Hide all / configured_products', checked: true },
          { value: 'show', label: 'Show them (every final product points at them)' }] }
      ];
    },

    style: function () {
      return [
        { selector: 'node', style: { 'font-size': 10, 'text-wrap': 'wrap', 'text-max-width': 140 } },
        { selector: 'node.pg-rule', style: { 'shape': 'round-rectangle', 'width': 'label', 'height': 18, 'padding': 6,
          'text-valign': 'center', 'text-margin-y': 0, 'color': '#ffffff', 'text-outline-width': 0 } },
        { selector: 'node.pg-checkpoint', style: { 'shape': 'cut-rectangle' } },
        { selector: 'node.pg-aggregate', style: { 'background-color': '#8c959f' } },
        { selector: 'node.pg-validation-rule', style: { 'background-color': '#1a7f37' } },
        { selector: 'node.pg-publish-rule', style: { 'background-color': '#8250df' } },
        { selector: 'node.pg-artifact', style: { 'shape': 'rectangle', 'font-size': 9 } },
        { selector: 'node.pg-stage', style: { 'shape': 'hexagon', 'font-size': 12, 'font-weight': 'bold' } },
        { selector: 'node.pg-optional', style: { 'border-style': 'dashed', 'border-width': 2 } },
        { selector: 'node.pg-ids', style: { 'shape': 'ellipse', 'font-size': 9 } },
        { selector: 'node.pg-source', style: { 'shape': 'barrel', 'font-size': 12 } },
        { selector: 'edge.pg-scientific_reference', style: { 'line-style': 'dashed', 'width': 2.5 } },
        { selector: 'edge.pg-consumes', style: { 'line-style': 'dotted' } },
        { selector: 'edge.pg-records', style: { 'line-style': 'dotted' } },
        { selector: 'edge.pg-deferred', style: { 'line-style': 'dashed', 'opacity': 0.5 } }
      ];
    },

    elements: function (state) {
      var pipelines = {}, kinds = {};
      state.pipelines.forEach(function (p) { pipelines[p] = true; });
      state.kinds.forEach(function (k) { kinds[k] = true; });
      var hideValidation = state.validation === 'hide';
      var nodes = [], shown = {};
      g.data.nodes.forEach(function (n) {
        if (n.views.indexOf(state.view) === -1) return;
        var owner = ownerOf(n);
        if (owner && !pipelines[owner] && n.kind !== 'source') return;
        if (state.aggregate !== 'show' && isAggregate(n)) return;
        if (hideValidation && isValidation(n)) return;
        shown[n.id] = true;
        nodes.push({ classes: classesOf(n), data: { id: n.id, label: labelOf(n), color: colorOf(n), size: sizeOf(n), kind: n.kind } });
      });
      var edges = [];
      g.data.edges.forEach(function (e, index) {
        if (e.views.indexOf(state.view) === -1 || !kinds[e.kind]) return;
        if (!shown[e.source] || !shown[e.target]) return;
        edges.push({ classes: 'pg-' + e.kind + (e.deferred ? ' pg-deferred' : ''),
          data: { id: 'pe' + index, source: e.source, target: e.target, kind: e.kind, width: 1.5, color: edgeColor(e.kind) } });
      });
      // A source with nothing publishing to it under these filters is noise.
      if (state.view === 'publication') {
        var touched = {};
        edges.forEach(function (e) { touched[e.data.source] = touched[e.data.target] = true; });
        nodes = nodes.filter(function (n) { return n.data.kind !== 'source' || touched[n.data.id]; });
      }
      return {
        nodes: nodes, edges: edges,
        layout: layout(1),
        focusLayout: function () { return layout(1.4); }
      };
    },

    searchItems: function () {
      return g.data.nodes.map(function (n) {
        return { id: n.id, hint: n.kind + (n.pipeline ? ' · ' + n.pipeline : '') + (n.path ? ' · ' + n.path : '') };
      });
    },

    resolveSearch: function (id, v) {
      var n = g.nodes[id];
      if (!n) return id;
      var state = {};
      if (n.views.indexOf(v.state.view) === -1) state.view = n.views[0];
      // whatever elements() would hide this node for is switched back on
      var owner = ownerOf(n);
      if (owner && n.kind !== 'source' && v.state.pipelines.indexOf(owner) === -1) state.pipelines = v.state.pipelines.concat([owner]);
      if (v.state.validation === 'hide' && isValidation(n)) state.validation = 'show';
      if (v.state.aggregate !== 'show' && isAggregate(n)) state.aggregate = 'show';
      return { focus: id, state: state };
    },

    details: function (id, state, v) {
      var n = g.nodes[id];
      if (!n) return '<p>' + v.escape(id) + '</p>';
      return ({ rule: ruleDetails, job: jobDetails, artifact: artifactDetails, stage: stageDetails, ids: idsDetails, source: sourceDetails }[n.kind] || genericDetails)(n, v);
    }
  };

  function ownerOf(n) {
    var stage = n.stage && g.nodes['stage:' + n.stage];
    return n.pipeline || n.produced_by || (stage ? stage.produced_by : '');
  }

  function isAggregate(n) {
    return !!(n.aggregate || (n.kind === 'job' && g.nodes[n.rule] && g.nodes[n.rule].aggregate));
  }

  function isValidation(n) {
    return !!((n.kind === 'rule' && g.validationRule[n.id]) || (n.kind === 'artifact' && n.role === 'validation') ||
      (n.kind === 'job' && g.validationRule[n.rule]));
  }

  function classesOf(n) {
    var c = ['pg-' + n.kind];
    if (n.kind === 'rule') {
      if (n.checkpoint) c.push('pg-checkpoint');
      if (n.aggregate) c.push('pg-aggregate');
      if (g.validationRule[n.id]) c.push('pg-validation-rule');
      if (/^replicate_/.test(n.label)) c.push('pg-publish-rule');
    }
    if (n.kind === 'stage' && n.optional) c.push('pg-optional');
    if (n.kind === 'source' && n.sparse) c.push('pg-optional');
    return c.join(' ');
  }

  function labelOf(n) {
    if (n.kind === 'job') return n.label + (n.wildcards.length ? '\n' + n.wildcards.join(', ') : '');
    if (n.kind === 'artifact') return n.product ? n.product : n.label;
    return n.label;
  }

  function colorOf(n) {
    if (n.kind === 'rule' || n.kind === 'job') return PIPELINE_COLOR[n.pipeline] || '#8c959f';
    if (n.kind === 'artifact') return ROLE_COLOR[n.role] || '#eaeef2';
    if (n.kind === 'stage') return PIPELINE_COLOR[n.produced_by] || '#d0d7de';
    if (n.kind === 'ids') return '#fff8c5';
    if (n.kind === 'source') return '#e8d8ff';
    return '#d0d7de';
  }

  function sizeOf(n) {
    return { stage: 30, source: 34, ids: 16, artifact: 16, job: 18 }[n.kind] || 20;
  }

  function list(v, edges, side, extra) {
    if (!edges.length) return '<p class="vg-meta">None.</p>';
    return '<ul class="vg-list">' + edges.map(function (e) {
      var other = g.nodes[e[side]];
      var name = other ? labelOf(other).split('\n')[0] : e[side];
      return '<li>' + v.nodeButton(e[side], name) + (extra ? extra(e, other) : '') + '</li>';
    }).join('') + '</ul>';
  }

  function by(edges, kinds) {
    return edges.filter(function (e) { return kinds.indexOf(e.kind) !== -1; });
  }

  function referenceNote(e) {
    return ' <span class="vg-meta">param <code>' + window.VaftGraph.escapeHtml(e.param) + '</code>: ' +
      window.VaftGraph.escapeHtml(e.note) + '</span>';
  }

  function ruleDetails(n, v) {
    var out = g.out[n.id] || [], inn = g.inn[n.id] || [];
    var flags = [];
    if (n.checkpoint) flags.push('checkpoint');
    if (n.aggregate) flags.push('aggregate target');
    if (g.validationRule[n.id]) flags.push('validation');
    var html = '<h3 class="vg-title"><code>' + v.escape(n.label) + '</code></h3><p class="vg-meta">rule · ' +
      v.escape(pipelineTitle(n.pipeline)) + (flags.length ? ' · ' + flags.join(', ') : '') + '</p>';
    if (n.label === 'configured_products') {
      html += '<p class="vg-note">What <code>rule all</code> reaches for the configured shots with the preflight and IMPA ' +
        'checkpoints taken as passed: a dry run cannot see past a checkpoint, so the documentation asks for this target. ' +
        'It is never the default and never run.</p>';
    }
    if (n.checkpoint) html += '<p class="vg-note">A checkpoint: what runs after it is decided from its output at run time.</p>';
    html += '<p class="vg-actions">' + [v.sourceLink(n.source, 'Snakefile'), n.script ? v.sourceLink({ path: n.script, line: 0 }, 'Script') : '']
      .filter(Boolean).join(' · ') + '</p>';
    html += '<h4>Runs after</h4>' + list(v, by(inn, ['execution']), 'source');
    html += '<h4>Runs before</h4>' + list(v, by(out, ['execution']), 'target');
    var refs = by(inn, ['scientific_reference']).filter(function (e) { return e.views.indexOf('rules') !== -1; });
    if (refs.length) html += '<h4>Scientific references (consulted, not scheduled)</h4>' + list(v, refs, 'source', referenceNote);
    var consulted = by(out, ['scientific_reference']).filter(function (e) { return e.views.indexOf('rules') !== -1; });
    if (consulted.length) html += '<h4>Consulted by (not scheduled)</h4>' + list(v, consulted, 'target', referenceNote);
    html += '<h4>Inputs</h4>' + list(v, by(inn, ['consumes']), 'source');
    html += '<h4>Outputs</h4>' + list(v, by(out, ['produces', 'validates']), 'target');
    return html;
  }

  function jobDetails(n, v) {
    var html = '<h3 class="vg-title"><code>' + v.escape(n.label) + '</code></h3><p class="vg-meta">job of ' +
      v.nodeButton(n.rule, n.label) + ' · ' + v.escape(pipelineTitle(n.pipeline)) + ', shot ' +
      v.escape(g.pipelines[n.pipeline].shot) + '</p>';
    if (n.wildcards.length) html += '<p class="vg-meta">' + n.wildcards.map(v.escape).join('<br>') + '</p>';
    html += '<h4>After</h4>' + list(v, by(g.inn[n.id] || [], ['execution']), 'source');
    html += '<h4>Before</h4>' + list(v, by(g.out[n.id] || [], ['execution']), 'target');
    var refs = by(g.inn[n.id] || [], ['scientific_reference']);
    if (refs.length) html += '<h4>Scientific references</h4>' + list(v, refs, 'source', referenceNote);
    return html;
  }

  function artifactDetails(n, v) {
    var inn = g.inn[n.id] || [], out = g.out[n.id] || [];
    var html = '<h3 class="vg-title"><code>' + v.escape(n.label) + '</code></h3><p class="vg-meta"><code>' + v.escape(n.path) + '</code></p>' +
      '<p>' + v.escape(ROLE_TEXT[n.role] || n.role) + '.</p>';
    html += '<p class="vg-meta">' + (n.product ? 'PipelinePaths product <code>' + v.escape(n.product) + '</code>' : 'not a single PipelinePaths product') +
      (n.stage ? ' · stage ' + v.nodeButton('stage:' + n.stage, n.stage) : '') + ' · ' + v.escape(pipelineTitle(n.pipeline)) + '</p>';
    html += '<h4>Produced by</h4>' + list(v, by(inn, ['produces', 'validates']), 'source');
    html += '<h4>Consumed by</h4>' + list(v, by(out, ['consumes']), 'target');
    var refs = by(out, ['scientific_reference']).filter(function (e) { return e.views.indexOf('artifacts') !== -1; });
    if (refs.length) html += '<h4>Consulted by (not scheduled)</h4>' + list(v, refs, 'target', referenceNote);
    return html;
  }

  function stageDetails(n, v) {
    var out = g.out[n.id] || [];
    var html = '<h3 class="vg-title"><code>' + v.escape(n.label) + '</code></h3><p class="vg-meta">stage · produced by the ' +
      v.escape(n.produced_by) + ' pipeline' + (n.optional ? ' · optional' : '') + '</p>';
    if (n.optional) html += '<p class="vg-note">Optional: a shot without this product is a normal, correctly published partial state, not a failure.</p>';
    if (n.deferred_to) html += '<p class="vg-note">Replication deferred to ' + v.escape(n.deferred_to) + ': the destination is planned but not wired.</p>';
    if (!n.destination) html += '<p class="vg-note">No per-shot HSDS destination.</p>';
    if (n.note) html += '<p class="vg-meta">' + v.escape(n.note) + '</p>';
    html += '<p class="vg-actions">' + [v.sourceLink(n.source, 'STAGE_REPLICATION entry'),
      v.link('/reference/database-data-sources/', 'Database and data sources')].filter(Boolean).join(' · ') + '</p>';
    html += '<h4>Owns (publishes only these IDS)</h4>' + list(v, by(out, ['owns']), 'target');
    html += '<h4>Replicated by</h4>' + list(v, by(out, ['replicated_by']), 'target');
    return html;
  }

  function idsDetails(n, v) {
    return '<h3 class="vg-title"><code>' + v.escape(n.label) + '</code></h3><p class="vg-meta">IDS owned by stage ' +
      v.nodeButton('stage:' + n.stage, n.stage) + '</p><h4>Published to</h4>' + list(v, by(g.out[n.id] || [], ['publishes']), 'target');
  }

  function sourceDetails(n, v) {
    var inn = g.inn[n.id] || [];
    var html = '<h3 class="vg-title"><code>' + v.escape(n.label) + '</code></h3><p class="vg-meta">HSDS source' +
      (n.parent ? ' under ' + v.nodeButton('source:' + n.parent, n.parent) : '') + (n.sparse ? ' · sparse' : '') +
      (n.writable ? '' : ' · read-only alias') + '</p>';
    if (n.purpose) html += '<p>' + v.escape(n.purpose) + '</p>';
    if (n.sparse) html += '<p class="vg-note">Sparse: a shot missing here means no published product, never that the shot or the baseline failed.</p>';
    html += '<p class="vg-actions">' + [v.sourceLink(n.source, 'Source registry'), v.link('/reference/database-data-sources/', 'Database and data sources')]
      .filter(Boolean).join(' · ') + '</p>';
    html += '<h4>Receives</h4>' + list(v, by(inn, ['publishes']), 'source');
    var children = by(inn, ['parent']);
    if (children.length) html += '<h4>Child sources</h4>' + list(v, children, 'source');
    return html;
  }

  function genericDetails(n, v) {
    return '<h3 class="vg-title"><code>' + v.escape(n.label || n.id) + '</code></h3>';
  }

  window.VaftGraphAdapters = window.VaftGraphAdapters || {};
  window.VaftGraphAdapters.pipeline = adapter;
})();
