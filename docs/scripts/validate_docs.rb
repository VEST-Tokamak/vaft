#!/usr/bin/env ruby
# frozen_string_literal: true

require "digest"
require "nokogiri"
require "pathname"
require "set"
require "uri"
require "yaml"

ROOT = Pathname(__dir__).parent

# The site is built twice from one source tree -- once for the stable track at
# /vaft and once for the development track at /vaft/develop -- so neither the
# baseurl nor the build destination can be hard-coded here.  Defaults match a
# plain `bundle exec jekyll build` of the stable track.
BASEURL = ENV.fetch("VAFT_DOCS_BASEURL", "/vaft").chomp("/")
SITE = Pathname(ENV.fetch("VAFT_DOCS_SITE", (ROOT / "_site").to_s))

# ROOT is the docs/ directory, so its parent is the checkout being documented.
SOURCE_ROOT = Pathname(ENV.fetch("VAFT_NOTEBOOK_SOURCE", ROOT.parent.to_s))

abort "VAFT_DOCS_BASEURL must start with '/' (got #{BASEURL.inspect})" unless BASEURL.start_with?("/")
abort "no built site at #{SITE} -- run `bundle exec jekyll build` first" unless SITE.directory?

errors = []

def data(name)
  YAML.safe_load_file(ROOT / "_data" / name, aliases: true)
end

def output_path(url)
  path = url.sub(/\A#{Regexp.escape(BASEURL)}/, "").sub(%r{\A/}, "")
  candidate = SITE / path
  return candidate if candidate.file?
  return candidate / "index.html" if (candidate / "index.html").file?
  html = SITE / "#{path}.html"
  html if html.file?
end

navigation = data("navigation.yml").fetch("sections")
items = navigation.flat_map { |section| section.fetch("items") }
%w[id url].each do |field|
  values = items.map { |item| item.fetch(field) }
  duplicates = values.tally.select { |_value, count| count > 1 }.keys
  errors << "duplicate navigation #{field}: #{duplicates.join(', ')}" unless duplicates.empty?
end
canonical_urls = items.map { |item| item.fetch("url") }.to_set
canonical_urls.each do |url|
  errors << "canonical navigation target is not built: #{url}" unless output_path(url)
end

# generators.yml is what the branch says its documentation is generated from.
# Each declared output is produced by docs/build.py; none of them is committed,
# so a missing one means the site was built without regenerating its data.
generators_path = ROOT / "generators.yml"
declared_outputs = []
if generators_path.file?
  declared_outputs = (YAML.safe_load_file(generators_path)["generators"] || []).map { |g| g["output"] }
  declared_outputs.each do |output|
    next if (ROOT / output).file?
    errors << "declared generator output is missing: #{output} (run `python docs/build.py`)"
  end
end

diagnostic_snapshot = data("vest_diagnostics.yml")
%w[schema_version source diagnostics].each do |field|
  errors << "diagnostic snapshot missing #{field}" unless diagnostic_snapshot.key?(field)
end
source = diagnostic_snapshot.fetch("source", {})
errors << "diagnostic snapshot source checksum is invalid" unless source["sha256"].to_s.match?(/\A[0-9a-f]{64}\z/)
diagnostics = diagnostic_snapshot.fetch("diagnostics", [])
errors << "diagnostic snapshot has no diagnostics" unless diagnostics.is_a?(Array) && !diagnostics.empty?
ids = diagnostics.map { |item| item["id"] }
errors << "diagnostic snapshot has duplicate IDs" unless ids.uniq.length == ids.length
diagnostics.each do |item|
  %w[id name ids ids_path responsible source availability lifecycle mapping_status].each do |field|
    errors << "diagnostic #{item['id'] || '(unknown)'} missing #{field}" if item[field].nil?
  end
end
registry_source = ENV["VAFT_REGISTRY_SOURCE"]
if registry_source && !registry_source.empty?
  registry_path = Pathname(registry_source) / "vaft/machine_mapping/vest.yaml"
  if registry_path.file?
    actual = Digest::SHA256.file(registry_path).hexdigest
    errors << "diagnostic snapshot does not match VAFT_REGISTRY_SOURCE" unless actual == source["sha256"]
  else
    errors << "VAFT_REGISTRY_SOURCE has no vest.yaml: #{registry_path}"
  end
end

# vaft.formula.catalog reached develop after main, so the formula reference is
# present on some branches and not others.  Validate it when the branch ships it.
formula_snapshot = nil
if (ROOT / "_data" / "formula_catalog.yml").file?
  formula_snapshot = data("formula_catalog.yml")
  %w[schema_version generator source categories formulas].each do |field|
    errors << "formula snapshot missing #{field}" unless formula_snapshot.key?(field)
  end
  formula_sources = formula_snapshot.fetch("source", [])
  errors << "formula snapshot has no sources" unless formula_sources.is_a?(Array) && !formula_sources.empty?
  formula_sources.each do |entry|
    errors << "formula snapshot source checksum is invalid: #{entry['path']}" unless entry["sha256"].to_s.match?(/\A[0-9a-f]{64}\z/)
  end
  formula_categories = formula_snapshot.fetch("categories", []).map { |item| item["name"] }
  formulas = formula_snapshot.fetch("formulas", [])
  errors << "formula snapshot has no formulas" unless formulas.is_a?(Array) && !formulas.empty?
  formula_ids = formulas.map { |item| item["id"] }
  errors << "formula snapshot has duplicate IDs" unless formula_ids.uniq.length == formula_ids.length
  formulas.each do |item|
    %w[id name category signature summary parameters returns sections references raises source empirical convention_sensitive].each do |field|
      errors << "formula #{item['id'] || '(unknown)'} missing #{field}" if item[field].nil?
    end
    errors << "formula #{item['id']} has no source location" unless item["source"].is_a?(Hash) && item["source"]["path"].to_s.start_with?("vaft/formula/") && item["source"]["line"].to_i.positive?
    errors << "formula #{item['id']} has unknown category #{item['category']}" unless formula_categories.include?(item["category"])
  end
  if registry_source && !registry_source.empty?
    formula_sources.each do |entry|
      source_path = Pathname(registry_source) / entry.fetch("path")
      if source_path.file?
        actual = Digest::SHA256.file(source_path).hexdigest
        errors << "formula snapshot does not match VAFT_REGISTRY_SOURCE: #{entry['path']}" unless actual == entry["sha256"]
      else
        errors << "VAFT_REGISTRY_SOURCE has no #{entry['path']}"
      end
    end
  end
end

process_snapshot = nil
if (ROOT / "_data" / "process_catalog.yml").file?
  process_snapshot = data("process_catalog.yml")
  %w[schema_version generator source categories functions].each do |field|
    errors << "process snapshot missing #{field}" unless process_snapshot.key?(field)
  end
  process_sources = process_snapshot.fetch("source", [])
  errors << "process snapshot has no sources" unless process_sources.is_a?(Array) && !process_sources.empty?
  process_sources.each do |entry|
    errors << "process snapshot source checksum is invalid: #{entry['path']}" unless entry["sha256"].to_s.match?(/\A[0-9a-f]{64}\z/)
  end
  process_categories = process_snapshot.fetch("categories", [])
  process_category_names = process_categories.map { |item| item["name"] }
  process_categories.each do |item|
    %w[name module count documented conforming].each do |field|
      errors << "process category #{item['name'] || '(unknown)'} missing #{field}" if item[field].nil?
    end
    errors << "process category #{item['name']} claims conforming with #{item['documented']}/#{item['count']}" if item["conforming"] && item["documented"] != item["count"]
  end
  functions = process_snapshot.fetch("functions", [])
  errors << "process snapshot has no functions" unless functions.is_a?(Array) && !functions.empty?
  function_ids = functions.map { |item| item["id"] }
  errors << "process snapshot has duplicate IDs" unless function_ids.uniq.length == function_ids.length
  functions.each do |item|
    # machine_scope is legitimately nil until a category is brought under the contract.
    %w[id name category signature summary parameters returns sections provenance raises source convention_sensitive conforming errors].each do |field|
      errors << "process function #{item['id'] || '(unknown)'} missing #{field}" if item[field].nil?
    end
    errors << "process function #{item['id']} has no source location" unless item["source"].is_a?(Hash) && item["source"]["path"].to_s.start_with?("vaft/process/") && item["source"]["line"].to_i.positive?
    errors << "process function #{item['id']} has unknown category #{item['category']}" unless process_category_names.include?(item["category"])
    errors << "process function #{item['id']} has unknown machine_scope #{item['machine_scope']}" unless [nil, "independent", "vest"].include?(item["machine_scope"])
    errors << "process function #{item['id']} is conforming but lists errors" if item["conforming"] && !item["errors"].empty?
  end
  # Every conforming category must have its reference page, and no pending one may.
  process_categories.each do |item|
    page = ROOT / "_guide" / "Process_reference_#{item['name']}.md"
    if item["conforming"]
      errors << "conforming process category #{item['name']} has no reference page" unless page.file?
    elsif page.file?
      errors << "pending process category #{item['name']} has a reference page it cannot fill"
    end
  end
  if registry_source && !registry_source.empty?
    process_sources.each do |entry|
      source_path = Pathname(registry_source) / entry.fetch("path")
      if source_path.file?
        actual = Digest::SHA256.file(source_path).hexdigest
        errors << "process snapshot does not match VAFT_REGISTRY_SOURCE: #{entry['path']}" unless actual == entry["sha256"]
      else
        errors << "VAFT_REGISTRY_SOURCE has no #{entry['path']}"
      end
    end
  end
end

# vaft.plot.docs_catalog and vaft.diagram.docs_catalog: same shape of checks as
# the formula and process snapshots, on the branches that ship them.
def check_snapshot_sources(errors, label, snapshot, registry_source)
  sources = snapshot.fetch("source", [])
  errors << "#{label} snapshot has no sources" unless sources.is_a?(Array) && !sources.empty?
  sources.each do |entry|
    errors << "#{label} snapshot source checksum is invalid: #{entry['path']}" unless entry["sha256"].to_s.match?(/\A[0-9a-f]{64}\z/)
    next if registry_source.nil? || registry_source.empty?
    source_path = Pathname(registry_source) / entry.fetch("path")
    if source_path.file?
      errors << "#{label} snapshot does not match VAFT_REGISTRY_SOURCE: #{entry['path']}" unless Digest::SHA256.file(source_path).hexdigest == entry["sha256"]
    else
      errors << "VAFT_REGISTRY_SOURCE has no #{entry['path']}"
    end
  end
end

def require_fields(errors, label, rows, fields)
  ids = rows.map { |item| item["id"] }
  errors << "#{label} snapshot has duplicate IDs" unless ids.uniq.length == ids.length
  rows.each do |item|
    fields.each do |field|
      errors << "#{label} #{item['id'] || '(unknown)'} missing #{field}" if item[field].nil?
    end
  end
end

plot_snapshot = nil
if (ROOT / "_data" / "plot_catalog.yml").file?
  plot_snapshot = data("plot_catalog.yml")
  %w[schema_version generator source views subjects plots entry_points].each do |field|
    errors << "plot snapshot missing #{field}" unless plot_snapshot.key?(field)
  end
  check_snapshot_sources(errors, "plot", plot_snapshot, registry_source)
  plots = plot_snapshot.fetch("plots", [])
  errors << "plot snapshot has no plots" unless plots.is_a?(Array) && !plots.empty?
  require_fields(errors, "plot", plots, %w[id name status subject view quantity domain description model adapter renderer ids required_paths optional_paths backends source])
  subjects = plot_snapshot.fetch("subjects", []).map { |item| item["name"] }
  views = plot_snapshot.fetch("views", [])
  plots.each do |item|
    errors << "plot #{item['id']} has unknown subject #{item['subject']}" unless subjects.include?(item["subject"])
    errors << "plot #{item['id']} has unknown view #{item['view']}" unless views.include?(item["view"])
    errors << "plot #{item['id']} has no source location" unless item["source"].is_a?(Hash) && item["source"]["path"].to_s.start_with?("vaft/plot/") && item["source"]["line"].to_i.positive?
  end
  entry_points = plot_snapshot.fetch("entry_points", [])
  require_fields(errors, "plot function", entry_points, %w[id name status module summary signature source])
  overlap = plots.map { |item| item["name"] } & entry_points.map { |item| item["name"] }
  errors << "plot names listed both as registered plots and as plot functions: #{overlap.join(', ')}" unless overlap.empty?
end

diagram_snapshot = nil
if (ROOT / "_data" / "diagram_catalog.yml").file?
  diagram_snapshot = data("diagram_catalog.yml")
  %w[schema_version generator source families builders assets].each do |field|
    errors << "diagram snapshot missing #{field}" unless diagram_snapshot.key?(field)
  end
  check_snapshot_sources(errors, "diagram", diagram_snapshot, registry_source)
  builders = diagram_snapshot.fetch("builders", [])
  assets = diagram_snapshot.fetch("assets", [])
  errors << "diagram snapshot has no builders" unless builders.is_a?(Array) && !builders.empty?
  require_fields(errors, "diagram builder", builders, %w[id name family module summary signature formula assets source])
  require_fields(errors, "diagram asset", assets, %w[id asset builder arguments call svg svg_sha256])
  builder_names = builders.map { |item| item["name"] }
  assets.each do |item|
    errors << "diagram asset #{item['asset']} names unknown builder #{item['builder']}" unless builder_names.include?(item["builder"])
    svg = ROOT / item["svg"].to_s
    if svg.file?
      errors << "diagram asset #{item['asset']} does not match its recorded SVG checksum" unless Digest::SHA256.file(svg).hexdigest == item["svg_sha256"]
    else
      errors << "diagram asset #{item['asset']} has no SVG at #{item['svg']}"
    end
  end
  # Hand-written pages may show a diagram only if it is still canonical.
  catalogued_svgs = assets.map { |item| item["asset"] }.to_set
  (ROOT.glob("{_guide,_pages,guide}/*.md") + ROOT.glob("*.{md,markdown,html}")).each do |page|
    page.read(encoding: "UTF-8").scan(%r{/assets/diagrams/([A-Za-z0-9_.-]+\.svg)}).flatten.uniq.each do |name|
      errors << "#{page.relative_path_from(ROOT)} shows #{name}, which is not a canonical diagram" unless catalogued_svgs.include?(name)
    end
  end
end

api_snapshot = nil
if (ROOT / "_data" / "api_catalog.yml").file?
  api_snapshot = data("api_catalog.yml")
  %w[schema_version generator source pages modules entries].each do |field|
    errors << "api snapshot missing #{field}" unless api_snapshot.key?(field)
  end
  check_snapshot_sources(errors, "api", api_snapshot, registry_source)
  api_entries = api_snapshot.fetch("entries", [])
  errors << "api snapshot has no entries" unless api_entries.is_a?(Array) && !api_entries.empty?
  require_fields(errors, "api", api_entries, %w[id name module page kind signature summary deprecated source reference exported_as])
  api_pages = api_snapshot.fetch("pages", []).map { |item| item["slug"] }
  api_entries.each do |item|
    errors << "api #{item['id']} belongs to no page" unless api_pages.include?(item["page"])
    reference = item["reference"].to_s
    next if reference.empty?
    errors << "api #{item['id']} links to #{reference}, which is not built" unless output_path("#{BASEURL}#{reference.split('#').first}")
  end
end

# A generated page whose data was not generated renders as an empty list and
# would otherwise pass everything below.
{ "Plot_reference.md" => "plot_catalog.yml", "Diagram_reference.md" => "diagram_catalog.yml",
  "Api_reference_core.md" => "api_catalog.yml" }.each do |page, snapshot|
  next unless (ROOT / "_guide" / page).file?
  errors << "_guide/#{page} is published but _data/#{snapshot} was not generated (declare its generator in generators.yml)" unless (ROOT / "_data" / snapshot).file?
end

# Every catalog entry is rendered exactly once on its reference page, and the
# page renders nothing the catalog no longer holds.  The pages mark each entry
# with data-catalog, so this reads the built HTML rather than trusting the
# Liquid that produced it.
def rendered_catalog(url, selector, attribute)
  built = output_path(url)
  return nil unless built
  document = Nokogiri::HTML(built.read)
  [document.css(selector).map { |node| node[attribute] }, document.css("[id]").map { |node| node["id"] }.tally]
end

def compare_rendered(errors, label, url, expected, selector, attribute = "id")
  rendered, ids = rendered_catalog(url, selector, attribute)
  if rendered.nil?
    errors << "#{label} reference page is not built: #{url}"
    return
  end
  # An anchor shared with another element (a heading, another entry) sends the
  # link to whichever comes first.
  expected.uniq.each { |name| errors << "#{label} #{name}: id is used more than once on #{url}" if attribute == "id" && ids.fetch(name, 0) > 1 }
  (expected - rendered).uniq.each { |name| errors << "#{label} #{name} is in the catalog but not rendered on #{url}" }
  (rendered - expected).uniq.each { |name| errors << "#{label} #{name} is rendered on #{url} but is not in the catalog" }
  rendered.tally.select { |_name, count| count > 1 }.each_key { |name| errors << "#{label} #{name} is rendered more than once on #{url}" }
end

# Source navigation (#1069).  Every entry with a source span renders exactly one
# [source] link, pinned to the commit its catalog was generated from and to the
# span's line range, never to a branch; when the catalog carries the span's code
# the page shows it once, collapsed, and it is that code.  catalog_coverage.py
# checks the spans against the tree itself, so link, inline source and snapshot
# describe one revision.  `spans` holds every span one page renders, keyed as
# the page's data-source attributes are: an entry's id, "<id>.<member>" for a
# class member.
SITE_PROVENANCE = (ROOT / "_data" / "provenance.yml").file? ? data("provenance.yml") : {}

def check_sources(errors, label, url, snapshot, spans)
  commit = snapshot.dig("provenance", "commit").to_s
  unless commit.match?(/\A[0-9a-f]{40}\z/)
    errors << "#{label} snapshot records no provenance commit, so its source links cannot be pinned (#{url})"
    return
  end
  site_commit = SITE_PROVENANCE["commit"].to_s
  errors << "#{label} snapshot was generated from #{commit[0, 7]}, but this track is built from #{site_commit[0, 7]}" unless site_commit.empty? || site_commit == commit
  built = output_path(url)
  return unless built # compare_rendered reports the missing page

  document = Nokogiri::HTML(built.read)
  links = document.css("a.ref-source").group_by { |node| node["data-source"].to_s }
  shown = document.css("details.ref-code").group_by { |node| node["data-source-code"].to_s }
  located = spans.select { |_key, src| src.is_a?(Hash) && src["line"].to_i.positive? }
  located.each do |key, src|
    href = "https://github.com/VEST-Tokamak/vaft/blob/#{commit}/#{src['path']}#L#{src['line']}-L#{src['end_line']}"
    found = links.fetch(key, [])
    errors << "#{label} #{key}: #{found.length} source links on #{url}, expected 1" unless found.length == 1
    found.each { |node| errors << "#{label} #{key}: source link is #{node['href']}, not #{href}" unless node["href"] == href }
    code = src["code"].to_s
    views = shown.fetch(key, [])
    if code.empty?
      errors << "#{label} #{key}: inline source is shown on #{url} but the catalog has none" unless views.empty?
    elsif views.length != 1
      errors << "#{label} #{key}: #{views.length} inline source views on #{url}, expected 1"
    elsif (pre = views.first.at_css("pre")).nil? || pre.text.chomp != code
      errors << "#{label} #{key}: inline source on #{url} differs from the catalog's"
    end
  end
  (links.keys - located.keys).each { |key| errors << "#{label}: #{url} renders a source link (#{key.empty? ? 'unkeyed' : key}) no catalog entry has" }
  (shown.keys - located.keys).each { |key| errors << "#{label}: #{url} renders inline source (#{key.empty? ? 'unkeyed' : key}) no catalog entry has" }
end

def api_spans(rows)
  rows.each_with_object({}) do |item, spans|
    spans[item["id"]] = item["source"]
    (item["members"] || []).each { |member| spans["#{item['id']}.#{member['name']}"] = member["source"] }
  end
end

if formula_snapshot
  formula_snapshot.fetch("formulas", []).group_by { |item| item["category"] }.each do |category, rows|
    compare_rendered(errors, "formula", "#{BASEURL}/reference/formula/#{category}/", rows.map { |item| item["name"] }, '[data-catalog="formula"]')
    check_sources(errors, "formula", "#{BASEURL}/reference/formula/#{category}/", formula_snapshot, rows.to_h { |item| [item["name"], item["source"]] })
  end
end
if process_snapshot
  conforming = process_snapshot.fetch("categories", []).select { |item| item["conforming"] }.map { |item| item["name"] }
  process_snapshot.fetch("functions", []).group_by { |item| item["category"] }.each do |category, rows|
    # A pending category has no page by design (see above).
    next unless conforming.include?(category)
    compare_rendered(errors, "process", "#{BASEURL}/reference/process/#{category}/", rows.map { |item| item["name"] }, '[data-catalog="process"]')
    check_sources(errors, "process", "#{BASEURL}/reference/process/#{category}/", process_snapshot, rows.to_h { |item| [item["name"], item["source"]] })
  end
end
if plot_snapshot
  compare_rendered(errors, "plot", "#{BASEURL}/reference/plot/", plot_snapshot.fetch("plots", []).map { |item| item["name"] }, '[data-catalog="plot"]')
  compare_rendered(errors, "plot function", "#{BASEURL}/reference/plot/", plot_snapshot.fetch("entry_points", []).map { |item| item["name"] }, '[data-catalog="plot-function"]')
  plot_rows = plot_snapshot.fetch("plots", []) + plot_snapshot.fetch("entry_points", [])
  check_sources(errors, "plot", "#{BASEURL}/reference/plot/", plot_snapshot, plot_rows.to_h { |item| [item["name"], item["source"]] })
  # Committed thumbnails (python -m vaft.plot.docs_thumbnails): one image per
  # rendered plot, none for any other, each resolving to the PNG the catalog
  # recorded.  Staleness is a label on the page, not an error.
  plots = plot_snapshot.fetch("plots", [])
  if (ROOT / "assets" / "plots").directory? || plots.any? { |item| item.key?("thumbnail") }
    missing_thumbnail = plots.reject { |item| item["thumbnail"].is_a?(Hash) }.map { |item| item["name"] }
    errors << "plot entries without a thumbnail record: #{missing_thumbnail.first(5).join(', ')}" unless missing_thumbnail.empty?
    rendered_plots = plots.select { |item| item.dig("thumbnail", "status") == "rendered" }
    compare_rendered(errors, "plot thumbnail", "#{BASEURL}/reference/plot/", rendered_plots.map { |item| item["name"] }, "img[data-thumbnail]", "data-thumbnail")
    rendered_plots.each do |item|
      png = ROOT / item.dig("thumbnail", "png").to_s
      if !png.file?
        errors << "plot thumbnail #{item['name']} has no committed PNG at #{item.dig('thumbnail', 'png')}"
      elsif Digest::SHA256.file(png).hexdigest != item.dig("thumbnail", "png_sha256")
        errors << "plot thumbnail #{item['name']} does not match its recorded checksum"
      end
    end
    if (built = output_path("#{BASEURL}/reference/plot/"))
      Nokogiri::HTML(built.read).css("img[data-thumbnail]").each do |image|
        errors << "plot thumbnail image does not resolve: #{image['src']}" unless output_path(image["src"].to_s)
        expected_src = "#{BASEURL}/assets/plots/#{image['data-thumbnail']}.png"
        errors << "plot thumbnail #{image['data-thumbnail']} shows #{image['src']}, not its own #{expected_src}" unless image["src"] == expected_src
      end
    end
  end
end
if api_snapshot
  entries_by_page = api_snapshot.fetch("entries", []).group_by { |item| item["page"] }
  api_snapshot.fetch("pages", []).each do |page|
    ids = (entries_by_page[page["slug"]] || []).map { |item| item["id"] }
    compare_rendered(errors, "api", "#{BASEURL}/reference/api/#{page['slug']}/", ids, '[data-catalog="api"]')
    check_sources(errors, "api", "#{BASEURL}/reference/api/#{page['slug']}/", api_snapshot, api_spans(entries_by_page[page["slug"]] || []))
  end
end
if diagram_snapshot
  url = "#{BASEURL}/reference/diagram/"
  compare_rendered(errors, "diagram", url, diagram_snapshot.fetch("builders", []).map { |item| item["name"] }, '[data-catalog="diagram"]')
  compare_rendered(errors, "diagram asset", url, diagram_snapshot.fetch("assets", []).map { |item| item["asset"] }, '[data-catalog="diagram-asset"]', "data-asset")
  check_sources(errors, "diagram", url, diagram_snapshot, diagram_snapshot.fetch("builders", []).to_h { |item| [item["name"], item["source"]] })
  if (built = output_path(url))
    Nokogiri::HTML(built.read).css('[data-catalog="diagram-asset"] img').each do |image|
      errors << "diagram gallery image does not resolve: #{image['src']}" unless output_path(image["src"].to_s)
    end
  end
end

migrations = data("page_migrations.yml")
legacy_urls = migrations.map { |item| item.fetch("legacy_url") }
errors << "duplicate legacy URL in page_migrations.yml" unless legacy_urls.uniq.length == legacy_urls.length
redirect_sources = ROOT.glob("_redirects/*.md") + ROOT.glob("_guide/*.md") + [ROOT / "guide/Examples.md"]
declared_redirects = redirect_sources.filter_map do |path|
  next unless path.file?
  front = path.read[/\A---\s*\n(.*?)\n---/m, 1]
  next unless front
  metadata = YAML.safe_load(front, aliases: true) || {}
  metadata["permalink"] if metadata["layout"] == "redirect"
end
unaccounted = declared_redirects.to_set - legacy_urls.to_set
errors << "unaccounted legacy redirect pages: #{unaccounted.to_a.sort.join(', ')}" unless unaccounted.empty?
migrations.each do |migration|
  legacy = migration.fetch("legacy_url")
  target = migration.fetch("canonical_url")
  errors << "redirect target is not canonical: #{legacy} -> #{target}" unless canonical_urls.include?(target)
  built = output_path(legacy)
  if built.nil?
    errors << "legacy URL is not built: #{legacy}"
    next
  end
  html = Nokogiri::HTML(built.read)
  canonical = html.at_css('link[rel="canonical"]')&.[]("href")
  errors << "legacy URL lacks canonical target: #{legacy}" unless canonical&.end_with?("#{BASEURL}#{target}")
end

resources = data("resources.yml")
resource_kinds = { "notebooks" => resources.fetch("notebooks"), "api" => resources.fetch("api"),
                   "data_sources" => resources.fetch("data_sources"),
                   "outputs" => data("notebook_outputs.yml").fetch("outputs") }
resource_refs = Hash.new { |hash, key| hash[key] = Set.new }
(ROOT.glob("_guide/*.md") + ROOT.glob("_pages/*.md")).each do |path|
  front = path.read[/\A---\s*\n(.*?)\n---/m, 1]
  next unless front
  metadata = YAML.safe_load(front, aliases: true) || {}
  related = metadata.fetch("related", {})
  related.each do |kind, ids|
    next unless resource_kinds.key?(kind)
    Array(ids).each do |id|
      resource_refs[kind] << id
      errors << "#{path}: unknown #{kind} resource #{id}" unless resource_kinds[kind].key?(id)
    end
  end
end

resources.fetch("notebooks").each do |id, notebook|
  notebook_path = SOURCE_ROOT / notebook.fetch("path")
  errors << "notebook resource #{id} is missing: #{notebook_path}" unless notebook_path.file?
end

inventory_source = (ROOT / "_guide/Examples.md").read
inventory = inventory_source.scan(/[A-Za-z0-9_]+\.ipynb/).uniq.sort
actual_notebooks = SOURCE_ROOT.glob("notebooks/*.ipynb").map(&:basename).map(&:to_s).sort
missing_inventory = actual_notebooks - inventory
extra_inventory = inventory - actual_notebooks
errors << "notebook inventory omissions: #{missing_inventory.join(', ')}" unless missing_inventory.empty?
errors << "notebook inventory has missing paths: #{extra_inventory.join(', ')}" unless extra_inventory.empty?

provenance = data("notebook_outputs.yml")
allow_pending = ENV["VAFT_ALLOW_PENDING_PROVENANCE"] == "1"

# Outputs whose export cannot be reproduced (see the header of
# _data/notebook_outputs.yml) waive their notebook checksum and their immutable
# source URL, and nothing else. The waiver has to be written down: a reason and
# an issue number, at the file level and again on each output that uses it.
LEGACY = "legacy-unreproducible"
legacy_file = provenance["provenance_status"] == LEGACY
if legacy_file
  %w[provenance_reason provenance_issue].each do |field|
    errors << "legacy provenance requires #{field}" if provenance[field].to_s.strip.empty?
  end
end
%w[source_repository source_commit baseline_commit branch export_command python_version vaft_version dependency_snapshot_sha256 timestamp outputs].each do |field|
  errors << "notebook provenance missing top-level #{field}" if provenance[field].to_s.strip.empty?
end
commit = provenance.fetch("source_commit")
unless commit.match?(/\A[0-9a-f]{40}\z/) || (allow_pending && commit == "PENDING_COMPANION_COMMIT")
  errors << "notebook provenance is not pinned to a companion commit: #{commit}"
end
required_output = %w[notebook_path notebook_sha256 source_url execution_mode data_source shot time_slice artifacts]
provenance.fetch("outputs").each do |id, output|
  missing = required_output.reject { |field| output.key?(field) && !output[field].nil? }
  errors << "output #{id} missing fields: #{missing.join(', ')}" unless missing.empty?
  legacy = output["verification"] == LEGACY
  notebook = SOURCE_ROOT / output.fetch("notebook_path", "")
  if !notebook.file?
    # Required of every output, legacy or not: the notebook it claims to come
    # from has to exist in the checkout being documented.
    errors << "output #{id} notebook path is missing: #{notebook}"
  elsif legacy
    errors << "output #{id} is marked #{LEGACY} but the file is not" unless legacy_file
    %w[verification_reason verification_issue].each do |field|
      errors << "output #{id} #{LEGACY} marker requires #{field}" if output[field].to_s.strip.empty?
    end
  else
    actual = Digest::SHA256.file(notebook).hexdigest
    errors << "output #{id} notebook SHA mismatch" unless actual == output["notebook_sha256"]
    expected_url = "https://github.com/VEST-Tokamak/vaft/blob/#{commit}/#{output['notebook_path']}"
    pending_url = allow_pending && commit == "PENDING_COMPANION_COMMIT" && output["source_url"] == "PENDING_COMPANION_COMMIT"
    errors << "output #{id} source URL is not immutable" unless output["source_url"] == expected_url || pending_url
  end
  Array(output["artifacts"]).each do |artifact|
    %w[path sha256 caption alt].each do |field|
      errors << "output #{id} artifact missing #{field}" if artifact[field].to_s.strip.empty?
    end
    asset = ROOT / artifact.fetch("path", "")
    if asset.file?
      actual = Digest::SHA256.file(asset).hexdigest
      errors << "output #{id} artifact SHA mismatch: #{asset}" unless actual == artifact["sha256"]
    else
      errors << "output #{id} artifact is missing: #{asset}"
    end
  end
  errors << "output #{id} is not referenced by a guide" unless resource_refs["outputs"].include?(id)
end

SITE.glob("**/*.html").each do |page|
  document = Nokogiri::HTML(page.read)
  if document.at_css('[data-resource-error="unknown"]')
    errors << "rendered unknown resource marker: #{page.relative_path_from(SITE)}"
  end
  document.css("a[href]").each do |link|
    href = link["href"]
    next if href.nil? || href.empty? || href == "." || href.start_with?("#", "mailto:", "tel:", "javascript:")
    begin
      uri = URI.parse(href)
    rescue URI::InvalidURIError
      errors << "invalid href in #{page.relative_path_from(SITE)}: #{href}"
      next
    end
    next if uri.scheme || href.start_with?("//")
    path_part = href.split("#", 2).first.split("?", 2).first
    next if path_part.empty?
    target = if path_part.start_with?("/")
               output_path(path_part)
             else
               resolved = (page.dirname / path_part).cleanpath
               resolved = resolved / "index.html" if resolved.directory?
               resolved = Pathname("#{resolved}.html") unless resolved.file? || resolved.extname != ""
               resolved if resolved.file?
             end
    errors << "broken internal link in #{page.relative_path_from(SITE)}: #{href}" unless target
  end
end

if errors.empty?
  puts "Documentation validation passed (#{canonical_urls.length} canonical pages, #{migrations.length} redirects, #{provenance.fetch('outputs').length} outputs)."
else
  warn errors.uniq.sort.join("\n")
  exit 1
end
