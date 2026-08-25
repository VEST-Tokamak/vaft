#!/usr/bin/env ruby
# frozen_string_literal: true

require "digest"
require "nokogiri"
require "pathname"
require "set"
require "uri"
require "yaml"

ROOT = Pathname(__dir__).parent
SITE = ROOT / "_site"
SOURCE_ROOT = Pathname(ENV.fetch("VAFT_NOTEBOOK_SOURCE", ROOT.parent / "vaft-issue60"))
errors = []

def data(name)
  YAML.safe_load_file(ROOT / "_data" / name, aliases: true)
end

def output_path(url)
  path = url.sub(%r{\A/vaft}, "").sub(%r{\A/}, "")
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
  errors << "legacy URL lacks canonical target: #{legacy}" unless canonical&.end_with?("/vaft#{target}")
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
  notebook = SOURCE_ROOT / output.fetch("notebook_path", "")
  if notebook.file?
    actual = Digest::SHA256.file(notebook).hexdigest
    errors << "output #{id} notebook SHA mismatch" unless actual == output["notebook_sha256"]
  else
    errors << "output #{id} notebook path is missing: #{notebook}"
  end
  expected_url = "https://github.com/VEST-Tokamak/vaft/blob/#{commit}/#{output['notebook_path']}"
  pending_url = allow_pending && commit == "PENDING_COMPANION_COMMIT" && output["source_url"] == "PENDING_COMPANION_COMMIT"
  errors << "output #{id} source URL is not immutable" unless output["source_url"] == expected_url || pending_url
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
