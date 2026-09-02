#!/usr/bin/env ruby
# frozen_string_literal: true

require "yaml"

commit = ARGV.fetch(0)
abort "expected a full 40-character commit SHA" unless commit.match?(/\A[0-9a-f]{40}\z/)

path = File.expand_path("../_data/notebook_outputs.yml", __dir__)
data = YAML.safe_load_file(path, aliases: true)
data["source_commit"] = commit
data.fetch("outputs").each_value do |output|
  output["source_url"] = "https://github.com/VEST-Tokamak/vaft/blob/#{commit}/#{output.fetch('notebook_path')}"
end
File.write(path, data.to_yaml(line_width: -1))
puts "Pinned notebook provenance to #{commit}"
