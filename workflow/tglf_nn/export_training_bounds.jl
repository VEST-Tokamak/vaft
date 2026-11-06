# Export the TGLF-NN training bounds (xbounds) beside each ONNX family (issue #2274, phase A).
#
# The ONNX distribution of TurbulentTransport.jl carries the normalisation moments
# (xm/xsigma) but not the per-input training min/max; those live only inside the
# Julia .bson twin of every family. This script reads them with BSON alone -- no Flux,
# no TurbulentTransport load -- and writes a copy of the checkout's ONNX families with
# one extra file per family:
#
#     <out>/Project.toml, <out>/models/SEMVER           (copied: version provenance)
#     <out>/models/<family>/*.onnx, *.txt                (copied unchanged)
#     <out>/models/<family>/xbounds.txt                  (new: "min max" per input, xnames order)
#     <out>/models/<family>/xbounds.source               (new: key=value provenance)
#
# The source depot is never modified. Point VAFT at the copy with
# TURBULENTTRANSPORTHOME=<out> or model_dir=<out>/models.
#
# A family is exported only when its .bson is provably the ONNX export's twin: the
# input names must match xnames.txt in order, and xm must match xm.txt. Ensemble
# members whose bounds differ are combined into the intersection (the range every
# member saw), and the source file says so.
#
# Usage:  julia --startup-file=no export_training_bounds.jl <TurbulentTransport checkout> <out>
#
# BSON must be loadable: it is resolved by UUID from the active environment's manifest
# (a FUSE/TurbulentTransport environment carries it), so it need not be a direct dependency.

using SHA

const BSON = Base.require(Base.PkgId(Base.UUID("fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"), "BSON"))

raise(x) = BSON.raise_recursive(x, IdDict(), Main)

"""The ensemble members of a raw-parsed .bson: each a (keys, raw values) pair."""
function members(raw)
    tagged_dict(x) = x isa AbstractDict && get(x, :tag, nothing) == "struct" && raise(x[:type]) <: AbstractDict
    if tagged_dict(raw)
        keys_ = raise(raw[:data][1])
        if all(k -> k isa Integer, keys_)          # an ensemble: Dict{Int, model}
            return [(raise(m[:data][1]), m[:data][2]) for m in raw[:data][2]]
        end
        return [(keys_, raw[:data][2])]            # a single model
    end
    error("unrecognised .bson layout")
end

field(member, name) = raise(member[2][findfirst(==(name), member[1])])

function export_family(src::AbstractString, bson::AbstractString, dst::AbstractString)
    xnames_txt = split(read(joinpath(src, "xnames.txt"), String))
    xm_txt = parse.(Float64, split(read(joinpath(src, "xm.txt"), String)))
    ms = members(BSON.parse(bson))
    lows = Vector{Vector{Float64}}(); highs = Vector{Vector{Float64}}()
    for m in ms
        names = String.(vec(field(m, :xnames)))
        names == xnames_txt || return "input names differ from xnames.txt"
        xm = Float64.(vec(field(m, :xm)))
        isapprox(xm, xm_txt; rtol=1e-5, atol=1e-6) || return "xm differs from xm.txt: not this export's twin"
        b = Float64.(field(m, :xbounds))
        size(b) == (length(names), 2) || return "xbounds has shape $(size(b))"
        all(b[:, 1] .<= b[:, 2]) || return "xbounds has min > max"
        push!(lows, b[:, 1]); push!(highs, b[:, 2])
    end
    low = reduce((a, b) -> max.(a, b), lows)
    high = reduce((a, b) -> min.(a, b), highs)
    agree = all(l -> l == lows[1], lows) && all(h -> h == highs[1], highs)
    mkpath(dst)
    for f in readdir(src)
        isfile(joinpath(src, f)) && cp(joinpath(src, f), joinpath(dst, f); force=true)
    end
    open(joinpath(dst, "xbounds.txt"), "w") do io
        for i in eachindex(low)
            println(io, string(low[i]), " ", string(high[i]))
        end
    end
    open(joinpath(dst, "xbounds.source"), "w") do io
        println(io, "bson=", basename(bson))
        println(io, "bson_sha256=", bytes2hex(open(sha256, bson)))
        println(io, "members=", length(ms))
        println(io, "members_agree=", agree ? 1 : 0)
        println(io, "combination=", agree ? "identical" : "intersection")
        println(io, "space=physical, log10 for *_log10 inputs (the xnames convention)")
    end
    return nothing
end

function main(args)
    length(args) == 2 || error("usage: export_training_bounds.jl <TurbulentTransport checkout> <out>")
    checkout, out = abspath(args[1]), abspath(args[2])
    models = joinpath(checkout, "models")
    isdir(models) || error("$models is not a directory")
    abspath(out) == checkout && error("refusing to write into the source checkout")
    mkpath(joinpath(out, "models"))
    for f in ("Project.toml",)
        isfile(joinpath(checkout, f)) && cp(joinpath(checkout, f), joinpath(out, f); force=true)
    end
    isfile(joinpath(models, "SEMVER")) && cp(joinpath(models, "SEMVER"), joinpath(out, "models", "SEMVER"); force=true)
    exported, refused = 0, 0
    for name in sort(readdir(models))
        src = joinpath(models, name)
        isdir(src) && isfile(joinpath(src, "xnames.txt")) || continue
        any(endswith(".onnx"), readdir(src)) || continue
        bson = joinpath(models, name * ".bson")
        if !isfile(bson)
            println("skip  $name: no $(name).bson twin"); refused += 1; continue
        end
        why = export_family(src, bson, joinpath(out, "models", name))
        if why === nothing
            println("ok    $name"); exported += 1
        else
            println("skip  $name: $why"); refused += 1
        end
    end
    println("exported $exported, skipped $refused -> $(joinpath(out, "models"))")
end

main(ARGS)
