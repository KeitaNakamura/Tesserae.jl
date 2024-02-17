using Literate

const EXAMPLES = ["elastic_impact.jl",
                  "tlmpm_vortex.jl"]

for filename in EXAMPLES
    path = joinpath(@__DIR__, "literate/examples", filename)
    outputdir = joinpath(@__DIR__, "src/examples")
    Literate.markdown(path, outputdir; codefence="```julia" => "```")
end
