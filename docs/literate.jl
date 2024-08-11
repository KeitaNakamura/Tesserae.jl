using Literate

const EXAMPLES = ["collision.jl",
                  "tlmpm_vortex.jl",
                  "implicit_jacobian_free.jl",
                  "implicit_jacobian_based.jl",
                  "dam_break.jl",
                  "rigid_body_contact.jl"]

for filename in EXAMPLES
    path = joinpath(@__DIR__, "literate/examples", filename)
    outputdir = joinpath(@__DIR__, "src/examples")
    Literate.markdown(path, outputdir; codefence="```julia" => "```")
end
