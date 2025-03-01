using Literate

const TUTORIALS = ["collision.jl",
                   "cpdi.jl",
                   "tlmpm_vortex.jl",
                   "implicit_jacobian_free.jl",
                   "implicit_jacobian_based.jl",
                   "collapse.jl",
                   "dam_break.jl",
                   "rigid_body_contact.jl",
                   "heat.jl"]

for filename in TUTORIALS
    path = joinpath(@__DIR__, "literate/tutorials", filename)
    outputdir = joinpath(@__DIR__, "src/tutorials")
    Literate.markdown(path, outputdir; codefence="```julia" => "```")
end
