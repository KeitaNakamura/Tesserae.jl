using Literate

const SIMULATIONS = ["sand_column_collapse.jl",
                     "dam_break.jl",
                     "axial_vibration_of_bar.jl",
                     "contacting_grains.jl",
                     "hyperelastic_material.jl"]
const MODELS = ["LinearElastic.jl",
                "NeoHookean.jl",
                "DruckerPrager.jl"]

for filename in SIMULATIONS
    path = joinpath(@__DIR__, "literate/simulations", filename)
    outputdir = joinpath(@__DIR__, "src/examples/simulations")
    Literate.markdown(path, outputdir; codefence="```julia" => "```")
end

for filename in MODELS
    path = joinpath(@__DIR__, "literate/models", filename)
    outputdir = joinpath(@__DIR__, "src/examples/models")
    Literate.markdown(path, outputdir; codefence="```julia" => "```")
end
