using Documenter
using Marble

ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"

# Setup for doctests in docstrings
DocMeta.setdocmeta!(Marble, :DocTestSetup, :(using Marble); recursive=true)

# generate documentation by Literate.jl
include("literate.jl")

makedocs(;
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Marble],
    sitename = "Marble.jl",
    pages=[
        "Home" => "index.md",
        "getting_started.md",
        "Manual" => ["manual/grid.md",
                     "manual/particles.md",
                     "manual/interpolations.md",
                     "manual/transfers.md",],
        "Examples" => [
            "Simulations" => ["examples/simulations/sand_column_collapse.md",
                              "examples/simulations/dam_break.md",
                              "examples/simulations/axial_vibration_of_bar.md",
                              "examples/simulations/contacting_grains.md"],
            "Material Models" => ["examples/models/LinearElastic.md",
                                  "examples/models/DruckerPrager.md"],
        ],
    ],
    doctest = true, # :fix
)

deploydocs(
    repo = "github.com/KeitaNakamura/Marble.jl.git",
    devbranch = "main",
    push_preview = true,
)
