using Documenter
using Tesserae

ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"

# Setup for doctests in docstrings
DocMeta.setdocmeta!(Tesserae, :DocTestSetup, :(using Tesserae); recursive=true)

# generate documentation by Literate.jl
include("literate.jl")

makedocs(;
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    modules = [Tesserae],
    sitename = "Tesserae.jl",
    pages=[
        "Home" => "index.md",
        "getting_started.md",
        "Examples" => [
            "examples/collision.md",
            "Implicit methods" => [
                "examples/implicit_jacobian_free.md",
                "examples/implicit_jacobian_based.md",
            ],
            "examples/dam_break.md",
            "examples/tlmpm_vortex.md",
            "examples/rigid_body_contact.md",
        ],
    ],
    doctest = true, # :fix
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/KeitaNakamura/Tesserae.jl.git",
    devbranch = "main",
)
