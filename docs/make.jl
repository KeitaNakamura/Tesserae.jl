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
        "Tutorials" => [
            "tutorials/getting_started.md",
            "tutorials/collision.md",
            "Implicit methods" => [
                "implicit.md",
                "Jacobian-free" => "tutorials/implicit_jacobian_free.md",
                "Jacobian-based" => "tutorials/implicit_jacobian_based.md",
            ],
            "Incompressible fluid flow" => "tutorials/dam_break.md",
            "CPDI" => "tutorials/cpdi.md",
            "tutorials/tlmpm_vortex.md",
            "tutorials/rigid_body_contact.md",
        ],
        "Manual" => [
            "mesh.md"
            "generation.md"
            "interpolation.md"
            "transfer.md"
            "implicit_utils.md"
            "export.md"
        ]
    ],
    doctest = true, # :fix
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/KeitaNakamura/Tesserae.jl.git",
    devbranch = "main",
)
