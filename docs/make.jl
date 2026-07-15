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
            "CPDI" => "tutorials/cpdi.md",
            "tutorials/tlmpm_vortex.md",
            "Implicit methods" => [
                "implicit.md",
                "Jacobian-free" => "tutorials/implicit_jacobian_free.md",
                "Jacobian-based" => "tutorials/implicit_jacobian_based.md",
            ],
            "tutorials/collapse.md",
            "Incompressible fluid flow" => "tutorials/dam_break.md",
            "Marker-and-cell method" => "tutorials/macgrid.md",
            "tutorials/rigid_body_contact.md",
            "tutorials/taylor_impact.md",
            "FEM" => [
                "tutorials/heat.md"
            ],
        ],
        "Manual" => [
            "Overview" => "manual.md"
            "Core workflow" => [
                "mesh.md"
                "generation.md"
                "basis.md"
                "transfer.md"
                "export.md"
            ]
            "Scaling simulations" => [
                "multithreading.md"
                "gpu.md"
                "sparray.md"
            ]
            "Advanced methods" => [
                "implicit_utils.md"
                "fem.md"
                "iga.md"
            ]
        ]
    ],
    doctest = true, # :fix
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/KeitaNakamura/Tesserae.jl.git",
    devbranch = "main",
    push_preview = true,
)
