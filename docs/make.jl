using Documenter
using Marble

# Setup for doctests in docstrings
DocMeta.setdocmeta!(Marble, :DocTestSetup, recursive = true,
    quote
        using Marble
        using Random
        Random.seed!(1234)
    end
)

makedocs(;
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Marble],
    sitename = "Marble.jl",
    pages=[
        "Home" => "index.md",
        "Grid" => "grid.md",
        "Interpolations" => "interpolations.md",
        "Contact mechanics" => "contact_mechanics.md",
        "VTK outputs" => "VTK_outputs.md",
    ],
    doctest = true, # :fix
)

deploydocs(
    repo = "github.com/KeitaNakamura/Marble.jl.git",
    devbranch = "main",
)
