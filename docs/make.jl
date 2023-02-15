using Documenter
using Marble

# Setup for doctests in docstrings
DocMeta.setdocmeta!(Marble, :DocTestSetup, :(using Marble); recursive=true)

makedocs(;
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Marble],
    sitename = "Marble.jl",
    pages=[
        "Home" => "index.md",
        "Grid" => "grid.md",
        "Interpolations" => "interpolations.md",
    ],
    doctest = true, # :fix
)

deploydocs(
    repo = "github.com/KeitaNakamura/Marble.jl.git",
    devbranch = "main",
)
