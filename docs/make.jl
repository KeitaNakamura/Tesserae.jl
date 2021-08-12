using Documenter
using Poingr

# Setup for doctests in docstrings
DocMeta.setdocmeta!(Poingr, :DocTestSetup, recursive = true,
    quote
        using Poingr
    end
)

makedocs(;
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Poingr],
    sitename = "Poingr.jl",
    pages=[
        "Home" => "index.md",
        "Grid" => "grid.md",
        "Shape functions" => "shape_functions.md",
        "ContactMechanics" => "contact_mechanics.md",
        "Loggers" => "logger.md",
        "VTK outputs" => "VTK_outputs.md",
    ],
    doctest = true, # :fix
)

deploydocs(
    repo = "github.com/KeitaNakamura/Poingr.jl.git",
    devbranch = "main",
)
