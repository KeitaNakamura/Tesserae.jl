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
        "Poingr.DofHelpers" => "DofHelpers.md",
        "Poingr.Arrays" => "Arrays.md",
        "Poingr.Collections" => "Collections.md",
        "Poingr.Grids" => "Grids.md",
        "Poingr.ShapeFunctions" => "ShapeFunctions.md",
        "Poingr.MPSpaces" => "MPSpaces.md",
        "Poingr.ContactMechanics" => "ContactMechanics.md",
        "Poingr.Loggers" => "Loggers.md",
        "Poingr.VTKOutputs" => "VTKOutputs.md",
    ],
    doctest = true, # :fix
)

deploydocs(
    repo = "github.com/KeitaNakamura/Poingr.jl.git",
)
