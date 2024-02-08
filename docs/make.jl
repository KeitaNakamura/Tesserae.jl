using Documenter
using Sequoia

ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"

# Setup for doctests in docstrings
DocMeta.setdocmeta!(Sequoia, :DocTestSetup, :(using Sequoia); recursive=true)

makedocs(;
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    modules = [Sequoia],
    sitename = "Sequoia.jl",
    pages=[
        "Home" => "index.md",
        "getting_started.md",
    ],
    doctest = true, # :fix
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/KeitaNakamura/Sequoia.jl.git",
    devbranch = "main",
)
