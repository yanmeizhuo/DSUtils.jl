using DSUtils
using Documenter

makedocs(;
    modules=[DSUtils],
    authors="Daymond Ling",
    repo="https://github.com/DaymondLing/DSUtils.jl/blob/{commit}{path}#L{line}",
    sitename="DSUtils.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://DaymondLing.github.io/DSUtils.jl/stable",
        assets=String[],
    ),
    pages = Any[
        "Home" => "index.md",
        "Manual" => Any[
            "Data preparation" => "man/dataprep.md",
            "Binning methods" => "man/binning.md",
            "Categorical encoding" => "man/categorical.md",
            "Binary Classifier" => "man/bcdiag.md",
        ],
        "Function Reference" => "Reference.md",
    ],
)

deploydocs(;
   repo="github.com/DaymondLing/DSUtils.jl",
)


#=
makedocs(;
    modules=[DSUtils],
    authors="Daymond Ling",
    repo="https://github.com/DaymondLing/DSUtils.jl/blob/{commit}{path}#L{line}",
    sitename="DSUtils.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://DaymondLing.github.io/DSUtils.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DaymondLing/DSUtils.jl",
)
=#
