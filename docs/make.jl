using DSUtils
using Documenter

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
