
using Documenter, RelevanceVectorMachine

makedocs(
    modules = [RelevanceVectorMachine],
    sitename="RelevanceVectorMachine.jl",
)

deploydocs(repo="https://github.com/svaniksharma/RelevanceVectorMachine.jl",)