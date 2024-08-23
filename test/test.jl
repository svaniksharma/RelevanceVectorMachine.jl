include("../src/RelevanceVectorMachine.jl")
using DataFrames
using StatsModels
using Distributions
using Test

function test_rvm(N)
    w = 0.5
    d = Normal(0, 1)
    ϵ = rand(d, N)
    x = randn(N,)
    y = w * x + ϵ
    df = DataFrame(x = x, y = y)
    rvm_mod = RelevanceVectorMachine.rvm(@formula(y ~ 0 + x), df)
    return isapprox(rvm_mod.μ, [w], rtol = 1)
end

function test_rvm2(N)
    w = [0.3, 0.5]
    d = Normal(0, 1)
    ϵ = rand(d, N)
    x = randn(N,)
    y = w[1] .+ w[2] .* x.^2 .+ ϵ
    df = DataFrame(x = x, y = y)
    rvm_mod = RelevanceVectorMachine.rvm(@formula(y ~ 1 + x^2), df)
    return isapprox(rvm_mod.μ, w, rtol = 1)
end

@test test_rvm(1000)
@test test_rvm2(1000)