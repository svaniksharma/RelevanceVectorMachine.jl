include("../src/RelevanceVectorMachine.jl")
using DataFrames
using StatsModels
using Distributions
using Test

function test_rvm_regression(N)
    w = 0.5
    d = Normal(0, 1)
    ϵ = rand(d, N)
    x = randn(N,)
    y = w * x + ϵ
    df = DataFrame(x = x, y = y)
    rvm_mod = RelevanceVectorMachine.rvm(@formula(y ~ 0 + x), df)
    return isapprox(rvm_mod.μ, [w], rtol = 1)
end

function test_rvm_regression2(N)
    w = [0.3, 0.5]
    d = Normal(0, 1)
    ϵ = rand(d, N)
    x = randn(N,)
    y = w[1] .+ w[2] .* x.^2 .+ ϵ
    df = DataFrame(x = x, y = y)
    rvm_mod = RelevanceVectorMachine.rvm(@formula(y ~ 1 + x^2), df)
    return isapprox(rvm_mod.μ, w, rtol = 1)
end

function test_rvm_classification(N)
    x1 = randn(N,)
    x2 = randn(N,)
    y = Float64.(x1 .- x2 .> 0)
    df = DataFrame(x1 = x1, x2 = x2, y = y)
    rvm_mod = RelevanceVectorMachine.rvm(@formula(y ~ x1 + x2), df, "classification")
    return isapprox(rvm_mod.μ, [1.0, -1.0], rtol = 1)
end

@test test_rvm_regression(1000)
@test test_rvm_regression2(1000)
@test test_rvm_classification(1000)