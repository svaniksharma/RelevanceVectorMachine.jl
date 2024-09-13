include("../src/RelevanceVectorMachine.jl")
using DataFrames
using StatsModels
using Distributions
using RDatasets
using Test

# Check that we get can get target and predictor variables from formula

function test_formula_parsing()
    df = RDatasets.dataset("datasets", "women")
    nrows = nrow(df)
    ncols = ncol(df)
    Φ = RelevanceVectorMachine.get_Φ(@formula(Height ~ Weight), df)
    t = RelevanceVectorMachine.get_t(@formula(Height ~ Weight), df)
    @test size(Φ, 1) == nrows
    @test size(Φ, 2) == ncols - 1
    @test size(t, 1) == nrows
    @test size(t, 2) == 1
end

# check that output is between 0 and 1 for classification tasks

function test_classification_output()
    df = RDatasets.dataset("datasets", "women")
    rvm = RelevanceVectorMachine.rvm(@formula(Height ~ Weight), df, "classification")
    X = Matrix(select(df, :Weight))
    predictions = RelevanceVectorMachine.predict(rvm, X)
    @test all(0 .≤ predictions .≤ 1)
end

# create a perfectly linear dataset and check that the data is overfit

function test_rvm_regression()
    X = randn(100, 4)
    w = randn(4,)
    y = X * w
    model_matrix = hcat(X, y)
    df = DataFrame(model_matrix, :auto)
    rvm = RelevanceVectorMachine.rvm(@formula(x5 ~ x1 + x2 + x3 + x4), df)
    predictions = RelevanceVectorMachine.predict(rvm, X)
    @test isapprox(y, predictions, rtol=1e-3)
end

# create a perfectly separable dataset and check that the data is overfit

function test_rvm_classification()
    X = randn(100, 1)
    w = rand()
    y = X * w
    y[y .< 0] .= -1
    y[y .≥ 0] .= 1
    model_matrix = hcat(X, y)
    df = DataFrame(model_matrix, :auto)
    rvm = RelevanceVectorMachine.rvm(@formula(x2 ~ x1), df, "classification")
    predictions = RelevanceVectorMachine.predict(rvm, X)
    predictions[predictions .< 0.5] .= -1
    predictions[predictions .≥ 0.5] .= 1
    @test mean(y .== predictions) == 1.0
end

test_formula_parsing()
test_classification_output()
test_rvm_regression()
test_rvm_classification()