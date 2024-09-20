# This is used by oracle_test.R

using RelevanceVectorMachine, DataFrames, StatsModels

x = collect(range(-1, 1, 100))
y1 = -23.4 * x
y1 = -23.4 * x
y2 = @. 1 + 0.2786 * x + 9.874 * x^2
y3 = @. sin(x / 5) + x/3
synthetic_df = DataFrame(
    x = x,
    y1 = y1,
    y2 = y2,
    y3 = y3
)

synthetic_dict = Dict()
synthetic_dict["y1"] = @formula(y1 ~ x)
synthetic_dict["y2"] = @formula(y2 ~ 1 + x + x^2)
synthetic_dict["y3"] = @formula(y3 ~ sin(x / 5) + x/3)

function fit_synthetic_func(col_name)
    rvm = RelevanceVectorMachine.rvm(synthetic_dict[col_name], synthetic_df)
    predict(rvm, synthetic_df)
end