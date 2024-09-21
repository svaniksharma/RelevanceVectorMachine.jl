### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 925038a5-8a4d-41bf-982c-e266f3c57378
begin
	import Pkg
	Pkg.activate(Base.current_project())
end

# ╔═╡ c83ce400-dc44-4fe5-8ee5-3941940711f1
begin
	using RDatasets
	using StatsModels
	using Statistics
end

# ╔═╡ b3644946-92cb-4054-942d-b8caf1f08358
include("../src/RelevanceVectorMachine.jl")

# ╔═╡ e2d43a3c-72f5-11ef-3fc0-991cbff513be
md"""
## Iris Dataset
"""

# ╔═╡ 87d0e2e9-b989-4e97-b86b-ab8c90fbc37c
md"""
## Introduction
In this example, we train an RVM on the Iris dataset to classify flower species. Since the relevance vector machine implemented is for binary classification, we use it on only the versicolor and virginica species.
"""

# ╔═╡ b25e56ba-8215-4ae1-b5c1-ad5d984a4674
md"""
## Processing Data
"""

# ╔═╡ ae02f097-de63-40e4-8083-89b65ccb606b
begin
	function load_dataset_with_two_categories()
		iris_dataset = RDatasets.dataset("datasets", "iris")
		iris_dataset = filter(row -> row.Species != "setosa", iris_dataset)
		iris_dataset
	end
	iris_dataset = load_dataset_with_two_categories()
end

# ╔═╡ 029dd796-0495-4193-91fc-5cbd26f26a99
md"""
## Training RVM
"""

# ╔═╡ e173bef2-b417-44d6-85b5-bd491a6fc78e
begin
	rvm_formula = @formula(Species ~ SepalLength + SepalWidth + PetalLength + PetalWidth)
	rvm = RelevanceVectorMachine.rvm(rvm_formula, iris_dataset, "classification")
end

# ╔═╡ 2bd11b12-c00b-4d6f-8c43-b4befb910294
md"""
## Making Predictions
"""

# ╔═╡ 8807f224-5413-4850-b499-aeb56d5dfa0a
predictions = RelevanceVectorMachine.predict(rvm, iris_dataset)

# ╔═╡ 374ff4bc-7c6e-4e22-b334-546543ea7776
md"""
`StatsModels` [uses the first level](https://juliastats.org/StatsModels.jl/stable/contrasts/#StatsModels.DummyCoding) in the Iris dataframe ("versicolor") as the "base" level, meaning that 0 corresponds to "versicolor" and 1 corresponds to "virginica". 
"""

# ╔═╡ fb1318ed-3684-46b5-99b5-28987f678c67
md"""
## Examining the Posterior Distribution
"""

# ╔═╡ 1057c497-191b-4eda-9510-2b17eb4a3663
md"""
We can examine the posterior distribution of the weight vector $w$ produced after training. This distribution will be multivariate normal. Below, we sample $1000$ values of the weight vector from the posterior distribution, then use each of these weight vectors to compute the probability that the given datapoint is "versicolor" or "virginia". Then, we use these predictions with the weight samples to construct a 95% credible interval, and plot the curve.
"""

# ╔═╡ d9fb2c2d-8c28-40d1-a556-6e95d0db420e
post = RelevanceVectorMachine.posterior(rvm)

# ╔═╡ cb08b0ba-be3d-4c42-9e7c-2771b7ade377
post_samples = rand(post, 1000)

# ╔═╡ 836cfee8-6bfd-4d68-9f0d-3415726f4d5a
σ(y) = 1 / (1 + exp(-y)) # sigmoid function

# ╔═╡ 91832a51-1040-4340-86a9-140eb076a318
begin
	X = Matrix(select(iris_dataset, Not(:Species)))
	pred_samples = σ.(X * post_samples)
end

# ╔═╡ 0f779131-9ae6-4e8f-8bb4-162276bbeb22
mean_pred_prob = mean(pred_samples, dims = 2)

# ╔═╡ 223fef89-9735-4123-ad7d-f147875fe481
function get_endpoints(sample_row)
	sort(sample_row)[[26, 975]]
end

# ╔═╡ 9f53e207-bec2-4eba-bd6c-fe08c43b6a8d
cred_pred_prob = stack(map(get_endpoints, eachrow(pred_samples)), dims = 1)

# ╔═╡ 22360b5e-a65c-43a9-9717-7cdc61a46e00
md"""
If this were in two or three dimensions, we could plot it, but in this case we are dealing with 4D data points (`SepalLength`, `SepalWidth`, `PetalLength`, `PetalWidth`).
"""

# ╔═╡ Cell order:
# ╠═e2d43a3c-72f5-11ef-3fc0-991cbff513be
# ╠═87d0e2e9-b989-4e97-b86b-ab8c90fbc37c
# ╠═b25e56ba-8215-4ae1-b5c1-ad5d984a4674
# ╠═925038a5-8a4d-41bf-982c-e266f3c57378
# ╠═b3644946-92cb-4054-942d-b8caf1f08358
# ╠═c83ce400-dc44-4fe5-8ee5-3941940711f1
# ╠═ae02f097-de63-40e4-8083-89b65ccb606b
# ╠═029dd796-0495-4193-91fc-5cbd26f26a99
# ╠═e173bef2-b417-44d6-85b5-bd491a6fc78e
# ╠═2bd11b12-c00b-4d6f-8c43-b4befb910294
# ╠═8807f224-5413-4850-b499-aeb56d5dfa0a
# ╠═374ff4bc-7c6e-4e22-b334-546543ea7776
# ╠═fb1318ed-3684-46b5-99b5-28987f678c67
# ╠═1057c497-191b-4eda-9510-2b17eb4a3663
# ╠═d9fb2c2d-8c28-40d1-a556-6e95d0db420e
# ╠═cb08b0ba-be3d-4c42-9e7c-2771b7ade377
# ╠═836cfee8-6bfd-4d68-9f0d-3415726f4d5a
# ╠═91832a51-1040-4340-86a9-140eb076a318
# ╠═0f779131-9ae6-4e8f-8bb4-162276bbeb22
# ╠═223fef89-9735-4123-ad7d-f147875fe481
# ╠═9f53e207-bec2-4eba-bd6c-fe08c43b6a8d
# ╠═22360b5e-a65c-43a9-9717-7cdc61a46e00
