### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ d1d9ae1e-22bb-4aae-af97-4b4c07abc0f0
begin
	import Pkg
	Pkg.activate(Base.current_project())
end

# ╔═╡ 15413012-9e17-45be-aaae-cc867ede63cb
begin
	using RDatasets
	using StatsModels
	using Plots
	using Statistics
end

# ╔═╡ a7b141eb-2085-4efd-9b9e-2e7851e4d8d7
include("../src/RelevanceVectorMachine.jl")

# ╔═╡ 42bdf6a4-73c0-11ef-129e-158f23f29580
md"""
# Boston Dataset Regression Example
"""

# ╔═╡ 5d96b1e1-6c81-4ea5-92a0-e9afe93b1bf3
md"""
## Introduction

The following illustrates how to do univariate regression with the relevance vector machine.
"""

# ╔═╡ 75230d1e-e217-4e55-8678-4cd930d8fdf1
md"""
## Processing Data
"""

# ╔═╡ a92e2baa-4f90-4418-9da6-4d16b773aba9
boston_data = RDatasets.dataset("MASS", "Boston")

# ╔═╡ f793afcf-0873-4bf8-9428-bebee2093fd1
md"""
## Training the Model
"""

# ╔═╡ 56a1ed96-456a-4163-927a-df4262396929
scatter(boston_data[:, :Rm], boston_data[:, :MedV])

# ╔═╡ 2740ddcc-ab93-4eca-856f-044c4ef6fb94
rvm = RelevanceVectorMachine.rvm(@formula(MedV ~ Rm), boston_data)

# ╔═╡ 17fd3b57-5cf0-4fb0-89b5-dc8f719ec906
md"""
## Plotting Predictions
"""

# ╔═╡ 0a0018b3-c361-4eb0-9eb1-71a5e5078e91
begin
	preds = RelevanceVectorMachine.predict(rvm, boston_data)
	p = plot(boston_data[:, :Rm], preds)
	scatter!(p, boston_data[:, :Rm], boston_data[:, :MedV])
end

# ╔═╡ 7b585bc7-2410-401d-8068-9e650a684ecf
md"""
## Examining the Posterior
"""

# ╔═╡ 40aa5ac2-179c-4775-8d77-820fc82e9ed7
post = RelevanceVectorMachine.posterior(rvm)

# ╔═╡ de8a74ab-f3f0-447f-a296-a24ce4562bdf
μ_post = rand(post, 10000)

# ╔═╡ f090679c-6f38-4f88-addf-6e42bb2e0606
pred_samples = boston_data[:, :Rm] * μ_post

# ╔═╡ d13e59f2-49ce-4235-a0e6-3bed30930b21
# This calculates the means from the samples
mean_pred_samples = mean(pred_samples, dims = 2)

# ╔═╡ adc6a865-e753-40a9-8b7a-de39b16f5195
begin
	# This calculates a 95% credible interval
	function get_endpoints(sample_row)
		sort(sample_row)[[26, 975]]
	end
	cred_pred = stack(map(get_endpoints, eachrow(pred_samples)), dims = 1)
end

# ╔═╡ Cell order:
# ╠═42bdf6a4-73c0-11ef-129e-158f23f29580
# ╠═5d96b1e1-6c81-4ea5-92a0-e9afe93b1bf3
# ╠═75230d1e-e217-4e55-8678-4cd930d8fdf1
# ╠═d1d9ae1e-22bb-4aae-af97-4b4c07abc0f0
# ╠═a7b141eb-2085-4efd-9b9e-2e7851e4d8d7
# ╠═15413012-9e17-45be-aaae-cc867ede63cb
# ╠═a92e2baa-4f90-4418-9da6-4d16b773aba9
# ╠═f793afcf-0873-4bf8-9428-bebee2093fd1
# ╠═56a1ed96-456a-4163-927a-df4262396929
# ╠═2740ddcc-ab93-4eca-856f-044c4ef6fb94
# ╠═17fd3b57-5cf0-4fb0-89b5-dc8f719ec906
# ╠═0a0018b3-c361-4eb0-9eb1-71a5e5078e91
# ╠═7b585bc7-2410-401d-8068-9e650a684ecf
# ╠═40aa5ac2-179c-4775-8d77-820fc82e9ed7
# ╠═de8a74ab-f3f0-447f-a296-a24ce4562bdf
# ╠═f090679c-6f38-4f88-addf-6e42bb2e0606
# ╠═d13e59f2-49ce-4235-a0e6-3bed30930b21
# ╠═adc6a865-e753-40a9-8b7a-de39b16f5195
