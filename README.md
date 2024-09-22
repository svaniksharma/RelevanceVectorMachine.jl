# RelevanceVectorMachine.jl

![build](https://github.com/svaniksharma/RelevanceVectorMachine.jl/actions/workflows/CI.yml/badge.svg)
![documentation](https://github.com/svaniksharma/RelevanceVectorMachine.jl/actions/workflows/Documentation.yml/badge.svg)
![Julia](https://img.shields.io/badge/-Julia-9558B2?style=for-the-badge&logo=julia&logoColor=white)

A [relevance vector machine](http://proceedings.mlr.press/r4/tipping03a/tipping03a.pdf) implementation written in Julia.

## Installation

Use Julia's [Pkg](https://docs.julialang.org/en/v1/stdlib/Pkg/) module to install it:

```julia
import Pkg; Pkg.add("RelevanceVectorMachine.jl")
```

## Quick Start

```julia
import RelevanceVectorMachine
using RDatasets
using StatsModels

boston_data = RDatasets.dataset("MASS", "Boston")

rvm = RelevanceVectorMachine.rvm(@formula(MedV ~ Rm), boston_data)

# Predict on new data
preds = RelevanceVectorMachine.predict(rvm, new_boston_data)
```

See the [examples](https://github.com/svaniksharma/RelevanceVectorMachine.jl/tree/master/examples) folder for more.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)