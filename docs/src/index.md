# RelevanceVectorMachine.jl Documentation

```@contents
```

```@meta
CurrentModule = RelevanceVectorMachine
```

## Quickstart

See the [examples](https://github.com/svaniksharma/RelevanceVectorMachine.jl/tree/master/examples) folder.
If you clone the repository, then you should be able to run the examples using [Pluto.jl](https://plutojl.org/) notebooks.

## Functions

```@docs
RVM
```

```@docs
rvm(formula::FormulaTerm, data, mode = "regression", max_iters = 100)
```

```@docs
predict(rvm::RVM, X)
```

```@docs
posterior(rvm::RVM)
```

## References

Tipping, M.E. &amp; Faul, A.C.. (2003). Fast Marginal Likelihood Maximisation for Sparse Bayesian Models. *Proceedings of the Ninth International Workshop on Artificial Intelligence and Statistics*, in *Proceedings of Machine Learning Research* R4:276-283 Available from [https://proceedings.mlr.press/r4/tipping03a.html](https://proceedings.mlr.press/r4/tipping03a.html). Reissued by PMLR on 01 April 2021.
