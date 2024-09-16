module RelevanceVectorMachine

export rvm, predict, posterior

using StatsModels
using Tables
using LinearAlgebra
using Statistics
using Distributions

struct RVM
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    α::Vector{Float64}
    B::Matrix{Float64}
    is_regression::Bool
end

function rvm(formula::FormulaTerm, data, mode = "regression")
    Φ = get_Φ(formula, data)
    t = get_t(formula, data)
    if mode != "regression" && mode != "classification"
        error("Specify mode as regression or classification")
    end
    sparse_seq_bayes(Φ, t, mode == "regression")
end

function predict(rvm::RVM, X)
    if rvm.is_regression
        X * rvm.μ
    else
        σ.(X * rvm.μ)
    end
end

posterior(rvm::RVM) = MvNormal(rvm.μ, Hermitian(rvm.Σ))

get_Φ(formula, data) = float.(modelmatrix(formula.rhs, data))
get_t(formula, data) = float.(vec(modelmatrix(formula.lhs, data)))
get_N(Φ) = size(Φ, 1)
get_M(Φ) = size(Φ, 2)
σ(y) = 1 / (1 + exp(-y))

function sparse_seq_bayes(Φ::Matrix{Float64}, t::Vector{Float64}, is_regression::Bool)
    N = get_N(Φ)
    B = randn(N, N)
    if is_regression
        β = var(t) * 0.1
        B = β * diagm(ones(N))
    end
    M = get_M(Φ)
    α = fill(Inf, M)
    μ = zeros(M,)
    Σ = zeros(M, M)
    B = compute_B(B, Φ, μ, is_regression)
    t_hat = compute_t_hat(B, Φ, μ, t, is_regression)
    mask = BitArray(fill(false, M))
    mask[1] = true
    S = compute_S(Φ, B, Σ, mask)
    Q = compute_Q(Φ, B, Σ, t_hat, mask)
    q = compute_q(Q, S, α)
    s = compute_s(S, α)
    α[1] = update_α(1, q, s)
    Σ[mask, mask] = compute_Σ(Φ, B, α, mask)
    μ[mask] = compute_μ(Φ, B, Σ, t_hat, mask)
    while !converged(α, q, s)
        for i ∈ 1:M
            if q[i]^2 > s[i] && α[i] < Inf
                α[i] = update_α(i, q, s)
            elseif q[i]^2 > s[i] && α[i] == Inf
                mask[i] = true
                α[i] = update_α(i, q, s)
            elseif q[i]^2 ≤ s[i] && α[i] < Inf
                mask[i] = false
                α[i] = Inf
            end
        end
        if is_regression
            B = update_β(Φ, μ, Σ, α, t, mask)
        end
        B = compute_B(B, Φ, μ, is_regression)
        t_hat = compute_t_hat(B, Φ, μ, t, is_regression)
        S = compute_S(Φ, B, Σ, mask)
        Q = compute_Q(Φ, B, Σ, t_hat, mask)
        q = compute_q(Q, S, α)
        s = compute_s(S, α)
        Σ[mask, mask] = compute_Σ(Φ, B, α, mask)
        μ[mask] = compute_μ(Φ, B, Σ, t_hat, mask)
    end
    RVM(μ, Σ, α, B, is_regression)
end

function compute_B(B::Matrix{Float64}, Φ::Matrix{Float64}, μ::Vector{Float64}, is_regression::Bool)
    if is_regression
        return B
    else
        return diagm(σ.(Φ * μ))
    end
end

function compute_t_hat(B::Matrix{Float64}, Φ::Matrix{Float64}, μ::Vector{Float64}, t::Vector{Float64}, is_regression::Bool)
    if is_regression
        return t
    else
        return Φ * μ + B^(-1) * (t .- σ.(Φ * μ))
    end
end

function compute_S(Φ::Matrix{Float64}, B::Matrix{Float64}, Σ::Matrix{Float64}, mask::BitVector)
    M = get_M(Φ)
    S = zeros(M,)
    Φ_vw = Φ[:, mask]
    Σ_vw = Σ[mask, mask]
    for i ∈ 1:M
        S[i] = transpose(Φ[:, i]) * B * Φ[:, i] - transpose(Φ[:, i]) * B * Φ_vw * Σ_vw * transpose(Φ_vw) * B * Φ[:, i]
    end
    S
end

function compute_Q(Φ::Matrix{Float64}, B::Matrix{Float64}, Σ::Matrix{Float64}, t_hat::Vector{Float64}, mask::BitVector)
    M = get_M(Φ)
    Q = zeros(M,)
    Φ_vw = Φ[:, mask]
    Σ_vw = Σ[mask, mask]
    for i ∈ 1:M
        Q[i] = transpose(Φ[:, i]) * B * t_hat - transpose(Φ[:, i]) * B * Φ_vw * Σ_vw * transpose(Φ_vw) * B * t_hat
    end
    Q
end

function compute_s(S::Vector{Float64}, α::Vector{Float64})
    s = zeros(size(S, 1),)
    for i ∈ eachindex(S)
        if α[i] == Inf
            s[i] = S[i]
        else
            s[i] = α[i] * S[i] / (α[i] - S[i])
        end
    end
    s
end

function compute_q(Q::Vector{Float64}, S::Vector{Float64}, α::Vector{Float64})
    q = zeros(size(Q, 1),)
    for i ∈ eachindex(Q)
        if α[i] == Inf
            q[i] = Q[i]
        else
            q[i] = α[i] * Q[i] / (α[i] - S[i])
        end
    end
    q
end

function update_α(i, q::Vector{Float64}, s::Vector{Float64})
    q[i]^2 / (q[i]^2 - s[i])
end

function compute_Σ(Φ::Matrix{Float64}, B::Matrix{Float64}, α::Vector{Float64}, mask::BitVector)
    Φ_vw = @view Φ[:, mask]
    α_vw = @view α[mask]
    A = diagm(α_vw)
    (transpose(Φ_vw) * B * Φ_vw + A)^(-1)
end

function compute_μ(Φ::Matrix{Float64}, B::Matrix{Float64}, Σ::Matrix{Float64}, t_hat::Vector{Float64}, mask::BitVector)
    Φ_vw = @view Φ[:, mask]
    Σ_vw = @view Σ[mask, mask]
    Σ_vw * transpose(Φ_vw) * B * t_hat
end

function update_β(Φ::Matrix{Float64}, μ::Vector{Float64}, Σ::Matrix{Float64}, α::Vector{Float64}, t_hat::Vector{Float64}, mask::BitVector)
    N = get_N(Φ)
    M = get_M(Φ)
    Σ_vw = @view Σ[mask, mask]
    Φ_vw = @view Φ[:, mask]
    μ_vw = @view μ[mask]
    α_vw = @view α[mask]
    (N - M + sum(α_vw .* diag(Σ_vw))) / (norm(t_hat - Φ_vw * μ_vw)^2) * diagm(ones(N))
end

function converged(α::Vector{Float64}, q::Vector{Float64}, s::Vector{Float64})
    all(log.(abs.(α)) .< 1e-6) && all(q.^2 .> s)
end

end # module RelevanceVectorMachine
