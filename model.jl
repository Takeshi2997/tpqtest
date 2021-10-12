include("./setup.jl")
using LinearAlgebra

mutable struct State{T<:Real}
    spin::Vector{T}
end

mutable struct GPmodel{T<:Complex, S<:Real}
    data_x::Vector{State}
    data_y::Vector{T}
    τ::S
    pvec::Vector{T}
    KI::Array{T}
end
function GPmodel(data_x::Vector{State}, data_y::Vector{T}, τ::S) where {T<:Complex, S<:Real}
    KI = Array{T}(undef, c.NData, c.NData)
    makematrix(KI, data_x, τ)
    makeinverse(KI)
    pvec = KI * exp.(data_y)
    GPmodel(data_x, data_y, τ, pvec, KI)
end
function GPmodel(model::GPmodel, τ::S) where {S<:Real}
    data_x, data_y = model.data_x, model.data_y
    GPmodel(data_x, data_y, τ)
end

function kernel(x1::State, x2::State, τ::S) where {S<:Real}
    v = norm(x1.spin - x2.spin)^2
    v /= c.NSpin
    c.B * exp(-v ./ (0.2 + τ))
end

function makematrix(K::Array{T}, data_x::Vector{State}, τ::S) where{T<:Complex, S<:Real}
    for i in 1:length(data_x)
        for j in i:length(data_x)
            K[i, j] = kernel(data_x[i], data_x[j], τ)
            K[j, i] = K[i, j]
        end
    end
end 

function makeinverse(KI::Array{T}) where {T<:Complex}
    # KI[:, :] = inv(KI)
    U, Δ, V = svd(KI)
    invΔ = Diagonal(1.0 ./ Δ .* (Δ .> 1e-6))
    KI[:, :] = V * invΔ * U'
end

function predict(x::State, model::GPmodel) where {S<:Real}
    data_x, τ, pvec, KI = model.data_x, model.τ, model.pvec, model.KI

    # Compute mu and var
    kv = map(x1 -> kernel(x1, x, τ), data_x)
    k0 = kernel(x, x, τ)
    mu = kv' * pvec
    var = k0 - kv' * KI * kv

    # sample from gaussian
    log(sqrt(var) * randn(typeof(mu)) + mu)
end

