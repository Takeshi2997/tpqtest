include("./setup.jl")
using LinearAlgebra

mutable struct Parameters{S<:Real}
    W::Array{S, 2}
    b::Vector{S}
end

mutable struct GPmodel{T<:Complex, S<:Real}
    data_x::Vector{Vector{S}}
    data_ψ::Vector{T}
    params::Parameters
    pvec::Vector{T}
    KI::Array{T}
end
function GPmodel(data_x::Vector{Vector{S}}, data_ψ::Vector{T}, params::Parameters) where {T<:Complex, S<:Real}
    KI = Array{T}(undef, c.NData, c.NData)
    makematrix(KI, params, data_x)
    makeinverse(KI)
    pvec = KI * data_ψ
    GPmodel(data_x, data_ψ, params, pvec, KI)
end
function GPmodel(model::GPmodel, params::Parameters)
    data_x, data_ψ = model.data_x, model.data_ψ
    GPmodel(data_x, data_ψ, params)
end

function kernel(params::Parameter, x1::Vector{S}, x2::Vector{S})
    W, b = params.W, params.b
    x = (x1 - x2) ./ c.NData
    exp(-dot(x, W*x + b))
end

function makematrix(K::Array{T}, params::Parameters, data_x::Vector{Vector{S}}) where {T<:Complex, S<:Real}
    for i in 1:length(data_x)
        for j in i:length(data_x)
            K[i, j] = kernel(params, data_x[i], data_x[j])
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

function predict(x::Vector{T}, model::GPmodel) where {T<:Real}
    data_x, params, pvec, KI = model.data_x, model.params, model.pvec, model.KI

    # Compute mu var
    kv = map(x1 -> kernel(params, x, x1), data_x)
    k0 = kernel(params, x, x)
    mu = kv' * pvec
    var = k0 - kv' * KI * kv

    # sample from gaussian
    log(sqrt(var) * randn(typeof(mu)) + mu)
end
