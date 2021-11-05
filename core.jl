include("./setup.jl")
include("./model.jl")
include("./hamiltonian.jl")
using Base.Threads, LinearAlgebra, Random, Folds

function imaginarytime(model::GPmodel, params::Parameters)
    data_x, data_ψ = model.data_x, model.data_ψ
    @threads for i in 1:c.NData
        x = data_x[i]
        ψ = data_ψ[i]
        epsi = localenergy_psi(x, log(ψ), model)
        data_ψ[i] = ψ - c.Δτ * epsi
    end
    data_ψ ./= norm(data_ψ)
    GPmodel(data_x, data_ψ, params)
end

function localenergy_psi(x::Vector{S}, y::T, model::GPmodel) where {T<:Complex, S<:Real}
    epsi = 0.0im
    @simd for i in 1:c.NSpin
        ep = hamiltonian(i, x, y, model)
        epsi += ep
    end
    epsi * exp(y)
end

function tryflip(x::Vector{T}, model::GPmodel, eng::MersenneTwister) where {T<:Real}
    pos = rand(eng, collect(1:c.NSpin))
    y = predict(x, model)
    xflip = copy(x)
    xflip[pos] *= -1
    y_new = predict(xflip, model)
    prob = exp(2 * real(y_new - y))
    x[pos] *= ifelse(rand(eng) < prob, -1, 1)
    x
end

function localenergy(x::Vector{T}, model::GPmodel) where {T<:Real}
    y = predict(x, model)
    eloc = 0.0im
    @simd for i in 1:c.NSpin
        e = hamiltonian(i, x, y, model)
        o = paramvector(i, x, y, model)
        eloc += e
    end
    eloc
end

function energy(x_mc::Vector{Vector{T}}, model::GPmodel) where {T<:Real}
    @threads for i in 1:c.NMC
        for j in 1:c.MCSkip
            eng = EngArray[threadid()]
            x_mc[i] = tryflip(x_mc[i], model, eng)
        end
    end
    ene = Folds.sum(physicalval(x, model) for x in x_mc)
    real(ene / c.NMC)
end

function physicalval(x::Vector{T}, model::GPmodel) where {T<:Real}
    y = predict(x, model)
    eloc = 0.0im
    @simd for i in 1:c.NSpin
        e = hamiltonian(i, x, y, model)
        o = paramvector(i, x, y, model)
        eloc += e
    end
    oloc = zeros(typeof(y), c.NData^2 + c.NData)
    k = 1
    @simd for i in 1:c.NData
        for j in 1:c.NData
            o = paramvector(i, j, x, y, model)
            oloc[k] += o
            k += 1
        end
    end
    @simd for i in 1:c.NData
        o = paramvector(i, x, y, model)
        oloc[k] += o
        k += 1
    end
    eoloc = e * oloc
    ooloc = oloc * oloc'
    [oloc, eoloc, ooloc]
end

function paramsupdate(params::Parameters, e::T, x_mc::Vector{Vector{T}}, model::GPmodel) where {T<:Real}
    @threads for i in 1:c.NMC
        for j in 1:c.MCSkip
            eng = EngArray[threadid()]
            x_mc[i] = tryflip(x_mc[i], model, eng)
        end
    end
    vals = Folds.sum(physicalval(x, model) for x in x_mc)
    o  = real(vals[1]) ./ c.NMC
    eo = real(vals[2]) ./ c.NMC
    oo = real(vals[3]) ./ c.NMC

    W, b = params.W, params.b
    R = eo - e * o
    S = oo - o * o'
    pramsvector = (S + 1e-3 * I)\R
    ΔWvector = paramsvector[1:c.NData^2]
    ΔW = reshape(ΔWvector, c.NData, c.NData) 
    Δb = paramsvector[c.NData^2:end]
    Parameters(W+c.η*ΔW, b+c.η*Δb)
end