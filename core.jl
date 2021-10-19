include("./setup.jl")
include("./model.jl")
include("./hamiltonian.jl")
using Base.Threads, LinearAlgebra, Random, Folds

function imaginarytime(model::GPmodel)
    data_x, data_y, τ = model.data_x, model.data_y, model.τ
    @threads for n in 1:c.NData
        x = data_x[n]
        h = localenergy(x, model)
        data_y[n] += log(1.0 - c.NSpin * c.Δτ / 2.0 * h)
    end
    data_y ./= norm(data_y)
    model_loc = GPmodel(data_x, data_y, τ)
    τ0 = parameterfitting(model_loc)
    GPmodel(model_loc, τ0)
end

function parameterfitting(model::GPmodel) where {S<:Real}
    τ0 = nls(diffloglikelifood, model, ini=0.2)
    return τ0
end

function tryflip(x::State, model::GPmodel, eng::MersenneTwister)
    pos = rand(eng, collect(1:c.NSpin))
    y = predict(x, model)
    xflip_spin = copy(x.spin)
    xflip_spin[pos] *= -1
    xflip = State(xflip_spin)
    y_new = predict(xflip, model)
    prob = exp(2 * real(y_new - y))
    x.spin[pos] *= ifelse(rand(eng) < prob, -1, 1)
    State(x.spin)
end

function localenergy(x::State, model::GPmodel)
    y = predict(x, model)
    eloc = 0.0im
    @simd for i in 1:c.NSpin
        e = hamiltonian(i, x, y, model)
        eloc += e / c.NSpin
    end
    eloc
end

function energy(x_mc::Vector{State}, model::GPmodel)
    @threads for i in 1:c.NMC
        @simd for j in 1:c.MCSkip
            eng = EngArray[threadid()]
            x_mc[i] = tryflip(x_mc[i], model, eng)
        end
    end
    ene = Folds.sum(localenergy(x, model) for x in x_mc)
    real(ene / c.NMC)
end

function nls(func, params...; ini = [0.0])
    if typeof(ini) <: Number
        r = nlsolve((vout,vin)->vout[1]=func(vin[1],params...), [ini])
        v = r.zero[1]
    else
        r = nlsolve((vout,vin)->vout .= func(vin,params...), ini)
        v = r.zero
    end
    return v
end
