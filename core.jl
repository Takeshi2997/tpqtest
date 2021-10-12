include("./setup.jl")
include("./model.jl")
include("./hamiltonian.jl")
using Base.Threads, LinearAlgebra, Random, Folds, FastGaussQuadrature

const tset, wset = gausslegendre(11)

function imaginarytime(model::GPmodel)
    data_x, data_y, τ = model.data_x, model.data_y, model.τ
    @threads for n in 1:c.NData
        x = data_x[n]
        y = data_y[n]
        epsi = [localenergy_func(t, x, τ, model) for t in tset]
        data_y[n] = log(exp(y) - c.Δτ / 2.0 * dot(wset, epsi))
    end
    data_y ./= norm(data_y)
    GPmodel(data_x, data_y, τ + c.Δτ)
end

function localenergy_func(t::S, x::State, τ::S, model::GPmodel)  where {T<:Complex, S<:Real}
    τ_loc = τ + c.Δτ / 2.0 * (t + 1.0)
    model_loc = GPmodel(model, τ_loc)
    epsi = 0.0im
    @simd for i in 1:c.NSpin
        ep = hamiltonian_psi(i, x, model_loc) / 2.0
        epsi += ep
    end
    epsi
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

function physicalvals(x::State, model::GPmodel)
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
    ene = Folds.sum(physicalvals(x, model) for x in x_mc)
    real(ene / c.NMC)
end
