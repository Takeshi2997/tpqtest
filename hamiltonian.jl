using LinearAlgebra

function hamiltonian(i::Integer, x::Vector{S}, y::T, model::GPmodel) where {T<:Complex, S<:Real}
#     hamiltonian_heisenberg(i, x, y, model)
    hamiltonian_ising(i, x, y, model)
#     hamiltonian_XY(i, x, y, model)
end

function hamiltonian_heisenberg(i::Integer, x::Vector{S}, y::T, model::GPmodel) where {T<:Complex, S<:Real}
    out = 0.0 + 0.0im
    out += x[i] * x[i%c.NSpin+1]
    if real(out) < 0.0
        xflip = copy(x)
        xflip[i] *= -1
        xflip[i%c.NSpin+1] *= -1
        yflip = predict(xflip, model)
        out += 2.0 * exp(yflip - y)
    end
    c.J * out / 4.0 / c.NSpin
end

function hamiltonian_ising(i::Integer, x::Vector{S}, y::T, model::GPmodel) where {T<:Complex, S<:Real}
    xflip = copy(x)
    xflip[i] *= -1
    yflip = predict(xflip, model)
    (-x.spin[i] * x.spin[i%c.NSpin+1] / 4.0 - c.H * exp(yflip - y) / 2.0) / c.NSpin
end

function hamiltonian_XY(i::Integer, x::Vector{S}, y::T, model::GPmodel) where {T<:Complex, S<:Real}
    out = 0.0 + 0.0im
    p = x[i] * x[i%c.NSpin+1]
    if p < 0.0
        xflip = copy(x)
        xflip[i] *= -1
        xflip[i%c.NSpin+1] *= -1
        yflip = predict(xflip, model)
        out += exp(yflip - y)
    end
    c.t * out / 2.0 / c.NSpin
end

function paramvector(i::Integer, j::Integer, x::Vector{S}, y::T, model::GPmodel) where {T<:Complex, S<:Real}
    out = 0.0 + 0.0im
    xflip = copy(x)
    xflip[i] *= -1
    xflip[j] *= -1
    yflip = predict(xflip, model) 
    out += -x[i] * x[j] * exp(yflip - y)
    4.0 * out / c.NData^2
end

function paramvector(i::Integer, x::Vector{S}, y::T, model::GPmodel) where {T<:Complex, S<:Real}
    out = 0.0 + 0.0im
    xflip = copy(x)
    xflip[i] *= -1
    yflip = predict(xflip, model) 
    out += -x[i] * exp(yflip - y)
    2.0 * out / c.NData
end