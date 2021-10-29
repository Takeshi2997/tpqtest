include("./setup.jl")
include("./model.jl")
include("./hamiltonian.jl")
using LinearAlgebra, CUDA

function update(model::GPmodel)
    ρ = model.ρ
    ρ = (c.l * I - h) * ρ
    GPmodel(ρ)
end

function energy(model::GPmodel)
    ρ = model.ρ
    C = cholesky(ρ)
    ψ = randn(ComplexF64, 2^c.NSpin)
    ψ = C.L * ψ
    dot(ψ, H * ψ)
end


