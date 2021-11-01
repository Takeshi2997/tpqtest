include("./setup.jl")
include("./model.jl")
include("./hamiltonian.jl")
using LinearAlgebra, CUDA

function update(model::GPmodel)
    ρ = model.ρ
    ρ1 = A * (A * ρ)'
    ρ  = Hermitian(ρ1 ./ tr(ρ1))
    GPmodel(ρ)
end

function energy(model::GPmodel)
    ρ = model.ρ
    C = cholesky(ρ)
    data_ψ2 = [CUDA.randn(Float64, 2^c.NSpin * 2) for i in 1:c.NData]
    data_ψ = [data_ψ2[i][1:2^c.NSpin] + im * data_ψ2[i][2^c.NSpin+1:end] for i in 1:c.NData]
    data_ψ = [C.L * data_ψ[i] for i in 1:c.NData]
    data_ψ = [data_ψ[i] ./ norm(data_ψ[i])  for i in 1:c.NData]
    sum([real(dot(data_ψ[i], h * data_ψ[i])) for i in 1:c.NData]) / c.NData
end


