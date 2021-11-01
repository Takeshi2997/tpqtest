
include("./setup.jl")
include("./hamiltonian.jl")
using LinearAlgebra, CUDA

mutable struct GPmodel{T<:Complex}
    ρ::Hermitian{T, CuArray{T, 2}}
end
 
