include("./setup.jl")
include("./hamiltonian.jl")
using LinearAlgebra

mutable struct GPmodel{T<:Complex}
    ρ::Array{T}
end
 