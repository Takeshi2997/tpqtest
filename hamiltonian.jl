using SparseArrays, CUDA, CUDA.CUSPARSE, LinearAlgebra

S⁰ = sparse([1.0+0.0im 0.0; 0.0 1.0+0.0im])
S¹ = sparse([0.0 1.0+0.0im; 1.0+0.0im 0.0]) / 2.0
S² = sparse([0.0 -1.0im; 1.0im 0.0]) / 2.0
S³ = sparse([1.0+0.0im 0.0; 0.0 -1.0+0.0im]) / 2.0

function ⊗(A,B,C...)
    A = kron(A, B)
    for Ci in C
        A = kron(A, Ci)
    end
    return A
end

function set_spins(N, sites, Ss)
    list_mats = fill(S⁰, N)
    for (site, S) in zip(sites, Ss)
        list_mats[site] = S
    end
    return list_mats
end

function hamiltonian_ising(N::T) where {T<:Integer}
    H = ⊗(set_spins(N, [N,1], [S³, S³])...)
    for i in 1:N-1
        H += ⊗(set_spins(N, [i,i+1], [S³,S³])...)
        H += -c.H * ⊗(set_spins(N, [i], [S¹])...)
    end
    return H ./ c.NSpin
end

function hamiltonian_heisenberg(N::T) where {T<:Integer}
    H  = c.J .* ⊗(set_spins(N, [N,1], [S¹, S¹])...)
    H += c.J .* ⊗(set_spins(N, [N,1], [S², S²])...)
    H += c.J .* ⊗(set_spins(N, [N,1], [S³, S³])...)
    for i in 1:N-1
        H += c.J .* ⊗(set_spins(N, [i,i+1], [S¹, S¹])...)
        H += c.J .* ⊗(set_spins(N, [i,i+1], [S², S²])...)
        H += c.J .* ⊗(set_spins(N, [i,i+1], [S³, S³])...)
    end
    return H ./ c.NSpin
end

function hamiltonian(N::T) where {T<:Integer}
#     hamiltonian_ising(N)
    hamiltonian_heisenberg(N)
#    hamiltonian_XY(N)
end

function setmatrix()
    CUDA.allowscalar(true)
    T = ComplexF64
    h_cpu = hamiltonian(c.NSpin)
    I_cpu =⊗(fill(S⁰, c.NSpin)...)
    h::CuSparseMatrixCSC{T} = CuSparseMatrixCSC(h_cpu)
    I::Hermitian{T, CuArray{T, 2}} = Hermitian(CuArray(I_cpu))
    A::Hermitian{T, CuSparseMatrixCSC{T}} = Hermitian(CuSparseMatrixCSC((c.l * I_cpu - h_cpu)))
    h, I, A
end

const h, I, A = setmatrix()

