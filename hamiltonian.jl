using SparseArrays

I = sparse([1.0+0.0im 0.0; 0.0 1.0+0.0im])
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

function set_spins(N, sites, σs)
    list_mats = fill(I, N)
    for (site, σ) in zip(sites, σs)
        list_mats[site] = σ
    end
    return list_mats
end

function hamiltonian_ising(N::T) where {T<:Integer}
    H = ⊗(set_spins(N, [N,1], [S³, S³])...)
    for i in 1:N-1
        H += ⊗(set_spins(N, [i,i+1], [S³,S³])...)
        H += -c.H * ⊗(set_spins(N, [i], [S¹]))
    end
    return H
end

function hamiltonian(N::T) where {T<:Integer}
    hamiltonian_ising(N)
#    hamiltonian_heisenberg(Ω)
#    hamiltonian_XY(Ω)
end
