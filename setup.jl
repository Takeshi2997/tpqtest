struct GP_Data{T<:Real, S<:Integer}
    NSpin::S
    NData::S
    NMC::S
    MCSkip::S
    iT::S
    H::T
    t::T
    J::T
    iM::S
    Δτ::T
    Δβ::T
    B::T
    λ::T
end
function GP_Data()
    NSpin = 80
    NData = 64
    NMC = 1024
    MCSkip = 16
    iT = 100
    H = 2.0
    t = 1.0
    J = 1.0
    iM = 100
    Δτ = 0.001
    Δβ = Δτ * iM
    B = 1.0
    λ = 0.2
    GP_Data(NSpin, NData, NMC, MCSkip, iT, H, t, J, iM, Δτ, Δβ, B, λ)
end
c = GP_Data()
