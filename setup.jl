struct GP_Data{T<:Real, S<:Integer}
    NSpin::S
    NData::S
    NMC::S
    MCSkip::S
    T::S
    H::T
    t::T
    J::T
    Δτ::T
    η::T
end
function GP_Data()
    NSpin = 80
    NData = 64
    NMC = 1024
    MCSkip = 16
    T = 50
    H = 2.0
    t = 1.0
    J = 1.0
    Δτ = 1e-4
    η = 1e-4
    GP_Data(NSpin, NData, NMC, MCSkip, T, H, t, J, Δτ, η)
end
c = GP_Data()
const I = Array(Diagonal(ones(Float64, c.NData)))