struct GP_Data{T<:Real, S<:Integer}
    NSpin::S
    NData::S
    NMC::S
    MCSkip::S
    NUM::S
    H::T
    t::T
    J::T
    l::T
end
function GP_Data()
    NSpin = 16
    NData = 64
    NMC = 1024
    MCSkip = 16
    NUM = 100
    H = 2.0
    t = 1.0
    J = 1.0
    l = 0.8
    GP_Data(NSpin, NData, NMC, MCSkip, NUM, H, t, J, l)
end
c = GP_Data()
