struct GP_Data{T<:Real, S<:Integer}
    NSpin::S
    NData::S
    T::S
    H::T
    t::T
    J::T
    l::T
end
function GP_Data()
    NSpin = 8
    NData = 256
    T = 100
    H = 2.0
    t = 1.0
    J = 1.0
    l = 5.0
    GP_Data(NSpin, NData, T, H, t, J, l)
end
c = GP_Data()
