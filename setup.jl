struct GP_Data{T<:Real, S<:Integer}
    NSpin::S
    T::S
    H::T
    t::T
    J::T
    l::T
end
function GP_Data()
    NSpin = 16
    T = 100
    H = 2.0
    t = 1.0
    J = 1.0
    l = 0.8
    GP_Data(NSpin, T, H, t, J, l)
end
c = GP_Data()
