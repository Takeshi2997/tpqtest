include("./setup.jl")
include("./model.jl")
include("./core.jl")
using LinearAlgebra, Random, CUDA

EngArray = Vector{MersenneTwister}(undef, nthreads())
function main(filename::String)
    model = GPmodel(I)

    for k in 1:c.T
        model = update(model)
        ϵ = energy(model)
        β = 2 * k / c.NSpin / (c.l - ϵ)
        open("./data/" * filename, "a") do io
            write(io, string(k))
            write(io, "\t")
            write(io, string(ϵ))
            write(io, "\t")
            write(io, string(β))
            write(io, "\n")
        end
    end
end

dirname = "./data"
rm(dirname, force=true, recursive=true)
mkdir("./data")
filename  = "physicalvalue.txt"
touch("./data/" * filename)
main(filename)
