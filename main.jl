include("./setup.jl")
include("./model.jl")
include("./core.jl")
using LinearAlgebra, Random, Distributions

EngArray = Vector{MersenneTwister}(undef, nthreads())
function main(filename::String)
    for i in 1:nthreads()
        EngArray[i] = MersenneTwister(i)
    end
    eng = EngArray[1]
    data_x = Vector{State}(undef, c.NData)
    for i in 1:c.NData
        data_x[i] = State(rand([1.0, -1.0], c.NSpin))
    end
    bimu = zeros(Float64, 2 * c.NData)
    biI  = Array(Diagonal(ones(Float64, 2 * c.NData)))
    biψ  = rand(MvNormal(bimu, biI))
    data_y = log.(biψ[1:c.NData] .+ im * biψ[c.NData+1:end])
    model = GPmodel(data_x, data_y, -1.0)

    batch_x = Vector{State}(undef, c.NMC)
    for i in 1:c.NMC
        x = rand(eng, [1.0, -1.0], c.NSpin)
        batch_x[i] = State(x)
    end

    ene = 0.0
    for k in 1:c.iT
        model = imaginarytime(model)
        ene = energy(batch_x, model)
        β = 2 * k / c.NSpin / (c.l - ene)
        open("./data/" * filename, "a") do io
            write(io, string(k))
            write(io, "\t")
            write(io, string(ene))
            write(io, "\t")
            write(io, string(β))
            write(io, "\t")
            write(io, string(exp(model.τ)))
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
