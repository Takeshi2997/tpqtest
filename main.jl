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
    data = Vector{Data}(undef, c.NBatch)
    models = Vector{GPmodel}(undef, c.NBatch)
    for i in 1:c.NBatch
        data_x = Vector{State}(undef, c.NData)
        for i in 1:c.NData
            data_x[i] = State(rand([1.0, -1.0], c.NSpin))
        end
        bimu = zeros(Float64, 2 * c.NData)
        biI  = Array(Diagonal(ones(Float64, 2 * c.NData)))
        biψ  = rand(MvNormal(bimu, biI))
        data_y = log.(biψ[1:c.NData] .+ im * biψ[c.NData+1:end])
        data[i] = Data(data_x, data_y)
        models[i]= GPmodel(data_x, data_y, 0.0)
    end

    batch_x = Vector{State}(undef, c.NMC)
    for i in 1:c.NMC
        x = rand(eng, [1.0, -1.0], c.NSpin)
        batch_x[i] = State(x)
    end

    ene = 0.0
    for k in 1:c.iT
        for l in 1:c.iM
            for n in 1:c.NBatch
                models[n] = imaginarytime(models[(k*l+n-2)%c.NBatch+1])
            end
        end
        ene = energy(batch_x, models[1])
        β = c.Δβ * k
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
