include("./setup.jl")
include("./model.jl")
include("./core.jl")
using LinearAlgebra, Random, Distributions
using Base.Threads


EngArray = Vector{MersenneTwister}(undef, nthreads())
function main(filename::String)
    for i in 1:nthreads()
        EngArray[i] = MersenneTwister(i)
    end

    eng = EngArray[1]
    data_x = [rand(eng, [1.0, -1.0], c.NSpin)  for i in 1:c.NData]
    bimu = zeros(Float64, 2 * c.NData)
    biI  = Array(Diagonal(ones(Float64, 2 * c.NData))) ./ 2^c.NSpin
    biψ  = rand(MvNormal(bimu, biI))
    data_ψ = biψ[1:c.NData] .+ im * biψ[c.NData+1:end]
    data_ψ ./= norm(data_ψ)
    model = GPmodel(data_x, data_ψ, I)
    batch_x = [rand(eng, [1.0, -1.0], c.NSpin)  for i in 1:c.NMC]
    
    params = Parameters(I, zeros(Float64, c.NData))
    for k in 1:c.T
        for l in 1:100
            # Kernel update
            params = paramsupdate(params, ene, batch_x, model)
            model  = GPmodel(model, params)
            # Data update
            model = imaginarytime(model, params)
        end

        ene = expectedvalues(batch_x, model)
        β = k * l * c.Δτ
        open("./data/" * filename, "a") do io
            write(io, string(k))
            write(io, "\t")
            write(io, string(ene))
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
