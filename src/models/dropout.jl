using CUDA, Enzyme, Random, Zygote
using Zygote: @nograd
import Flux: testmode!, trainmode!

function s4d_real_dropout(x, mask, default_LogA_init)
    return mask .* x .+ (.!mask .* default_LogA_init')
end

function _s4d_real_dropout_kernel(x, mask, default_LogA_init)
    return mask .* x .+ (.!mask .* default_LogA_init')
end

@nograd function create_mask(p, size_x)
    return CUDA.rand(Float32, size_x) .> p
end

mutable struct S4DRealDropout
    p::Float32
    default_LogA_init::CuArray{Float32, 1}
    active::Union{Bool}
end

Flux.@layer S4DRealDropout trainable=()

testmode!(m::S4DRealDropout, mode=true) = (m.active = false)
trainmode!(m::S4DRealDropout, mode=true) = (m.active = true)

function S4DRealDropout(p, n; active=nothing)
    @assert 0 ≤ p ≤ 1
    default_LogA_init = CuArray(Float32.(log.(collect(1:n))))
    return S4DRealDropout(p, default_LogA_init, false)
end

function (m::S4DRealDropout)(x)
    if !m.active
        return x    
    end
    mask = create_mask(m.p, size(x))
    return s4d_real_dropout(x, mask, m.default_LogA_init)
end

#= n = 16
x = zeros(Float32, (4, n)) |> gpu_device()

dropout = S4DRealDropout(0.5, n)

testmode!(dropout)
y = dropout(x)
println(y)
trainmode!(dropout)
y = dropout(x)
println(y)
dx = Zygote.gradient(x -> sum(dropout(x)), x) =#


