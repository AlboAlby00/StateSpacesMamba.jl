using CUDA, Enzyme, Random, Zygote, Flux
using Zygote: @nograd
import Flux: testmode!, trainmode!

function s4d_real_dropout(x, mask, default_LogA_init)
    return mask .* x .+ (.!mask .* default_LogA_init')
end

function _s4d_real_dropout_kernel(x, mask, default_LogA_init)
    return mask .* x .+ (.!mask .* default_LogA_init')
end

@nograd function create_mask(p, size_x)
    return CUDA.rand(Float32, size_x) .>= p
end

mutable struct S4DRealDropout
    p::Float32
    default_LogA_init::CuArray{Float32, 1}
    active::Union{Bool}
end

Flux.@layer S4DRealDropout trainable=()

testmode!(m::S4DRealDropout, mode=true) = (m.active = !mode)

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
x_s4d = ones(Float32, (4, n)) |> gpu_device()
x = ones(Float32, (4, n))

dropout_s4d = S4DRealDropout(0.5, n)
dropout = Dropout(0.5)

testmode!(dropout_s4d)
testmode!(dropout)
y_s4d = dropout_s4d(x_s4d)
y = dropout(y)
println(y_s4d)
trainmode!(dropout_s4d)
trainmode!(dropout)
y_s4d = dropout_s4d(x_s4d)
y = dropout(y)
println(y_s4d)
println(y)
dx_s4d = Zygote.gradient(x -> sum(dropout_s4d(x)), x_s4d)
 =#

