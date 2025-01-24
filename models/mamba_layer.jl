module MambaLayer

using Flux
using CUDA
using OMEinsum
using Zygote

include("../scan/scan.jl")

# Structs definition

struct SSM
    LogA::AbstractArray
    project_x_to_Δ::Dense
    project_x_to_B::Dense
    project_x_to_C::Dense
    project_Δ::Dense # Project Δ from Δrank to D to have the correct size to perform discretization
end

struct MambaBlock
    norm::LayerNorm
    project_input::Dense
    project_res::Dense
    conv1d::Conv
    ssm::SSM
    project_output::Dense
end

struct MambaArgs
    input_dim::Int
    output_dim::Int
    mamba_block_output_dim::Int
    n_layers::Int
    N::Int
    D::Int
    kernel_size::Int
end


function MambaArgs(in, out; mamba_block_output_dim=16, n_layers=3, N=16, expand=16, kernel_size=5)
    @assert kernel_size % 2 == 1 "kernel size must be a odd integer"
    return MambaArgs(in, out, mamba_block_output_dim, n_layers, N, expand * in, kernel_size)
end

function Mamba(args::MambaArgs)
    model = Chain(
        MambaBlock(args.input_dim, args.mamba_block_output_dim, D=args.D, N=args.N),
        [MambaBlock(args.mamba_block_output_dim, args.mamba_block_output_dim, D=args.D, N=args.N) for _ in 1:args.n_layers-1]...,
        Dense(args.mamba_block_output_dim => args.output_dim)
    )
    return model
end

# Constructor for MambaBlock
function MambaBlock(input_dim, output_dim; D=64, N=16, kernel_size=5) # IMPORTANT! kernel_size must be odd, otherwise conv1d input D != output D
    norm = LayerNorm(input_dim)
    project_input = Dense(input_dim => D)
    project_res = Dense(input_dim => D)
    conv1d = Conv((kernel_size,), D => D, relu, pad=(Int((kernel_size - 1) / 2),)) # pad defined so that input D == output D
    ssm = SSM(D=D, N=N, Δrank=max(1, D ÷ 2))
    project_output = Dense(D => output_dim)
    return MambaBlock(norm, project_input, project_res, conv1d, ssm, project_output)
end

# Forward pass for MambaBlock
function (m::MambaBlock)(x)
    #out_norm = m.norm(x)
    out_project = m.project_input(x)
    # Conv layer requires size l, d, b but out_project size is d, l, b. We need to permute the dims
    out_project = permutedims(out_project, (2, 1, 3)) # permute l and d
    out_conv = swish(m.conv1d(out_project))
    out_conv = permutedims(out_conv, (2, 1, 3)) # permute d and l to return to original size

    out_ssm = swish(m.ssm(out_conv))

    # residual connection
    out_res = out_ssm .+ swish(m.project_res(x))

    out = m.project_output(out_res)
    return out
end

Flux.@layer MambaBlock trainable = (project_input, project_res, conv1d, ssm, project_output)

# Forward pass for SSM
function (m::SSM)(x)
    A = -exp.(m.LogA)
    B = m.project_x_to_B(x)
    C = m.project_x_to_C(x)
    Δ = softplus(m.project_Δ(relu(m.project_x_to_Δ(x))))
    y = associative_selective_scan(x, Δ, A, B, C)
    return y
end

# Constructor for SSM
function SSM(; D::Int, N::Int, Δrank::Int)
    LogA = log.((repeat(1:N, 1, D)'))
    project_x_to_Δ = Dense(D => Δrank)
    project_x_to_B = Dense(D => N)
    project_x_to_C = Dense(D => N)
    project_Δ = Dense(Δrank => D)
    return SSM(LogA, project_x_to_Δ, project_x_to_B, project_x_to_C, project_Δ)
end

Flux.@layer SSM trainable = (LogA, project_x_to_Δ, project_x_to_B, project_x_to_C, project_Δ)

end  # End of module
