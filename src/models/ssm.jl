using Flux
using CUDA
using Zygote

include("dropout.jl")
include("../scan/cuda_scan.jl")
include("../scan/scan.jl")


struct SSM
    LogA::AbstractArray
    project_x_to_Δ::Dense
    project_x_to_B::Dense
    project_x_to_C::Dense
    project_Δ::Dense # Project Δ from Δrank to D in order to have the correct size to perform discretization
    dropout::Dropout
    logA_dropout::Union{Dropout, S4DRealDropout}
    scan::Function
end

# Forward pass for SSM
function (m::SSM)(x)
    A = -exp.(m.logA_dropout(m.LogA))
    B = m.dropout(m.project_x_to_B(x))
    C = m.dropout(m.project_x_to_C(x))
    Δ = softplus(m.project_Δ(swish(m.project_x_to_Δ(x))))
    y = m.scan(x, Δ, A, B, C)
    return y
end

# Constructor for SSM
function SSM(; D::Int, N::Int, Δrank::Int, ssm_dropout=0.0, use_A_dropout = true, use_cuda_scan = true)
    LogA = log.((repeat(1:N, 1, D)'))
    project_x_to_Δ = Dense(D => Δrank)
    project_x_to_B = Dense(D => N)
    project_x_to_C = Dense(D => N)
    project_Δ = Dense(Δrank => D)
    dropout = Dropout(ssm_dropout)
    logA_dropout = use_A_dropout ? S4DRealDropout(ssm_dropout, N) : Dropout(ssm_dropout)
    scan = use_cuda_scan ? cuda_scan : associative_selective_scan
    return SSM(LogA, project_x_to_Δ, project_x_to_B, project_x_to_C, project_Δ, dropout, logA_dropout, scan)
end

Flux.@layer SSM trainable = (LogA, project_x_to_Δ, project_x_to_B, project_x_to_C, project_Δ)

