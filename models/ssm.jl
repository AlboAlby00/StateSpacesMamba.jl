using Flux
using CUDA
using Zygote

struct SSM
    LogA::AbstractArray
    project_x_to_Δ::Dense
    project_x_to_B::Dense
    project_x_to_C::Dense
    project_Δ::Dense # Project Δ from Δrank to D in order to have the correct size to perform discretization
end

# Forward pass for SSM
function (m::SSM)(x)
    A = -exp.(m.LogA)
    B = m.project_x_to_B(x)
    C = m.project_x_to_C(x)
    Δ = softplus(m.project_Δ(swish(m.project_x_to_Δ(x))))
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

