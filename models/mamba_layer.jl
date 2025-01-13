module MambaLayer

using Flux
using CUDA
using OMEinsum

struct SSM
    A::CuArray{Float64}
    project_x_to_Δ::Dense
    project_x_to_B::Dense
    project_x_to_C::Dense
    project_Δ::Dense # Project Δ from Δrank to D to have the correct size to perform discretization
end

# Forward pass for SSM
function (m::SSM)(x)
    A = m.A
    B = m.project_x_to_B(x)
    C = m.project_x_to_C(x)
    Δ = m.project_Δ(softplus(m.project_x_to_Δ(x)))
    y = selective_scan(x, Δ, A, B, C)
    return y  # Add more logic here if needed
end

# Constructor for MambaBlock
function SSM(; D::Int, N::Int, Δrank::Int)
    A = CuArray{Float64}(repeat(1:N, 1, D)') # Example initialization for A
    project_x_to_Δ = Dense(D => Δrank)
    project_x_to_B = Dense(D => N)
    project_x_to_C = Dense(D => N)
    project_Δ = Dense(Δrank => D)
    return SSM(A, project_x_to_Δ, project_x_to_B, project_x_to_C, project_Δ)
end

Flux.@layer SSM trainable=(A, project_x_to_Δ, project_x_to_B, project_x_to_C, project_Δ)

function complex_log(input::AbstractArray, eps::Float64 = 1e-12)
    real_part = log.(max.(abs.(input), eps))
    imaginary_part = pi .* (input .< 0)
    return complex.(real_part, imaginary_part)
end
# Scan function for SSM
# SSM equations are:
# h′ = Ā * h + B̄ * x
# y = C̄ * h′
function selective_scan(x, Δ, A, B, C)
    # discretization
    @ein ΔA[d, n, l, b] := Δ[d, l, b] * A[d, n]
    Ā = exp.(ΔA)
    @ein B̄x[d, n, l, b] := Δ[d, l, b] * x[d, l, b] * B[n, l, b]
	# scan
	Ā_cumsum = cumsum(Ā, dims=3) # cumulative sums on the l dimension
	temp = B̄x ./ (Ā_cumsum .+ 1e-12)
	h′ = cumsum(temp, dims=3) .* Ā_cumsum
	@ein y[d, l, b] := h′[d, n , l, b] * C[n, l, b]
	return y

end

struct MambaBlock
	ssm::SSM
	conv1d::Conv
	project_input::Dense
	project_res::Dense
	project_output::Dense
	norm::LayerNorm
end

function MambaBlock(;)
	
end

# Forward pass for MambaBlock
function (m::MambaBlock)(x)
	out_norm = m.norm(x)
    out_project = m.project_input(out_norm)
	out_conv = swish(m.conv1d(out_project))
	out_ssm = m.ssm(out_conv)

	# residual connection
	out_res = out_ssm + swish(m.project_res(x))

	out = m.project_output(out_res)
    return out 
end

Flux.@layer MambaBlock trainable=(ssm, conv1d, project_input, project_output)

end  # End of module
