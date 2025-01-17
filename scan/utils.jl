using Base: cumsum!

function complex_log(input::AbstractArray, eps::Float64=1e-12)
    real_part = log.(max.(abs.(input), eps))
    imaginary_part = pi * (input .< 0)
    complex.(real_part, imaginary_part)
end

function mylogcumsumexp(A::AbstractArray{T}; dims::Integer) where T
    out = similar(A, promote_op(add_sum, T, T))
    cumsum!(out, A, dims=dims)
    return out
end

function discretize(x, Δ, A, B)
    @ein ΔA[d, n, l, b] := Δ[d, l, b] * A[d, n]
    Ā = exp.(ΔA)
    @ein B̄x[d, n, l, b] := Δ[d, l, b] * x[d, l, b] * B[n, l, b]
    return Ā, B̄x
end