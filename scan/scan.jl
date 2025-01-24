using OMEinsum
using Flux
using GPUArrays
import Base: promote_op, add_sum, cumsum!

include("utils.jl")

# op(x, y) = x + y
# GPUArrays.neutral_element(::typeof(op), ::Type{T}) where T = zero(T)

# Scan function for SSM, unstable implementation 
# SSM equations are:
# h′ = Ā * h + B̄ * x
# y = C̄ * h′
function selective_scan(x, Δ, A, B, C)
    d, l, b = size(x)
    n = size(A, 2)
    
    # discretization
    @ein log_Ā[d, n, l, b] := Δ[d, l, b] * A[d, n]
    Ā = exp.(log_Ā)
    @ein B̄x[d, n, l, b] := Δ[d, l, b] * x[d, l, b] * B[n, l, b]

    # scan
    y_stack = Zygote.Buffer(copy(x))
    h = CuArray(zeros(d, n, b))
    for t in 1:l
        @ein h[d, n, b] := Ā[d, n, t, b] * h[d, n, b]
        h = h .+ B̄x[:, :, t, :]
        @ein y[d, b] := h[d, n, b] * C[n, t, b]
        y_stack[:, t, :] = y
    end
    return copy(y_stack)
end

function associative_selective_scan(x, Δ, A, B, C; mode=:logcumsumexp)

    # discretization
    @ein log_Ā[d, n, l, b] := Δ[d, l, b] * A[d, n]
    Ā = exp.(log_Ā)
    @ein B̄x[d, n, l, b] := Δ[d, l, b] * x[d, l, b] * B[n, l, b]

    if mode == :cumsum
        Ā_cumsum = cumsum(Ā, dims=3) # cumulative sums on the l dimension
        temp = B̄x ./ (Ā_cumsum .+ 1e-12)
        h′ = cumsum(temp, dims=3) .* Ā_cumsum
        @ein y[d, l, b] := h′[d, n, l, b] * C[n, l, b]
    elseif mode == :logcumsumexp
        B̄x_log = complex_log(B̄x)
        log_Ā_cumsum = cumsum(log_Ā, dims=3) # cumulative sums on the l dimension
        h_log = logcumsumexp(B̄x_log - log_Ā_cumsum, dims=3) + log_Ā_cumsum
        h = exp.(real.(h_log))
        @ein y[d, l, b] := h[d, n, l, b] * C[n, l, b]
    end

    return y
end
