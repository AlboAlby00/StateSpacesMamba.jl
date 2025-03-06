function complex_log(input::AbstractArray, eps::Float64=1e-12)
    real_part = log.(max.(abs.(input), eps))
    imaginary_part = pi * (input .< 0)
    complex.(real_part, imaginary_part)
end

function logsumexp_op(x, y)
    max_val = max(real(x), real(y))
    log(exp(x - max_val) + exp(y - max_val)) + max_val
end

function logcumsumexp(A::AbstractArray; dims::Integer)
    accumulate(logsumexp_op, A, dims=dims)
end

function rrule(::typeof(logcumsumexp), x::AbstractArray{T,N}; dims::Integer) where {T,N}
    function logcumsumexp_pullback(dy)
        project = ProjectTo(x)
        x_max = maximum(real(x); dims=dims)
        exp_x = exp.(x)
        cumsum_exp = cumsum(exp.(x .- x_max), dims=dims) .* exp.(x_max)
        res1 = reverse(dy, dims=dims) ./ reverse(cumsum_exp, dims=dims)
        res2 = reverse(cumsum(res1, dims=dims), dims=dims) .* exp_x
        return (NoTangent(), project(res2))
    end
    return logcumsumexp(x; dims=dims), logcumsumexp_pullback
end
rrule(::typeof(logcumsumexp), x::AbstractArray{T,N})  where {T,N} = rrule(logcumsumexp, x; dims=1)
