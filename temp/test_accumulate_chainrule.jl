using Base: promote_op, add_sum
using ChainRulesCore: rrule

function logsumexp_op(x, y)
    max_val = if abs(x) > abs(y)
        x
    else
        y
    end
    log(exp(x - max_val) + exp(y - max_val)) + max_val
end

function logcumsumexp(A::AbstractArray{T}; dims::Integer) where T
    accumulate(logsumexp_op, A, dims=dims)
end

A = [5.0, 1.0, 5.0]
dims = 1

# Forward function
#Y = logcumsumexp(A, dims=dims)
#println(Y)
#println(sum(Y))
# Compute gradient

gradient_A = Zygote.jacobian(x -> cumsum(x, dims=dims), A)
println(gradient_A)