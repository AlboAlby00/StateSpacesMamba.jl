using Test
using BenchmarkTools
using CUDA
using Plots

K = 32
BLOCKS = 1
SEQLEN = K * BLOCKS

function ema(x, alpha)
	y = []
	h = 0
	for k in eachindex(x)
		h = alpha * h + (1 - alpha) * x[k]
		push!(y, h)
	end
	return y
end

function hillis_steele_scan!(x, op, tid)

	offset = 1

	while offset < blockDim().x
		temp = x[tid]  # Preserve current value
		if tid > offset
			temp = op(temp, x[tid-offset])
		end
		x[tid] = temp  # Update shared memory
        sync_threads()
        offset *= 2
	end

end

function ssm_associative_op(a, b)
	return (a[1] * b[1], b[1] * a[2] + b[2], b[3])
end

function cuda_simple_ssm(X::CuDeviceVector{T}, A::CuDeviceVector{T},
	B::CuDeviceVector{T}, C::CuDeviceVector{T}, Y::CuDeviceVector{T}) where T

	tid = threadIdx().x
	block_dim = blockDim().x
	kid = (blockIdx().x - 1) * block_dim + tid

	# t[1] is A, 
	t = CuDynamicSharedArray(NTuple{3, T}, block_dim)

	t[tid] = (A[kid], B[kid] * X[kid], C[kid])

	sync_threads()

	hillis_steele_scan!(t, ssm_associative_op, tid)

    @cuprintln(t[tid][3])

	Y[kid] = t[tid][2] * t[tid][3]

	return nothing

end

X = Float32.(CuArray(1:SEQLEN))
Y = CUDA.zeros(Float32, SEQLEN)

alpha = 0.5f0
A = CUDA.ones(Float32, SEQLEN) .* alpha
B = CUDA.ones(Float32, SEQLEN) .- alpha
C = CUDA.ones(Float32, SEQLEN)

shared_memory = (sizeof(NTuple{3, Float32})) * K
@cuda threads = K blocks = BLOCKS shmem = shared_memory cuda_simple_ssm(X, A, B, C, Y)
println(Y)
ema(1:K,alpha)
