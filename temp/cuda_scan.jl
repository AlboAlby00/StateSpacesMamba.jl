using Test
using BenchmarkTools
using CUDA
using Plots
using Zygote
CUDA.allowscalar(false)

K = 128 # 4 => 21 # 1 => 3 # 2 => 6
BLOCKS = 1024
SEQLEN = K * BLOCKS

function ema(x, alpha)
	y = Zygote.Buffer(x)
	h = zero(eltype(x))
	for k in eachindex(x)
		h = alpha * h + (1 - alpha) * x[k]
		y[k] = h
	end
	return copy(y)
end

function hillis_steele_scan!(h1, h2, op, tid, block_dim)

	offset = 1
	while offset < block_dim
		h1_temp, h2_temp = h1[tid], h2[tid]
		if tid > offset
			h1_temp, h2_temp = op((h1_temp, h2_temp), (h1[tid-offset], h2[tid-offset]))
		end
		h1[tid], h2[tid] = h1_temp, h2_temp
		sync_threads()
		offset *= 2
	end

end

function ssm_associative_op(a, b)
	return (a[1] * b[1], a[1] * b[2] + a[2])
end

function cuda_simple_ssm!(X::CuDeviceArray{T}, A::CuDeviceArray{T},
	B::CuDeviceArray{T}, C::CuDeviceArray{T}, H::CuDeviceArray{T}) where T

	block_dim = blockDim().x

	tid = threadIdx().x

	h1 = CUDA.@cuDynamicSharedMem(T, block_dim)
	h2 = CUDA.@cuDynamicSharedMem(T, block_dim, block_dim * sizeof(T))
	c = CUDA.@cuDynamicSharedMem(T, block_dim, 2 * block_dim * sizeof(T))

    # load
    @inbounds h1[tid] = A[tid]
    @inbounds h2[tid] = B[tid] * X[tid]
    @inbounds c[tid] = C[tid]

    sync_threads()

    hillis_steele_scan!(h1, h2, ssm_associative_op, tid, blockDim().x)

    H[1,tid] = h1[tid]
    H[2,tid] = h2[tid]

	return nothing

end


function cuda_ssm_for_each_block!(X::CuDeviceArray{T}, A::CuDeviceArray{T}, B::CuDeviceArray{T},
	C::CuDeviceArray{T}, H0::CuDeviceArray{T}, Y::CuDeviceArray{T}, H::CuDeviceArray{T}) where T

	tid = threadIdx().x
	block_dim = blockDim().x
	block_index = blockIdx().x
	kid = (block_index - 1) * block_dim + tid

	# allocate
	h1 = CUDA.@cuDynamicSharedMem(T, block_dim)
	h2 = CUDA.@cuDynamicSharedMem(T, block_dim, block_dim * sizeof(T))
	c = CUDA.@cuDynamicSharedMem(T, block_dim, 2 * block_dim * sizeof(T))

	# load
	@inbounds h1[tid] = A[kid]
	@inbounds h2[tid] = B[kid] * X[kid]
	@inbounds c[tid] = C[kid]

	sync_threads()

	if tid == 1
		h1[tid], h2[tid] = ssm_associative_op((h1[1], h2[1]), (H0[1, block_index], H0[2, block_index]))
	end

	hillis_steele_scan!(h1, h2, ssm_associative_op, tid, block_dim)

	if tid == 1
		@inbounds H[1, block_index] = h1[end]
		@inbounds H[2, block_index] = h2[end]
	end

	@inbounds Y[kid] = h2[tid][2] * c[tid]

	return nothing
end

function get_H0(blocks::Int)
	arr = CUDA.zeros(Float32, (2, blocks))
	arr[1, :] .= 1.0f0
	return arr
end


X = Float32.(CuArray(1:SEQLEN))
X_cpu = Float32.(1:SEQLEN)
Y = CUDA.zeros(Float32, SEQLEN)

alpha = 0.5f0
A = CUDA.ones(Float32, SEQLEN) .* alpha
B = CUDA.ones(Float32, SEQLEN) .- alpha
C = CUDA.ones(Float32, SEQLEN)
H0 = get_H0(BLOCKS)
H = CUDA.zeros(Float32, 2, BLOCKS)
O = CUDA.ones(Float32, BLOCKS)


CUDA.@sync @cuda threads = K blocks = BLOCKS shmem = (3 * sizeof(Float32) * K) cuda_ssm_for_each_block!(X, A, B, C, H0, Y, H)
H0 .= H

CUDA.@sync @cuda threads = BLOCKS blocks = 1 shmem = (3 * sizeof(Float32) * BLOCKS) cuda_simple_ssm!(H0[2, :], H0[1, :], O, O, H)
#CUDA.@sync @cuda threads = K blocks = BLOCKS shmem = (3 * sizeof(Float32) * K) cuda_ssm_for_each_block!(X, A, B, C, H0, Y, H)

H0 = get_H0(BLOCKS)
H0[:, 2:end] .= H[:, 1:end-1]
CUDA.@sync @cuda threads = K blocks = BLOCKS shmem = (3 * sizeof(Float32) * K) cuda_ssm_for_each_block!(X, A, B, C, H0, Y, H)

@assert isapprox(ema(X_cpu, alpha), Array(Y), rtol=1e-2)
