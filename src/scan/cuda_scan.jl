using Test
using BenchmarkTools
using CUDA
using Plots
using Zygote

CUDA.allowscalar(false)

include("scan.jl")

K = 512
BLOCKS = 32
BATCH_SIZE = 32
N = 16
D = 32
SEQLEN = K * BLOCKS

function hillis_steele_scan!(h1, h2, op, tid, block_dim; reversed::Bool = false)
	offset = 1
	while offset < block_dim
		h1_temp, h2_temp = h1[tid], h2[tid]
		if reversed
			if tid + offset <= block_dim
				h1_temp, h2_temp = op((h1_temp, h2_temp), (h1[tid+offset], h2[tid+offset]))
			end
		else
			if tid > offset
				h1_temp, h2_temp = op((h1_temp, h2_temp), (h1[tid-offset], h2[tid-offset]))
			end
		end
		h1[tid], h2[tid] = h1_temp, h2_temp
		sync_threads()
		offset *= 2
	end
end

function ssm_associative_op(a, b)
	return (a[1] * b[1], a[1] * b[2] + a[2])
end

function get_H0(blocks::Int, batch_size, n, d)
	arr = CUDA.zeros(Float32, (2, blocks))
	arr[1, :] .= 1.0f0
	temp = reshape(arr, 2, 1, 1, blocks, 1)
	H0 = repeat(temp, 1, n, d, 1, batch_size)
	return H0
end

function discretize(a, b, Δ)
	Δa = Δ * a
	ā = exp(Δa)
	b̄ = b * Δ
	return ā, b̄
end

@enum Step FIRST SECOND

function cuda_ssm!(
	X::CuDeviceArray{T}, Ẋ::CuDeviceArray{T},
	A::CuDeviceArray{T}, Ȧ::CuDeviceArray{T},
	B::CuDeviceArray{T}, Ḃ::CuDeviceArray{T},
	C::CuDeviceArray{T}, Ċ::CuDeviceArray{T},
	Δ::CuDeviceArray{T}, Δ̇::CuDeviceArray{T},
	Y::CuDeviceArray{T}, Ẏ::CuDeviceArray{T},
	H0::CuDeviceArray{T}, H0̇::CuDeviceArray{T},
	H::CuDeviceArray{T}, Ḣ::CuDeviceArray{T},
	step, back) where T

	n = size(A, 2)
	d, l, b = size(X)

	nd_dim, block_dim, batch_dim = blockDim()
	nd_index, block_index, batch_index = blockIdx()

	n_index = (nd_index - 1) % n + 1
	d_index = (nd_index - 1) ÷ n + 1

	tid = threadIdx().y
	n_threads = nd_dim * block_dim * batch_dim
	l_index = (block_index - 1) * block_dim + tid

	if l_index > l || n_index > n || d_index > d || batch_index > b
		return nothing
	end

	# Allocate shared memory
	h1 = CUDA.@cuDynamicSharedMem(T, n_threads)
	h2 = CUDA.@cuDynamicSharedMem(T, n_threads, n_threads * sizeof(T))
	c = CUDA.@cuDynamicSharedMem(T, n_threads, 2 * n_threads * sizeof(T))
	shared_A = CUDA.@cuDynamicSharedMem(T, 1, 3 * n_threads * sizeof(T))

	# Load A into shared memory and intitialize h once per block
	if tid == 1
		shared_A[1] = A[d_index, n_index]
	end
	sync_threads()
	a = shared_A[1]

	delta = Δ[d_index, l_index, batch_index]
	b_val = B[n_index, l_index, batch_index]
	ā, b̄ = discretize(a, b_val, delta)

	# Load data
	@inbounds h1[tid] = ā
	@inbounds h2[tid] = b̄ * X[d_index, l_index, batch_index]
	@inbounds c[tid] = C[n_index, l_index, batch_index]

	# Set first h value using H0 data
	if tid == 1
		h0_1 = H0[1, n_index, d_index, block_index, batch_index]
		h0_2 = H0[2, n_index, d_index, block_index, batch_index]
		h1[1], h2[1] = ssm_associative_op((h1[1], h2[1]), (h0_1, h0_2))
	end

	sync_threads()

	# Perform scan
	hillis_steele_scan!(h1, h2, ssm_associative_op, tid, block_dim)

	# Store results
	if step == FIRST && tid == 1
		H[1, n_index, d_index, block_index, batch_index] = h1[end]
		H[2, n_index, d_index, block_index, batch_index] = h2[end]
	elseif step == SECOND
		@inbounds CUDA.@atomic Y[d_index, l_index, batch_index] += h2[tid] * c[tid]
	end

	return nothing
end

function reduce!(H::CuDeviceArray{T}) where T
	tid = threadIdx().y
	nd_index, block_index, batch_index = blockIdx().x, blockIdx().y, blockIdx().z
	n = size(H, 2)
	n_index = (nd_index - 1) % n + 1
	d_index = (nd_index - 1) ÷ n + 1

	h1_shared = CUDA.@cuDynamicSharedMem(T, blockDim().y)
	h2_shared = CUDA.@cuDynamicSharedMem(T, blockDim().y, blockDim().y * sizeof(T))

	@inbounds h1_shared[tid] = H[1, n_index, d_index, tid, batch_index]
	@inbounds h2_shared[tid] = H[2, n_index, d_index, tid, batch_index]
	sync_threads()

	hillis_steele_scan!(h1_shared, h2_shared, ssm_associative_op, tid, blockDim().y)

	@inbounds H[1, n_index, d_index, tid, batch_index] = h1_shared[tid]
	@inbounds H[2, n_index, d_index, tid, batch_index] = h2_shared[tid]
	return nothing
end

function cuda_scan!(X, Δ, A, B, C; K = 128)
	d, l, b = size(X)
	blocks = l ÷ K
	n = size(A, 2)
	H0 = get_H0(blocks, b, n, d)
	H = CUDA.zeros(Float32, 2, n, d, blocks, b)
	Y = CUDA.zeros(Float32, d, l, b)

	shmem_size = (3 * K + 1) * sizeof(Float32)
	CUDA.@sync @cuda threads = (1, K, 1) blocks = (n * d, blocks, b) shmem = shmem_size cuda_ssm!(
		X, similar(X), A, similar(A), B, similar(B), C, similar(C), Δ, similar(Δ),
		Y, similar(Y), H0, similar(H0), H, similar(H), FIRST, false,
	)
	CUDA.@sync @cuda threads = (1, blocks, 1) blocks = (n * d, 1, b) shmem = 2 * blocks * sizeof(Float32) reduce!(H)
	H0[:, :, :, 2:end, :] .= H[:, :, :, 1:end-1, :]
	CUDA.@sync @cuda threads = (1, K, 1) blocks = (n * d, blocks, b) shmem = shmem_size cuda_ssm!(
		X, similar(X), A, similar(A), B, similar(B), C, similar(C), Δ, similar(Δ),
		Y, similar(Y), H0, similar(H0), H, similar(H), SECOND, false,
	)
	return Y
end

X = CuArray(repeat(Float32.(1:SEQLEN)', D, 1, BATCH_SIZE))

alpha = 0.9f0
A = CUDA.ones(Float32, D, N) .* alpha
B = CUDA.ones(Float32, N, SEQLEN, BATCH_SIZE) .- alpha
C = CUDA.ones(Float32, N, SEQLEN, BATCH_SIZE)
Δ = CUDA.ones(Float32, D, SEQLEN, BATCH_SIZE) .* 0.01f0

@time Q = associative_selective_scan(X, Δ, A, B, C)
@time Y = cuda_scan!(X, Δ, A, B, C, K = K)

@assert isapprox(Array(Q), Array(Y), rtol = 0.1)

println("---")
