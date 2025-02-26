using Test
using BenchmarkTools
using CUDA
using Plots
using Zygote

CUDA.allowscalar(false)

include("scan.jl")

K = 512
BLOCKS = 4
BATCH_SIZE = 32
N = 16
D = 256
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

function cuda_simple_ssm_forward!(X::CuDeviceArray{T}, A::CuDeviceArray{T},
	B::CuDeviceArray{T}, C::CuDeviceArray{T}, H::CuDeviceArray{T}) where T

	block_dim = blockDim().x

	tid = threadIdx().x

	# allocate memory
	h1 = CUDA.@cuDynamicSharedMem(T, block_dim)
	h2 = CUDA.@cuDynamicSharedMem(T, block_dim, block_dim * sizeof(T))
	c = CUDA.@cuDynamicSharedMem(T, block_dim, 2 * block_dim * sizeof(T))

	# load
	@inbounds h1[tid] = A[tid]
	@inbounds h2[tid] = B[tid] * X[tid]
	@inbounds c[tid] = C[tid]

	sync_threads()

	hillis_steele_scan!(h1, h2, ssm_associative_op, tid, blockDim().x)

	# store
	H[1, tid] = h1[tid]
	H[2, tid] = h2[tid]

	return nothing

end


function cuda_ssm_forward_for_each_block!(X::CuDeviceArray{T}, A::CuDeviceArray{T}, B::CuDeviceArray{T},
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

	# store
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

function cuda_ssm_forward(A, B, C, X, Y)

	H0 = get_H0(BLOCKS)
	H = CUDA.zeros(Float32, 2, BLOCKS)
	O = CUDA.ones(Float32, BLOCKS)

	CUDA.@sync @cuda threads = K blocks = BLOCKS shmem = (3 * sizeof(Float32) * K) cuda_ssm_forward_for_each_block!(X, A, B, C, H0, Y, H)
	H0 .= H

	CUDA.@sync @cuda threads = BLOCKS blocks = 1 shmem = (3 * sizeof(Float32) * BLOCKS) cuda_simple_ssm_forward!(H0[2, :], H0[1, :], O, O, H)

	H0 = get_H0(BLOCKS)
	H0[:, 2:end] .= H[:, 1:end-1]
	CUDA.@sync @cuda threads = K blocks = BLOCKS shmem = (3 * sizeof(Float32) * K) cuda_ssm_forward_for_each_block!(X, A, B, C, H0, Y, H)
end

X = Float32.(CuArray(1:SEQLEN))
X_cpu = Float32.(1:SEQLEN)
Y = CUDA.zeros(Float32, SEQLEN)

alpha = 0.5f0
A = CUDA.ones(Float32, SEQLEN) .* alpha
B = CUDA.ones(Float32, SEQLEN) .- alpha
C = CUDA.ones(Float32, SEQLEN)

cuda_ssm_forward(A, B, C, X, Y)

@assert isapprox(ema(X_cpu, alpha), Array(Y), rtol = 1e-2)

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
	H::CuDeviceArray{T}, Ḣ::CuDeviceArray{T};
	step::Step = SECOND, back = false) where T

	n = size(A, 2)
	d, l, b = size(X)

	nd_dim, block_dim, batch_dim = blockDim()
	nd_index, block_index, batch_index = blockIdx()

	n_index = (nd_index - 1) % n + 1
	d_index = (nd_index - 1) ÷ n + 1

	tid = threadIdx().y

	n_threads = nd_dim * block_dim * batch_dim

	l_index = (block_index - 1) * block_dim + threadIdx().y

	if l_index > l || n_index > n || d_index > d || batch_index > b
		return nothing
	end

	# allocate memory
	h1 = CUDA.@cuDynamicSharedMem(T, n_threads)
	h2 = CUDA.@cuDynamicSharedMem(T, n_threads, n_threads * sizeof(T))
	c = CUDA.@cuDynamicSharedMem(T, n_threads, 2 * n_threads * sizeof(T))

	ā, b̄ = discretize(A[d_index, n_index], B[n_index, l_index, batch_index], Δ[d_index, l_index, batch_index])

	# load
	@inbounds h1[tid] = ā
	@inbounds h2[tid] = b̄ * X[d_index, l_index, batch_index]
	@inbounds c[tid] = C[n_index, l_index, batch_index]

	if tid == 1
		h1[tid], h2[tid] = ssm_associative_op((h1[1], h2[1]), (H0[1, n_index, d_index, block_index, batch_index], H0[2, n_index, d_index, block_index, batch_index]))
	end

	sync_threads()

	# forward
	hillis_steele_scan!(h1, h2, ssm_associative_op, tid, block_dim)

	# store forward
	if step == FIRST
		@inbounds H[1, n_index, d_index, block_index, batch_index] = h1[tid]
		@inbounds H[2, n_index, d_index, block_index, batch_index] = h2[tid]
	elseif step == SECOND
		@inbounds CUDA.@atomic Y[d_index, l_index, batch_index] += h2[tid] * c[tid]

	else
		throw(ErrorException("Step is not correctly defined"))
	end

	if !back
		return nothing
	end

	return nothing

end

function cuda_scan(X, Δ, A, B, C; K = 128, BLOCKS = 64, N = 16, D = 16)
	Y = CUDA.zeros(Float32, D, SEQLEN, BATCH_SIZE)
	temp = reshape(get_H0(BLOCKS), 2, 1, 1, BLOCKS, 1)
	H0 = repeat(temp, 1, N, D, 1, BATCH_SIZE)
	H = CUDA.zeros(Float32, 2, N, D, BLOCKS, BATCH_SIZE)
	Ẋ, Δ̇, Ȧ, Ḃ, Ċ, Ẏ, H0̇, Ḣ = (similar(Z) .= 0 for Z in (X, Δ, A, B, C, Y, H0, H))
	CUDA.@sync @cuda threads = (1, K, 1) blocks = (N * D, BLOCKS, BATCH_SIZE) shmem = (3 * sizeof(Float32) * K) cuda_ssm!(X, Ẋ, A, Ȧ, B, Ḃ, C, Ċ, Δ, Δ̇, Y, Ẏ, H0, H0̇, H, Ḣ)
	return Y
end

X = CuArray(repeat(Float32.(1:SEQLEN)', D, 1, BATCH_SIZE))

alpha = 0.9f0
A = CUDA.rand(Float32, D, N) .* alpha
B = CUDA.rand(Float32, N, SEQLEN, BATCH_SIZE) .- alpha
C = CUDA.rand(Float32, N, SEQLEN, BATCH_SIZE)
Δ = CUDA.rand(Float32, D, SEQLEN, BATCH_SIZE) .* 5f0


@time Y = cuda_scan(X, Δ, A, B, C)

#@time Y_ = associative_selective_scan(X, Δ, A, B, C)

#println(Y[4, :, 4])
println("---")
#println(Y_[4, :, 4])
