using Test
using BenchmarkTools
using CUDA
using Plots
using Zygote
using ChainRulesCore

CUDA.allowscalar(false)

include("scan.jl")

function hillis_steele_scan!(t1, t2, op, tid, block_dim, reversed)
	offset = 1
	while offset < block_dim
		t1_temp, t2_temp = t1[tid], t2[tid]
		sync_threads()
		if reversed
			if tid + offset <= block_dim
				t1_temp, t2_temp = op((t1_temp, t2_temp), (t1[tid+offset], t2[tid+offset]))
			end
		else
			if tid > offset
				t1_temp, t2_temp = op((t1_temp, t2_temp), (t1[tid-offset], t2[tid-offset]))
			end
		end
		t1[tid], t2[tid] = t1_temp, t2_temp
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

function discretize_back(a, b, Δ, dā, db̄)
	Δa = Δ * a
	ā = exp(Δa)

	da_da = Δ * ā
	da_dΔ = a * ā

	db_db = Δ
	db_dΔ = b

	return dā * da_da, db̄ * db_db, dā * da_dΔ + db̄ * db_dΔ
end

function discretize(a, b, Δ)
	Δa = Δ * a
	ā = exp(Δa)
	b̄ = b * Δ
	return ā, b̄
end

@enum Step FIRST SECOND

function cuda_ssm!(
	X::CuDeviceArray{T}, dX::CuDeviceArray{T},
	A::CuDeviceArray{T}, dA::CuDeviceArray{T},
	B::CuDeviceArray{T}, dB::CuDeviceArray{T},
	C::CuDeviceArray{T}, dC::CuDeviceArray{T},
	Δ::CuDeviceArray{T}, dΔ::CuDeviceArray{T},
	Y::CuDeviceArray{T}, dY::CuDeviceArray{T},
	H0::CuDeviceArray{T}, dH0::CuDeviceArray{T},
	H::CuDeviceArray{T}, dH::CuDeviceArray{T},
	step, back) where {T}

	n = size(A, 2)
	d, l, b = size(X)

	nd_dim, block_dim, batch_dim = blockDim()
	nd_index, block_index, batch_index = blockIdx()
	_, blocks, _ = gridDim()

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
	dh1 = CUDA.@cuDynamicSharedMem(T, n_threads, 2 * n_threads * sizeof(T))
	dh2 = CUDA.@cuDynamicSharedMem(T, n_threads, 3 * n_threads * sizeof(T))
	shared_A = CUDA.@cuDynamicSharedMem(T, 1, 4 * n_threads * sizeof(T))

	# Load A into shared memory and intitialize h once per block
	if tid == 1
		shared_A[1] = A[d_index, n_index]
	end
	sync_threads()

	# Forward pass
	a = shared_A[1]
	b = B[n_index, l_index, batch_index]
	delta = Δ[d_index, l_index, batch_index]
	ā, b̄ = discretize(a, b, delta)
	c = C[n_index, l_index, batch_index]
	x = X[d_index, l_index, batch_index]

	# Load data
	@inbounds h1[tid] = ā
	@inbounds h2[tid] = b̄ * x

	# Set first h value using H0 data
	if tid == 1
		h0_1 = H0[1, n_index, d_index, block_index, batch_index]
		h0_2 = H0[2, n_index, d_index, block_index, batch_index]
		h1[1], h2[1] = ssm_associative_op((h1[1], h2[1]), (h0_1, h0_2))
	end

	# Perform scan
	hillis_steele_scan!(h1, h2, ssm_associative_op, tid, block_dim, false)

	# Store results
	if step == FIRST && tid == 1
		@inbounds H[1, n_index, d_index, block_index, batch_index] = h1[end]
		@inbounds H[2, n_index, d_index, block_index, batch_index] = h2[end]
	elseif step == SECOND
		@inbounds CUDA.@atomic Y[d_index, l_index, batch_index] += h2[tid] * c
	end

	# Backward pass

	if !back
		return
	end

	dy = dY[d_index, l_index, batch_index]

	if tid == block_dim || l_index + 1 > l
		delta_shifted = 0
	else
		delta_shifted = Δ[d_index, l_index+1, batch_index]
	end
	a_shifted, _ = discretize(a, d, delta_shifted)

	@inbounds dh1[tid] = a_shifted
	@inbounds dh2[tid] = c * dy

	# Set initial dh value using dH0 data
	if tid == block_dim
		@inbounds dh0_1 = dH0[1, n_index, d_index, block_index, batch_index]
		@inbounds dh0_2 = dH0[2, n_index, d_index, block_index, batch_index]
		@inbounds dh1[end], dh2[end] = ssm_associative_op((dh1[end], dh2[end]), (dh0_1, dh0_2))
	end

	sync_threads()

	scan_array_dim = l_index <= (blocks - 1) * block_dim ? block_dim : l - (blocks - 1) * block_dim

	hillis_steele_scan!(dh1, dh2, ssm_associative_op, tid, scan_array_dim, true)

	if step == FIRST && tid == 1
		dH[1, n_index, d_index, block_index, batch_index] = dh1[1]
		dH[2, n_index, d_index, block_index, batch_index] = dh2[1]
	elseif step == SECOND
		CUDA.@atomic dC[n_index, l_index, batch_index] += h2[tid] * dy
		CUDA.@atomic dX[d_index, l_index, batch_index] += b̄ * dh2[tid]

		db̄ = dh2[tid] * x

		if tid > 1
			previous_h = h2[tid-1]
		else
			previous_h = H0[2, n_index, d_index, block_index, batch_index]
		end
		dā = dh2[tid] * previous_h

		da, db, ddelta = discretize_back(a, b, delta, dā, db̄)

		CUDA.@atomic dA[d_index, n_index] += da
		CUDA.@atomic dB[n_index, l_index, batch_index] += db
		CUDA.@atomic dΔ[d_index, l_index, batch_index] += ddelta

	end

	return nothing
end

function reduce!(M::CuDeviceArray{T}, M0::CuDeviceArray{T}, reversed) where {T}
	tid = threadIdx().y
	nd_index, block_index, batch_index = blockIdx().x, blockIdx().y, blockIdx().z
	_, n, d, _, b = size(M)
	n_index = (nd_index - 1) % n + 1
	d_index = (nd_index - 1) ÷ n + 1

	if n_index > n || d_index > d || batch_index > b
		return nothing
	end

	t1 = CUDA.@cuDynamicSharedMem(T, blockDim().y)
	t2 = CUDA.@cuDynamicSharedMem(T, blockDim().y, blockDim().y * sizeof(T))

	@inbounds t1[tid] = M[1, n_index, d_index, tid, batch_index]
	@inbounds t2[tid] = M[2, n_index, d_index, tid, batch_index]

	sync_threads()

	hillis_steele_scan!(t1, t2, ssm_associative_op, tid, blockDim().y, reversed)

	if tid > 1 && !reversed
		@inbounds M0[1, n_index, d_index, tid, batch_index] = t1[tid-1]
		@inbounds M0[2, n_index, d_index, tid, batch_index] = t2[tid-1]
	elseif tid > 1 && reversed
		@inbounds M0[1, n_index, d_index, tid-1, batch_index] = t1[tid]
		@inbounds M0[2, n_index, d_index, tid-1, batch_index] = t2[tid]
	end

	return nothing
end

# K is the number of threads per block
function cuda_scan(X, Δ, A, B, C; K = 512, compute_backward = false, w = nothing)
	d, l, b = size(X)
	n = size(A, 2)
	blocks = ceil(Int, l / K)
	H0, dH0 = get_H0(blocks, b, n, d), get_H0(blocks, b, n, d)
	H = CUDA.zeros(Float32, 2, n, d, blocks, b)
	Y = CUDA.zeros(Float32, d, l, b)
	dY = isnothing(w) ? CUDA.ones(size(Y)) : w

	dX, dA, dB, dC, dΔ, dH = CUDA.zeros(size(X)), CUDA.zeros(size(A)), CUDA.zeros(size(B)), CUDA.zeros(size(C)), CUDA.zeros(size(Δ)), CUDA.zeros(size(H))
	shmem_size = (4 * K + 1) * sizeof(Float32)

	CUDA.@cuda threads = (1, K, 1) blocks = (n * d, blocks, b) shmem = shmem_size cuda_ssm!(
		X, dX, A, dA, B, dB, C, dC, Δ, dΔ, Y, dY, H0, dH0, H, dH, FIRST, compute_backward)

	CUDA.@cuda threads = (1, blocks, 1) blocks = (n * d, 1, b) shmem = 2 * blocks * sizeof(Float32) reduce!(H, H0, false)
	CUDA.@cuda threads = (1, blocks, 1) blocks = (n * d, 1, b) shmem = 2 * blocks * sizeof(Float32) reduce!(dH, dH0, true)

	CUDA.@cuda threads = (1, K, 1) blocks = (n * d, blocks, b) shmem = shmem_size cuda_ssm!(
		X, dX, A, dA, B, dB, C, dC, Δ, dΔ, Y, dY, H0, dH0, H, dH, SECOND, compute_backward)

	return compute_backward ? (Y, (dX, dΔ, dA, dB, dC)) : Y
end

function ChainRulesCore.rrule(::typeof(cuda_scan), X, Δ, A, B, C; K = 512)

	Y = cuda_scan(X, Δ, A, B, C; K = K, compute_backward = false)

	function cuda_scan_pullback(w)
		Y, (dX, dΔ, dA, dB, dC) = cuda_scan(X, Δ, A, B, C; K = K, compute_backward = true, w = w)
		return (NoTangent(), dX, dΔ, dA, dB, dC)
	end

	return Y, cuda_scan_pullback
end


#= K = 512
BATCH_SIZE = 32
N = 16
D = 32
SEQLEN = 2873

X = CuArray(repeat(Float32.(1:SEQLEN)', D, 1, BATCH_SIZE))

alpha = 0.1f0
A = CUDA.rand(Float32, D, N) .* alpha
B = CUDA.rand(Float32, N, SEQLEN, BATCH_SIZE) .- alpha
C = CUDA.rand(Float32, N, SEQLEN, BATCH_SIZE)
Δ = CUDA.rand(Float32, D, SEQLEN, BATCH_SIZE) .* 0.01f0

# check forward is equivalent
Q = associative_selective_scan(X, Δ, A, B, C)
Y = cuda_scan(X, Δ, A, B, C; compute_backward=false, K=K)

alpha = 0.1f0
A = CUDA.rand(Float32, D, N) .* alpha
B = CUDA.rand(Float32, N, SEQLEN, BATCH_SIZE) .- alpha
C = CUDA.rand(Float32, N, SEQLEN, BATCH_SIZE)
Δ = CUDA.rand(Float32, D, SEQLEN, BATCH_SIZE) .* 0.01f0

CUDA.@time Q = associative_selective_scan(X, Δ, A, B, C)
CUDA.@time Y = cuda_scan(X, Δ, A, B, C; compute_backward=false, K=K)
@assert isapprox(Array(Q), Array(Y), rtol=0.1)

println("---")

# check backward is equivalent
associative_selective_scan_loss = (X, Δ, A, B, C) -> sum(associative_selective_scan(X, Δ, A, B, C))
Yg = Zygote.gradient(associative_selective_scan_loss, X, Δ, A, B, C)
CUDA.@time  Yg = Zygote.gradient(associative_selective_scan_loss, X, Δ, A, B, C)

cuda_scan_loss = (X, Δ, A, B, C) -> sum(cuda_scan(X, Δ, A, B, C; K=K))
Qg = Zygote.gradient(cuda_scan_loss, X, Δ, A, B, C)
CUDA.@time  Qg = Zygote.gradient(cuda_scan_loss, X, Δ, A, B, C)

for (S, T) in zip(Yg, Qg)
	@assert isapprox(Array(S), Array(T), rtol=0.1)
end

println("---") =#
