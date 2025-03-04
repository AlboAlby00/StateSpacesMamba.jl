using Test
using BenchmarkTools
using CUDA
using Plots
using Zygote

CUDA.allowscalar(false)

include("scan.jl")

K = 4
BLOCKS = 1
BATCH_SIZE = 4
N = 16
D = 16
SEQLEN = K * BLOCKS

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
        sync_threads()
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

function discretize_back(a, b, d, da_, db_)
    da = d * a
    a_ = exp(da)

    da_da = d * a_
    da_dΔ = a * a_

    db_db = d
    db_dΔ = b

    return da_ * da_da, db_ * db_db, da_ * da_dΔ + db_ * db_dΔ
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
    Δ::CuDeviceArray{T}, dΔ::CuDeviceArray{T},
    Y::CuDeviceArray{T}, Ẏ::CuDeviceArray{T},
    H0::CuDeviceArray{T}, dH0::CuDeviceArray{T},
    H::CuDeviceArray{T}, Ḣ::CuDeviceArray{T},
    step, back) where {T}

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

    # Load data
    @inbounds h1[tid] = ā
    @inbounds h2[tid] = b̄ * X[d_index, l_index, batch_index]

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
        H[1, n_index, d_index, block_index, batch_index] = h1[end]
        H[2, n_index, d_index, block_index, batch_index] = h2[end]
    elseif step == SECOND
        @inbounds CUDA.@atomic Y[d_index, l_index, batch_index] += h2[tid] * c
    end

    # Backward pass

    if !back
        return
    end

    ẏ = Ẏ[d_index, l_index, batch_index]

    if tid == block_dim
        delta_shifted = 0
    else
        delta_shifted = Δ[d_index, l_index+1, batch_index]
    end
    a_shifted, _ = discretize(a, d, delta_shifted)

    @inbounds dh1[tid] = a_shifted
    @inbounds dh2[tid] = c * ẏ

    # Set first (for reverse) dh value using dH0 data
    if tid == 1
        dh0_1 = dH0[1, n_index, d_index, block_index, batch_index]
        dh0_2 = dH0[2, n_index, d_index, block_index, batch_index]
        dh1[end], dh2[end] = ssm_associative_op((dh1[end], dh2[end]), (dh0_1, dh0_2))
    end

    hillis_steele_scan!(dh1, dh2, ssm_associative_op, tid, block_dim, reversed=true)

    if step == FIRST
        Ḣ[1, n_index, d_index, block_index, batch_index] = dh1[1]
        Ḣ[2, n_index, d_index, block_index, batch_index] = dh2[1]
    elseif step == SECOND
        CUDA.@atomic Ċ[n_index, l_index, batch_index] += h2[tid] * ẏ
        CUDA.@atomic Ẋ[d_index, l_index, batch_index] += b̄ * dh2[tid]
		# since h2[tid] = b̄ * X[d_index, l_index, batch_index] and since we don't store x in shared memory, we can avoid accessing X in global memory by doing
		# x = current_h / b̄
		current_h = h2[tid]
		db̄ = dh2[tid] * (current_h / b̄) 
		
		# computing h at previous timestamp repeating again the forward pass, since it is less computationally expensive than storing the values in memory
		if tid > 1
			previous_h = h2[tid-1]
		else
			previous_h = H[2, n_index, d_index, block_index, batch_index]
		end
		dā = dh2[tid] * previous_h

		ȧ, ḃ, ddelta = discretize_back(a, b, delta, dā, db̄)

		CUDA.@atomic Ȧ[d_index, n_index] += ȧ
        CUDA.@atomic Ḃ[d_index, l_index, batch_index] += ḃ
		CUDA.@atomic dΔ[d_index, l_index, batch_index] += ddelta

    end

    return nothing
end

function reduce!(M::CuDeviceArray{T}, reversed) where {T}
    tid = threadIdx().y
    nd_index, block_index, batch_index = blockIdx().x, blockIdx().y, blockIdx().z
    n = size(M, 2)
    n_index = (nd_index - 1) % n + 1
    d_index = (nd_index - 1) ÷ n + 1

    t1 = CUDA.@cuDynamicSharedMem(T, blockDim().y)
    t2 = CUDA.@cuDynamicSharedMem(T, blockDim().y, blockDim().y * sizeof(T))

    @inbounds t1[tid] = M[1, n_index, d_index, tid, batch_index]
    @inbounds t2[tid] = M[2, n_index, d_index, tid, batch_index]
    sync_threads()

    hillis_steele_scan!(t1, t2, ssm_associative_op, tid, blockDim().y, reversed)

    @inbounds M[1, n_index, d_index, tid, batch_index] = t1[tid]
    @inbounds M[2, n_index, d_index, tid, batch_index] = t2[tid]
    return nothing
end

function cuda_scan(X, Δ, A, B, C; K=128)
    d, l, b = size(X)
    blocks = l ÷ K
    n = size(A, 2)
    H0 = get_H0(blocks, b, n, d)
    H = CUDA.zeros(Float32, 2, n, d, blocks, b)
    Y = CUDA.zeros(Float32, d, l, b)

    Ẋ, Ȧ, Ḃ, Ċ, dΔ, dH0, Ḣ = CUDA.zeros(size(X)), CUDA.zeros(size(A)), CUDA.zeros(size(B)), CUDA.zeros(size(C)), CUDA.zeros(size(Δ)), CUDA.zeros(size(H0)), CUDA.zeros(size(H))
    Ẏ = CUDA.ones(size(Y))

    shmem_size = (4 * K + 1) * sizeof(Float32)
    CUDA.@sync CUDA.@cuda threads = (1, K, 1) blocks = (n * d, blocks, b) shmem = shmem_size cuda_ssm!(
        X, Ẋ, A, Ȧ, B, Ḃ, C, Ċ, Δ, dΔ, Y, Ẏ, H0, dH0, H, Ḣ, FIRST, true)

    CUDA.@sync CUDA.@cuda threads = (1, blocks, 1) blocks = (n * d, 1, b) shmem = 2 * blocks * sizeof(Float32) reduce!(H, false)
    H0[:, :, :, 2:end, :] .= H[:, :, :, 1:end-1, :]

	CUDA.@sync CUDA.@cuda threads = (1, blocks, 1) blocks = (n * d, 1, b) shmem = 2 * blocks * sizeof(Float32) reduce!(Ḣ, true)
	dH0[:, :, :, 1:end-1, :] .= Ḣ[:, :, :, 2:end, :]

    CUDA.@sync CUDA.@cuda threads = (1, K, 1) blocks = (n * d, blocks, b) shmem = shmem_size cuda_ssm!(
        X, Ẋ, A, Ȧ, B, Ḃ, C, Ċ, Δ, dΔ, Y, Ẏ, H0, dH0, H, Ḣ, SECOND, true)

    return Y, (Ẋ, dΔ, Ȧ, Ḃ, Ċ)
end

X = CuArray(repeat(Float32.(1:SEQLEN)', D, 1, BATCH_SIZE))

alpha = 0.1f0
A = CUDA.rand(Float32, D, N) .* alpha
B = CUDA.rand(Float32, N, SEQLEN, BATCH_SIZE) .- alpha
C = CUDA.rand(Float32, N, SEQLEN, BATCH_SIZE)
Δ = CUDA.rand(Float32, D, SEQLEN, BATCH_SIZE) .* 0.01f0

#CUDA.@time Q = associative_selective_scan(X, Δ, A, B, C)
loss = (X, Δ, A, B, C) -> sum(associative_selective_scan(X, Δ, A, B, C))
CUDA.@time Qg = Zygote.gradient(loss, X, Δ, A, B, C)

CUDA.@time Y, Yg = cuda_scan(X, Δ, A, B, C, K=K)

#@assert isapprox(Array(Q), Array(Y), rtol = 0.1)

println(Qg[4][1, :, 1])
println(Yg[4][1, :, 1])

#@assert isapprox(Array(Qg[2]), Array(Yg[2]), rtol=0.1)

#= for (S, T) in zip(Yg, Qg)
	TEST = CUDA.zeros(size(S))
	@assert isapprox(Array(S), Array(TEST), rtol = 0.1)
end =#

println("---")
