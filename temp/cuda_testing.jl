using Test
using BenchmarkTools
using CUDA
using Plots

CUDA.allowscalar(false)

K = 32
BLOCKS = 16
SEQLEN = K * BLOCKS

x = 1:SEQLEN
y = zeros(SEQLEN)

function seq_cumsum(x)
	y = []
	h = 0
	for k in eachindex(x)
		h = h + x[k]
		push!(y, h)
	end
	return h, y
end

h_, y_ = seq_cumsum(x)

# @btime seq_cumsum($x)

# bar(x, y_, xlabel="Index", ylabel="Value", title="Bar Plot", legend=false)

op = (x, y) -> x .+ y

x_d = CuArray(1:SEQLEN)
y_d = CUDA.zeros(SEQLEN)
function cuda_cumsum_1!(x, y)
	CUDA.scan!(+, y, x; dims = 1)
end

#cuda_cumsum_1!(x_d, y_d)
#@assert y_ == Array(y_d)

function hillis_steele_scan!(x, op, tid)

	offset = 1
	while offset < blockDim().x
		temp = x[tid]  # Preserve current value
		if tid > offset
			temp = op(temp, x[tid-offset])
		end
		x[tid] = temp  # Update shared memory
		offset *= 2
		sync_threads()
	end

end


function cuda_cumsum_for_each_block!(
	X::CuDeviceVector{T}, Y::CuDeviceVector{T},
	H0::CuDeviceVector{T}, H::CuDeviceVector{T}) where T

    tid = threadIdx().x
    block_dim = blockDim().x
	kid = (blockIdx().x - 1) * block_dim + tid
	block_index = blockIdx().x

	x = CuDynamicSharedArray(T, block_dim)
	@inbounds x[tid] = X[kid]

	if tid == 1
		@inbounds x[1] += H0[block_index]
	end

	sync_threads()

	hillis_steele_scan!(x, +, tid)

	@inbounds Y[kid] = x[tid]

	if tid == 1
		@inbounds H[block_index] = x[end]
	end

	sync_threads()

	return nothing
end

function cuda_cumsum!(X, Y, k, blocks)
	H0 = CUDA.zeros(Float32, blocks)
	H = CUDA.zeros(Float32, blocks)
	@cuda threads = k blocks = blocks shmem = (k * sizeof(Float32)) cuda_cumsum_for_each_block!(X, Y, H0, H0)
	H[2:end] = cumsum(H0)[1:end-1]
	@cuda threads = k blocks = blocks shmem = (k * sizeof(Float32)) cuda_cumsum_for_each_block!(X, Y, H, H)

	return nothing
end


x_d = Float32.(CuArray(1:SEQLEN))
y_d = CUDA.zeros(Float32, SEQLEN)

# cuda_cumsum!(x_d, y_d, K, BLOCKS)

#= println(CUDA.@profile cumsum(x_d))
println("-----------------")
CUDA.@profile cuda_cumsum!(x_d, y_d, K, BLOCKS) =#

# @btime cuda_cumsum!(x_d, y_d, K, BLOCKS)

cuda_cumsum!(x_d, y_d, K, BLOCKS)
#@assert cumsum(x_d) ≈ y_d
println(y_d)


